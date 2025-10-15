"""Visualize camera poses and point cloud to debug coordinate system issues."""

import pathlib
import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData
from splatflow.colmap_utils import ColmapTransforms


def create_camera_frustum(c2w, size=0.1, color="red", name="camera"):
    """Create a camera frustum from camera-to-world matrix.

    Args:
        c2w: 4x4 camera-to-world transformation matrix
        size: Size of the frustum
        color: Color of the frustum
        name: Name for the trace
    """
    # Camera frustum vertices in camera space
    # Looking down -Z axis (OpenGL/NeRF convention)
    vertices = np.array(
        [
            [0, 0, 0],  # camera center
            [-size, -size, -size * 2],  # bottom-left
            [size, -size, -size * 2],  # bottom-right
            [size, size, -size * 2],  # top-right
            [-size, size, -size * 2],  # top-left
        ]
    )

    # Transform to world space
    vertices_world = (c2w[:3, :3] @ vertices.T + c2w[:3, 3:4]).T

    # Define edges of the frustum
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),  # from center to corners
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),  # around the image plane
    ]

    # Create line segments
    x, y, z = [], [], []
    for i, j in edges:
        x.extend([vertices_world[i, 0], vertices_world[j, 0], None])
        y.extend([vertices_world[i, 1], vertices_world[j, 1], None])
        z.extend([vertices_world[i, 2], vertices_world[j, 2], None])

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color=color, width=2),
        name=name,
        showlegend=True,
    )


def visualize_scene(transforms_path: pathlib.Path, use_inverted: bool = False):
    """Visualize cameras and point cloud.

    Args:
        transforms_path: Path to transforms.json
        use_inverted: If True, invert the transform matrices (treat as W2C instead of C2W)
    """
    # Load transforms
    transforms = ColmapTransforms.load(transforms_path)

    # Load point cloud
    transforms_dir = transforms_path.parent
    ply_path = transforms_dir / transforms.ply_file_path

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]

    # Extract point cloud
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
        np.float32
    )
    colors_rgb = np.stack(
        [vertex["red"], vertex["green"], vertex["blue"]], axis=1
    ).astype(np.uint8)

    # Subsample points for visualization (if too many)
    if len(points) > 50000:
        indices = np.random.choice(len(points), 50000, replace=False)
        points = points[indices]
        colors_rgb = colors_rgb[indices]

    # Create plotly figure
    fig = go.Figure()

    # Plot point cloud
    colors_hex = [f"rgb({r},{g},{b})" for r, g, b in colors_rgb]
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=1, color=colors_hex),
            name="Point Cloud",
            showlegend=True,
        )
    )
    applied_transform = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Plot cameras
    for i, frame in enumerate(transforms.frames):
        transform = frame.transform_matrix

        if use_inverted:
            # Treat as W2C and invert to get C2W
            c2w = np.linalg.inv(transform)
            color = "blue"
            name = f"Camera {i} (inverted)"
        else:
            # Treat as C2W directly
            c2w = applied_transform @ transform
            color = "red"
            name = f"Camera {i} (direct)"

        frustum = create_camera_frustum(c2w, size=0.2, color=color, name=name)
        fig.add_trace(frustum)

    # Plot camera centers
    camera_centers = []
    for frame in transforms.frames:
        transform = frame.transform_matrix
        if use_inverted:
            c2w = np.linalg.inv(transform)
        else:
            c2w = applied_transform @ transform
        camera_centers.append(c2w[:3, 3])

    camera_centers = np.array(camera_centers)
    fig.add_trace(
        go.Scatter3d(
            x=camera_centers[:, 0],
            y=camera_centers[:, 1],
            z=camera_centers[:, 2],
            mode="markers",
            marker=dict(size=4, color="yellow" if use_inverted else "orange"),
            name="Camera Centers",
            showlegend=True,
        )
    )

    # Compute scene bounds
    all_points = np.vstack([points, camera_centers])
    center = all_points.mean(axis=0)
    extent = np.abs(all_points - center).max()

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[center[0] - extent, center[0] + extent]),
            yaxis=dict(range=[center[1] - extent, center[1] + extent]),
            zaxis=dict(range=[center[2] - extent, center[2] + extent]),
            aspectmode="cube",
        ),
        title=f"Camera Visualization {'(Matrices Inverted)' if use_inverted else '(Direct)'}",
        showlegend=True,
        width=1200,
        height=800,
    )

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize cameras and point cloud")
    parser.add_argument(
        "transforms_path", type=pathlib.Path, help="Path to transforms.json"
    )
    parser.add_argument(
        "--invert", action="store_true", help="Invert transform matrices (treat as W2C)"
    )
    parser.add_argument(
        "--both", action="store_true", help="Show both direct and inverted"
    )

    args = parser.parse_args()

    if args.both:
        # Create two figures side by side
        fig1 = visualize_scene(args.transforms_path, use_inverted=False)
        fig2 = visualize_scene(args.transforms_path, use_inverted=True)
        fig1.show()
        fig2.show()
    else:
        fig = visualize_scene(args.transforms_path, use_inverted=args.invert)
        fig.show()
