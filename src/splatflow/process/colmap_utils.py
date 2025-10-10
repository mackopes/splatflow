import pycolmap
import pathlib
import numpy as np
import json


def create_transforms_json(processed_dataset_dir: pathlib.Path):
    colmap_dir = processed_dataset_dir / "sparse" / "0"

    if not colmap_dir.is_dir():
        raise ValueError(
            f"{colmap_dir} is not a valid directory. Has the dataset been processed?"
        )
    reconstruction = pycolmap.Reconstruction(str(colmap_dir))

    # Get the first camera (assuming single camera model)
    camera_id = list(reconstruction.cameras.keys())[0]
    camera = reconstruction.cameras[camera_id]

    # Extract camera intrinsics
    # camera.params contains: [fx, fy, cx, cy, k1, k2, p1, p2] for OPENCV model
    params = camera.params

    transforms = {
        "w": camera.width,
        "h": camera.height,
        "fl_x": params[0],  # fx
        "fl_y": params[1],  # fy
        "cx": params[2],  # cx
        "cy": params[3],  # cy
        "camera_model": camera.model.name,  # e.g., "OPENCV"
        "frames": [],
    }

    # Add distortion parameters if available
    if len(params) > 4:
        transforms["k1"] = params[4]
        transforms["k2"] = params[5]
        transforms["p1"] = params[6]
        transforms["p2"] = params[7]

    # COLMAP to NeRF/3DGS coordinate system transform
    # COLMAP: +X right, +Y down, +Z forward
    # NeRF/3DGS: +X right, +Y up, +Z backward
    applied_transform = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # Flip Y
            [0, 0, -1, 0],  # Flip Z
            [0, 0, 0, 1],
        ]
    )

    for image_id, image in reconstruction.images.items():
        image_id: int
        image: pycolmap.Image

        # Get camera-to-world transformation (already in the correct form)
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation  # 3x3 rotation matrix (camera-to-world)
        t = cam_from_world.translation  # 3x1 translation vector (camera-to-world)

        # Create 4x4 transformation matrix (camera-to-world)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R.matrix()
        transform_matrix[:3, 3] = t

        # Apply coordinate system transformation
        transform_matrix = applied_transform @ transform_matrix

        frame = {
            "file_path": f"../input/{image.name}",
            "transform_matrix": transform_matrix.tolist(),
            "colmap_im_id": image_id,
        }
        transforms["frames"].append(frame)

    transforms["applied_transform"] = applied_transform.tolist()

    # Export point cloud as PLY
    ply_path = processed_dataset_dir / "sparse_pc.ply"
    reconstruction.export_PLY(str(ply_path))
    transforms["ply_file_path"] = "sparse_pc.ply"

    # Write transforms.json
    output_path = processed_dataset_dir / "transforms.json"
    with open(output_path, "w") as f:
        json.dump(transforms, f, indent=4)

    print(f"Created transforms.json with {len(transforms['frames'])} frames")
    return transforms
