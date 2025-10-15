"""Load initial points and colors from COLMAP sparse reconstruction for Gaussian Splatting initialization."""

import pathlib
import numpy as np
from plyfile import PlyData

from splatflow.colmap_utils import ColmapTransforms


def load_points_from_transforms(
    transforms_path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load 3D points and RGB colors from the PLY file referenced in transforms.json.

    Args:
        transforms_path: Path to the transforms.json file

    Returns:
        points: (N, 3) array of xyz coordinates
        points_rgb: (N, 3) array of RGB colors (0-255)
    """
    # Load transforms
    transforms = ColmapTransforms.load(transforms_path)

    # Resolve PLY path relative to transforms.json directory
    transforms_dir = transforms_path.parent
    ply_path = transforms_dir / transforms.ply_file_path

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    # Load PLY file
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]

    # Extract xyz coordinates
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
        np.float32
    )

    # Extract RGB colors
    points_rgb = np.stack(
        [vertex["red"], vertex["green"], vertex["blue"]], axis=1
    ).astype(np.uint8)

    return points, points_rgb
