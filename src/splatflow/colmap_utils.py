import pycolmap
import pathlib
import numpy as np
import json
import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class ColmapTransformsFrame:
    file_path: pathlib.Path
    transform_matrix: np.ndarray
    colmap_image_id: int


@dataclasses.dataclass
class ColmapTransforms:
    width: int
    height: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    camera_model: pycolmap.CameraModelId
    applied_transform: np.ndarray
    ply_file_path: pathlib.Path
    k1: Optional[float] = None
    k2: Optional[float] = None
    p1: Optional[float] = None
    p2: Optional[float] = None

    frames: List[ColmapTransformsFrame] = dataclasses.field(default_factory=list)

    def save(self, path: pathlib.Path) -> None:
        """Save ColmapTransforms to a JSON file."""
        transforms_dict = {
            "w": self.width,
            "h": self.height,
            "fl_x": self.fl_x,
            "fl_y": self.fl_y,
            "cx": self.cx,
            "cy": self.cy,
            "camera_model": self.camera_model.name,
            "applied_transform": self.applied_transform.tolist(),
            "ply_file_path": str(self.ply_file_path),
            "frames": [],
        }

        # Add optional distortion parameters
        if self.k1 is not None:
            transforms_dict["k1"] = self.k1
        if self.k2 is not None:
            transforms_dict["k2"] = self.k2
        if self.p1 is not None:
            transforms_dict["p1"] = self.p1
        if self.p2 is not None:
            transforms_dict["p2"] = self.p2

        # Convert frames to dict format
        for frame in self.frames:
            frame_dict = {
                "file_path": str(frame.file_path),
                "transform_matrix": frame.transform_matrix.tolist(),
                "colmap_im_id": frame.colmap_image_id,
            }
            transforms_dict["frames"].append(frame_dict)

        # Write to JSON file
        with open(path, "w") as f:
            json.dump(transforms_dict, f, indent=4)

    @classmethod
    def load(cls, path: pathlib.Path) -> "ColmapTransforms":
        """Load ColmapTransforms from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Parse frames
        frames = []
        for frame_data in data["frames"]:
            frame = ColmapTransformsFrame(
                file_path=pathlib.Path(frame_data["file_path"]),
                transform_matrix=np.array(frame_data["transform_matrix"]),
                colmap_image_id=frame_data["colmap_im_id"],
            )
            frames.append(frame)

        # Convert camera model string to CameraModelId
        camera_model = data["camera_model"]

        # Parse applied_transform and ply_file_path
        applied_transform = np.array(data["applied_transform"])
        ply_file_path = pathlib.Path(data["ply_file_path"])

        # Create ColmapTransforms instance
        return cls(
            width=data["w"],
            height=data["h"],
            fl_x=data["fl_x"],
            fl_y=data["fl_y"],
            cx=data["cx"],
            cy=data["cy"],
            camera_model=camera_model,
            applied_transform=applied_transform,
            ply_file_path=ply_file_path,
            k1=data.get("k1"),
            k2=data.get("k2"),
            p1=data.get("p1"),
            p2=data.get("p2"),
            frames=frames,
        )


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

    # COLMAP to NeRF/3DGS coordinate system transform
    # COLMAP: +X right, +Y down, +Z forward
    # NeRF/3DGS: +X right, +Y up, +Z backward
    applied_transform = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],  # Flip Y
            [0, 0, 1, 0],  # Flip Z
            [0, 0, 0, 1],
        ]
    )

    frames: List[ColmapTransformsFrame] = []

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

        transform_matrix = np.linalg.inv(transform_matrix)

        # Apply coordinate system transformation
        transform_matrix = applied_transform @ transform_matrix

        transform_matrix[:3, 2] *= -1

        frame = ColmapTransformsFrame(
            pathlib.Path(f"../input/{image.name}"), transform_matrix, image_id
        )
        frames.append(frame)

    # transforms["applied_transform"] = applied_transform.tolist()

    ply_path = processed_dataset_dir / "sparse_pc.ply"
    reconstruction.export_PLY(str(ply_path))

    output_path = processed_dataset_dir / "transforms.json"

    transforms = ColmapTransforms(
        width=camera.width,
        height=camera.height,
        fl_x=params[0],
        fl_y=params[1],
        cx=params[2],
        cy=params[3],
        camera_model=camera.model,
        applied_transform=applied_transform,
        ply_file_path=ply_path,
        frames=frames,
    )

    if len(params) > 4:
        transforms.k1 = params[4]
        transforms.k2 = params[5]
        transforms.p1 = params[6]
        transforms.p2 = params[7]

    transforms.save(output_path)

    print(f"Created transforms.json with {len(transforms.frames)} frames")

    return transforms
