from splatflow.train.gsplat_simple_trainer import entrypoint
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple


@dataclasses.dataclass
class GsplatCommandSettings:
    """Type-safe builder for hloc processing commands."""

    dataset_dir: Path  # data-dir
    output_dir: Path  # result-dir

    # Downsample factor for the dataset
    data_factor: int = 4

    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None

    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Normalize the world space
    normalize_world_space: bool = True

    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # TODO: Add this to global settings
    # # Port for the viewer server
    # port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # This will be controlled by step scaler
    # # Number of training steps
    # max_steps: int = 30_000

    # we always save
    # # Whether to save ply file (storage size can be large)
    # save_ply: bool = False

    # Initialization strategy
    init_type: Literal["sfm", "random"] = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0

    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    # strategy: Union[DefaultStrategy, MCMCStrategy] = field(
    #     default_factory=DefaultStrategy
    # )

    # TODO: Disabled for now
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    # packed: bool = False
    # # Use sparse gradients for optimization. (experimental)
    # sparse_grad: bool = False
    # # Use visible adam from Taming 3DGS. (experimental)
    # visible_adam: bool = False

    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = True

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = True
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6

    # This is only for testing.
    # # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    # pose_noise: float = 0.0

    # This is disabled for now
    # TODO: Enable
    # Enable appearance optimization. (experimental)
    # app_opt: bool = False
    # # Appearance embedding dimension
    # app_embed_dim: int = 16
    # # Learning rate for appearance optimization
    # app_opt_lr: float = 1e-3
    # # Regularization for appearance optimization as weight decay
    # app_opt_reg: float = 1e-6

    # TODO: Enable
    # # Enable bilateral grid. (experimental)
    # use_bilateral_grid: bool = False
    # # Shape of the bilateral grid (X, Y, W)
    # bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # TODO: Enable depths
    # # Enable depth loss. (experimental)
    # depth_loss: bool = False
    # # Weight for depth loss
    # depth_lambda: float = 1e-2

    # Not used at the moment
    # 3DGUT (uncented transform + eval 3D)
    # with_ut: bool = False
    # with_eval3d: bool = False

    # I don't think this will work at the moment
    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_dir": str(self.dataset_dir),
            "output_dir": str(self.output_dir),
        }

    def build(self) -> List[str]:
        """Build the CLI command list."""
        cmd = [
            "poetry",
            "run",
            "train",
            "default",
            "--data-dir",
            self.dataset_dir,
            "--result-dir",
            self.output_dir,
            "--save-ply",
            "--steps-scaler",
            "0.1",
        ]

        return cmd


def main():
    entrypoint()
