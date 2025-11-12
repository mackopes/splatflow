from splatflow.train.gsplat_simple_trainer import entrypoint
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union


@dataclasses.dataclass
class GsplatCommandSettings:
    """Type-safe builder for hloc processing commands."""

    dataset_dir: Path  # data-dir
    output_dir: Path  # result-dir

    # Strategy for GS densification
    strategy: Literal["default", "mcmc"] = dataclasses.field(default="default")

    # Downsample factor for the dataset
    data_factor: int = 4

    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

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
    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

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
            "strategy": self.strategy,
            "data_factor": self.data_factor,
            "steps_scaler": self.steps_scaler,
            "patch_size": self.patch_size,
            "global_scale": self.global_scale,
            "normalize_world_space": self.normalize_world_space,
            "camera_model": self.camera_model,
            "batch_size": self.batch_size,
            "init_type": self.init_type,
            "init_num_pts": self.init_num_pts,
            "init_extent": self.init_extent,
            "sh_degree": self.sh_degree,
            "sh_degree_interval": self.sh_degree_interval,
            "init_opa": self.init_opa,
            "init_scale": self.init_scale,
            "ssim_lambda": self.ssim_lambda,
            "near_plane": self.near_plane,
            "far_plane": self.far_plane,
            "antialiased": self.antialiased,
            "random_bkgd": self.random_bkgd,
            "means_lr": self.means_lr,
            "scales_lr": self.scales_lr,
            "opacities_lr": self.opacities_lr,
            "quats_lr": self.quats_lr,
            "sh0_lr": self.sh0_lr,
            "shN_lr": self.shN_lr,
            "opacity_reg": self.opacity_reg,
            "scale_reg": self.scale_reg,
            "pose_opt": self.pose_opt,
            "pose_opt_lr": self.pose_opt_lr,
            "pose_opt_reg": self.pose_opt_reg,
            "app_opt": self.app_opt,
            "app_embed_dim": self.app_embed_dim,
            "app_opt_lr": self.app_opt_lr,
            "app_opt_reg": self.app_opt_reg,
            "use_fused_bilagrid": self.use_fused_bilagrid,
        }

    def build(self) -> List[str]:
        """Build the CLI command list."""
        cmd = [
            "poetry",
            "run",
            "train",
            self.strategy,
            "--data-dir",
            str(self.dataset_dir),
            "--result-dir",
            str(self.output_dir),
        ]

        # Add all optional parameters
        cmd.extend(["--data-factor", str(self.data_factor)])
        cmd.extend(["--steps-scaler", str(self.steps_scaler)])

        if self.patch_size is not None:
            cmd.extend(["--patch-size", str(self.patch_size)])

        cmd.extend(["--global-scale", str(self.global_scale)])

        if self.normalize_world_space:
            cmd.append("--normalize-world-space")
        else:
            cmd.append("--no-normalize-world-space")

        cmd.extend(["--camera-model", self.camera_model])
        cmd.extend(["--batch-size", str(self.batch_size)])
        cmd.extend(["--init-type", self.init_type])
        cmd.extend(["--init-num-pts", str(self.init_num_pts)])
        cmd.extend(["--init-extent", str(self.init_extent)])
        cmd.extend(["--sh-degree", str(self.sh_degree)])
        cmd.extend(["--sh-degree-interval", str(self.sh_degree_interval)])
        cmd.extend(["--init-opa", str(self.init_opa)])
        cmd.extend(["--init-scale", str(self.init_scale)])
        cmd.extend(["--ssim-lambda", str(self.ssim_lambda)])
        cmd.extend(["--near-plane", str(self.near_plane)])
        cmd.extend(["--far-plane", str(self.far_plane)])

        if self.antialiased:
            cmd.append("--antialiased")
        else:
            cmd.append("--no-antialiased")

        if self.random_bkgd:
            cmd.append("--random-bkgd")
        else:
            cmd.append("--no-random-bkgd")

        cmd.extend(["--means-lr", str(self.means_lr)])
        cmd.extend(["--scales-lr", str(self.scales_lr)])
        cmd.extend(["--opacities-lr", str(self.opacities_lr)])
        cmd.extend(["--quats-lr", str(self.quats_lr)])
        cmd.extend(["--sh0-lr", str(self.sh0_lr)])
        cmd.extend(["--shN-lr", str(self.shN_lr)])
        cmd.extend(["--opacity-reg", str(self.opacity_reg)])
        cmd.extend(["--scale-reg", str(self.scale_reg)])

        if self.pose_opt:
            cmd.append("--pose-opt")
        else:
            cmd.append("--no-pose-opt")

        cmd.extend(["--pose-opt-lr", str(self.pose_opt_lr)])
        cmd.extend(["--pose-opt-reg", str(self.pose_opt_reg)])

        if self.app_opt:
            cmd.append("--app-opt")
        else:
            cmd.append("--no-app-opt")

        cmd.extend(["--app-embed-dim", str(self.app_embed_dim)])
        cmd.extend(["--app-opt-lr", str(self.app_opt_lr)])
        cmd.extend(["--app-opt-reg", str(self.app_opt_reg)])

        if self.use_fused_bilagrid:
            cmd.append("--use-fused-bilagrid")
        else:
            cmd.append("--no-use-fused-bilagrid")

        # Always save ply files
        cmd.append("--save-ply")

        return cmd


def main():
    entrypoint()
