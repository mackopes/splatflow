"""Standalone viewer script for trained Gaussian Splatting models."""

import argparse
import pathlib
import sys
import time
import torch
import viser
import numpy as np

from splatflow.train.gsplat_viewer import GsplatViewer, GsplatRenderTabState
from splatflow.train.colmap import TransformsDataset
from splatflow.train.utils import CameraOptModule, AppearanceOptModule, set_random_seed
from nerfview import CameraState, RenderTabState, apply_float_colormap

from gsplat.rendering import rasterization
from typing import Tuple, Optional, Dict, Literal
from torch import Tensor


class ModelViewer:
    """Viewer for a trained Gaussian Splatting model."""

    def __init__(
        self,
        checkpoint_path: pathlib.Path,
        data_dir: pathlib.Path,
        port: int = 8080,
    ):
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.port = port
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        set_random_seed(42)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load dataset metadata to get scene scale
        path_to_transforms = data_dir / "transforms.json"
        if not path_to_transforms.exists():
            raise FileNotFoundError(
                f"transforms.json not found at {path_to_transforms}"
            )

        trainset = TransformsDataset(path_to_transforms, patch_size=None, factor=4)

        # Compute scene scale from camera positions
        camera_positions = []
        for frame in trainset.transforms.frames:
            w2c = frame.transform_matrix
            cam_pos = w2c[:3, 3]
            camera_positions.append(cam_pos)
        camera_positions = np.array(camera_positions)
        scene_center = np.mean(camera_positions, axis=0)
        dists = np.linalg.norm(camera_positions - scene_center, axis=1)
        computed_scene_scale = np.max(dists)
        self.scene_scale = computed_scene_scale * 1.1

        print(f"Scene scale: {self.scene_scale:.4f}")

        # Load splats from checkpoint
        self.splats = torch.nn.ParameterDict()
        splat_state = checkpoint["splats"]
        for key, value in splat_state.items():
            self.splats[key] = torch.nn.Parameter(value.to(self.device))

        print(f"Loaded {len(self.splats['means'])} Gaussian splats")

        # Load camera pose adjustments if they exist
        self.pose_adjust = None
        if "pose_adjust" in checkpoint:
            num_cameras = len(trainset)
            self.pose_adjust = CameraOptModule(num_cameras).to(self.device)
            self.pose_adjust.load_state_dict(checkpoint["pose_adjust"])
            print("Loaded camera pose adjustments")

        # Load appearance module if it exists
        self.app_module = None
        if "app_module" in checkpoint:
            # Infer feature_dim and sh_degree from checkpoint
            if "features" not in self.splats:
                raise ValueError("Appearance module requires features in checkpoint")

            feature_dim = self.splats["features"].shape[-1]
            sh_degree = 3  # Default, could be inferred from shN shape
            app_embed_dim = 16  # Default
            num_cameras = len(trainset)

            self.app_module = AppearanceOptModule(
                num_cameras, feature_dim, app_embed_dim, sh_degree
            ).to(self.device)
            self.app_module.load_state_dict(checkpoint["app_module"])
            print("Loaded appearance module")

        # Determine sh_degree from loaded splats
        if "shN" in self.splats:
            # shN has shape [N, K-1, 3] where K = (sh_degree + 1)^2
            # So K-1 = shN.shape[1]
            K_minus_1 = self.splats["shN"].shape[1]
            K = K_minus_1 + 1
            # Solve (sh_degree + 1)^2 = K
            import math

            self.sh_degree = int(math.sqrt(K)) - 1
        else:
            self.sh_degree = 3  # Default

        print(f"SH degree: {self.sh_degree}")

        # Start viewer server
        print(f"Starting viewer on port {port}...")
        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = GsplatViewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            output_dir=checkpoint_path.parent.parent,  # model directory
            mode="rendering",
        )

        print(f"\nViewer running at http://localhost:{port}")
        print("Press Ctrl+C to exit")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Rasterize the Gaussian splats."""
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.app_module is not None:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "classic"
        if camera_model is None:
            camera_model = "pinhole"

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=False,
            camera_model=camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        """Render function for the viewer."""
        assert isinstance(render_tab_state, GsplatRenderTabState)

        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        match render_tab_state.render_mode:
            case "rgb":
                render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
                renders = render_colors.cpu().numpy()
            case "depth(accumulated)" | "depth(expected)":
                depth = render_colors[0, ..., 0:1]
                if render_tab_state.normalize_nearfar:
                    near_plane = render_tab_state.near_plane
                    far_plane = render_tab_state.far_plane
                else:
                    near_plane = depth.min()
                    far_plane = depth.max()
                depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
                depth_norm = torch.clip(depth_norm, 0, 1)
                if render_tab_state.inverse:
                    depth_norm = 1 - depth_norm
                renders = (
                    apply_float_colormap(depth_norm, render_tab_state.colormap)
                    .cpu()
                    .numpy()
                )
            case "alpha":
                alpha = render_alphas[0, ..., 0:1]
                if render_tab_state.inverse:
                    alpha = 1 - alpha
                renders = (
                    apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
                )

        return renders

    def run(self):
        """Run the viewer (blocking)."""
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down viewer...")


def entrypoint():
    """CLI entrypoint for the model viewer."""
    parser = argparse.ArgumentParser(
        description="View a trained Gaussian Splatting model"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        required=True,
        help="Path to the checkpoint file (.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the processed dataset directory (containing transforms.json)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viewer server (default: 8080)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Create and run viewer
    try:
        viewer = ModelViewer(
            checkpoint_path=args.checkpoint_path,
            data_dir=args.data_dir,
            port=args.port,
        )
        viewer.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    entrypoint()
