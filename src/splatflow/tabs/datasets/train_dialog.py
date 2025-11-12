import pathlib
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Checkbox

from splatflow.scripts.train import GsplatCommandSettings
from splatflow.tabs.datasets.dataset import Dataset, ProcessedDataset

if TYPE_CHECKING:
    from splatflow.main import SplatflowApp


class TrainDialog(ModalScreen[GsplatCommandSettings | None]):
    """Modal dialog to get processed dataset name."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, root_dataset: Dataset, processed_dataset: ProcessedDataset):
        super().__init__()
        self.root_dataset = root_dataset
        self.processed_dataset = processed_dataset

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Label(
                f"Train dataset: {self.root_dataset.name}/{self.processed_dataset.name}"
            )
            yield Label("Model name:", classes="mt-1")
            yield Input(
                placeholder="e.g., model_name",
                value="default",
                classes="text-input",
                id="model-name",
            )

            with VerticalScroll(id="settings-scroll", classes="mt-1") as v:
                v.border_title = "Settings"
                # Strategy
                with Horizontal(classes="setting-row"):
                    yield Label("Strategy:", classes="setting-label")
                    yield Select(
                        [("default", "default"), ("mcmc", "mcmc")],
                        value="default",
                        allow_blank=False,
                        id="strategy",
                        classes="setting-input",
                        compact=True,
                    )

                # Data factor
                with Horizontal(classes="setting-row"):
                    yield Label("Data factor:", classes="setting-label")
                    yield Select(
                        [("1", 1), ("2", 2), ("4", 4), ("8", 8), ("16", 16)],
                        value=1,
                        allow_blank=False,
                        id="data-factor",
                        classes="setting-input",
                        compact=True,
                    )

                # Steps scaler
                with Horizontal(classes="setting-row"):
                    yield Label("Steps scaler:", classes="setting-label")
                    yield Input(
                        type="number",
                        value="4.0",
                        id="steps-scaler",
                        classes="setting-input",
                        compact=True,
                    )

                # Patch size (optional)
                with Horizontal(classes="setting-row"):
                    yield Label("Patch size (optional):", classes="setting-label")
                    yield Input(
                        type="integer",
                        placeholder="None",
                        id="patch-size",
                        classes="setting-input",
                        compact=True,
                    )

                # Global scale
                with Horizontal(classes="setting-row"):
                    yield Label("Global scale:", classes="setting-label")
                    yield Input(
                        value="1.0",
                        id="global-scale",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # Normalize world space
                with Horizontal(classes="setting-row"):
                    yield Checkbox(
                        "Normalize world space",
                        value=True,
                        id="normalize-world-space",
                        compact=True,
                    )

                # Camera model
                with Horizontal(classes="setting-row"):
                    yield Label("Camera model:", classes="setting-label")
                    yield Select(
                        [
                            ("pinhole", "pinhole"),
                            ("ortho", "ortho"),
                            ("fisheye", "fisheye"),
                        ],
                        value="pinhole",
                        id="camera-model",
                        classes="setting-input",
                        compact=True,
                    )

                # Batch size
                with Horizontal(classes="setting-row"):
                    yield Label("Batch size:", classes="setting-label")
                    yield Input(
                        value="1",
                        id="batch-size",
                        classes="setting-input",
                        compact=True,
                        type="integer",
                    )

                # Init type
                with Horizontal(classes="setting-row"):
                    yield Label("Init type:", classes="setting-label")
                    yield Select(
                        [("sfm", "sfm"), ("random", "random")],
                        value="sfm",
                        allow_blank=False,
                        id="init-type",
                        classes="setting-input",
                        compact=True,
                    )

                # Init num pts
                with Horizontal(classes="setting-row"):
                    yield Label("Init num GSs:", classes="setting-label")
                    yield Input(
                        value="100000",
                        id="init-num-pts",
                        classes="setting-input",
                        type="integer",
                        compact=True,
                    )

                # Init extent
                with Horizontal(classes="setting-row"):
                    yield Label("Init extent:", classes="setting-label")
                    yield Input(
                        value="3.0",
                        id="init-extent",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # SH degree
                with Horizontal(classes="setting-row"):
                    yield Label("SH degree:", classes="setting-label")
                    yield Input(
                        value="3",
                        id="sh-degree",
                        classes="setting-input",
                        type="integer",
                        compact=True,
                    )

                # SH degree interval
                with Horizontal(classes="setting-row"):
                    yield Label("SH degree interval:", classes="setting-label")
                    yield Input(
                        value="1000",
                        id="sh-degree-interval",
                        classes="setting-input",
                        type="integer",
                        compact=True,
                    )

                # Init opacity
                with Horizontal(classes="setting-row"):
                    yield Label("Init opacity:", classes="setting-label")
                    yield Input(
                        value="0.1",
                        id="init-opa",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # Init scale
                with Horizontal(classes="setting-row"):
                    yield Label("Init scale:", classes="setting-label")
                    yield Input(
                        value="1.0",
                        id="init-scale",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # SSIM lambda
                with Horizontal(classes="setting-row"):
                    yield Label("SSIM lambda:", classes="setting-label")
                    yield Input(
                        value="0.2",
                        id="ssim-lambda",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # Near plane
                with Horizontal(classes="setting-row"):
                    yield Label("Near plane:", classes="setting-label")
                    yield Input(
                        value="0.01",
                        id="near-plane",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # Far plane
                with Horizontal(classes="setting-row"):
                    yield Label("Far plane:", classes="setting-label")
                    yield Input(
                        value="1e10",
                        id="far-plane",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # Antialiased
                with Horizontal(classes="setting-row"):
                    yield Checkbox(
                        "Antialiased",
                        value=False,
                        id="antialiased",
                        compact=True,
                    )

                # Random background
                with Horizontal(classes="setting-row"):
                    yield Checkbox(
                        "Random background",
                        value=True,
                        id="random-bkgd",
                        compact=True,
                    )

                # Learning rates
                with Horizontal(classes="setting-row"):
                    yield Label("Means LR:", classes="setting-label")
                    yield Input(
                        value="0.00016",
                        id="means-lr",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("Scales LR:", classes="setting-label")
                    yield Input(
                        value="0.005",
                        id="scales-lr",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("Opacities LR:", classes="setting-label")
                    yield Input(
                        value="0.05",
                        id="opacities-lr",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("Quats LR:", classes="setting-label")
                    yield Input(
                        value="0.001",
                        id="quats-lr",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("SH0 LR:", classes="setting-label")
                    yield Input(
                        value="0.0025",
                        id="sh0-lr",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("SHN LR:", classes="setting-label")
                    yield Input(
                        value="0.000125",
                        id="shN-lr",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # Regularization
                with Horizontal(classes="setting-row"):
                    yield Label("Opacity reg:", classes="setting-label")
                    yield Input(
                        value="0.0",
                        id="opacity-reg",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("Scale reg:", classes="setting-label")
                    yield Input(
                        value="0.0",
                        id="scale-reg",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # Pose optimization
                with Horizontal(classes="setting-row"):
                    yield Checkbox(
                        "Pose optimization",
                        value=True,
                        id="pose-opt",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("Pose opt LR:", classes="setting-label")
                    yield Input(
                        value="1e-05",
                        id="pose-opt-lr",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("Pose opt reg:", classes="setting-label")
                    yield Input(
                        value="1e-06",
                        id="pose-opt-reg",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # Appearance optimization
                with Horizontal(classes="setting-row"):
                    yield Checkbox(
                        "Appearance optimization",
                        value=False,
                        id="app-opt",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("App embed dim:", classes="setting-label")
                    yield Input(
                        value="16",
                        id="app-embed-dim",
                        classes="setting-input",
                        type="integer",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("App opt LR:", classes="setting-label")
                    yield Input(
                        value="0.001",
                        id="app-opt-lr",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                with Horizontal(classes="setting-row"):
                    yield Label("App opt reg:", classes="setting-label")
                    yield Input(
                        value="1e-06",
                        id="app-opt-reg",
                        classes="setting-input",
                        type="number",
                        compact=True,
                    )

                # Fused bilagrid
                with Horizontal(classes="setting-row"):
                    yield Checkbox(
                        "Use fused bilagrid",
                        value=False,
                        id="use-fused-bilagrid",
                        compact=True,
                    )

            with Horizontal(id="buttons"):
                yield Button("Train", variant="primary", id="train")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        """Focus the input when dialog opens."""
        self.query_one("#model-name", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train":
            # Return the command settings (not the built command)
            command_settings = self._construct_command()
            self.dismiss(command_settings)
        else:
            self.dismiss(None)

    def _construct_command(self) -> GsplatCommandSettings:
        model_name_input = self.query_one("#model-name", Input)
        app: SplatflowApp = self.app  # type: ignore
        model_output_dir = (
            pathlib.Path(app.config.splatflow_data_root)
            / "models"
            / self.root_dataset.name
            / model_name_input.value
        )

        # Read all settings from inputs
        strategy = self.query_one("#strategy", Select).value
        data_factor_value = self.query_one("#data-factor", Select).value
        assert data_factor_value is not Select.BLANK
        data_factor = int(data_factor_value)  # type: ignore we asserted it's not blank, but typechecker doesn't know
        steps_scaler = float(self.query_one("#steps-scaler", Input).value)

        # Patch size is optional
        patch_size_str = self.query_one("#patch-size", Input).value.strip()
        patch_size = int(patch_size_str) if patch_size_str else None

        global_scale = float(self.query_one("#global-scale", Input).value)
        normalize_world_space = self.query_one("#normalize-world-space", Checkbox).value
        camera_model = self.query_one("#camera-model", Select).value
        batch_size = int(self.query_one("#batch-size", Input).value)
        init_type = self.query_one("#init-type", Select).value
        init_num_pts = int(self.query_one("#init-num-pts", Input).value)
        init_extent = float(self.query_one("#init-extent", Input).value)
        sh_degree = int(self.query_one("#sh-degree", Input).value)
        sh_degree_interval = int(self.query_one("#sh-degree-interval", Input).value)
        init_opa = float(self.query_one("#init-opa", Input).value)
        init_scale = float(self.query_one("#init-scale", Input).value)
        ssim_lambda = float(self.query_one("#ssim-lambda", Input).value)
        near_plane = float(self.query_one("#near-plane", Input).value)
        far_plane = float(self.query_one("#far-plane", Input).value)
        antialiased = self.query_one("#antialiased", Checkbox).value
        random_bkgd = self.query_one("#random-bkgd", Checkbox).value
        means_lr = float(self.query_one("#means-lr", Input).value)
        scales_lr = float(self.query_one("#scales-lr", Input).value)
        opacities_lr = float(self.query_one("#opacities-lr", Input).value)
        quats_lr = float(self.query_one("#quats-lr", Input).value)
        sh0_lr = float(self.query_one("#sh0-lr", Input).value)
        shN_lr = float(self.query_one("#shN-lr", Input).value)
        opacity_reg = float(self.query_one("#opacity-reg", Input).value)
        scale_reg = float(self.query_one("#scale-reg", Input).value)
        pose_opt = self.query_one("#pose-opt", Checkbox).value
        pose_opt_lr = float(self.query_one("#pose-opt-lr", Input).value)
        pose_opt_reg = float(self.query_one("#pose-opt-reg", Input).value)
        app_opt = self.query_one("#app-opt", Checkbox).value
        app_embed_dim = int(self.query_one("#app-embed-dim", Input).value)
        app_opt_lr = float(self.query_one("#app-opt-lr", Input).value)
        app_opt_reg = float(self.query_one("#app-opt-reg", Input).value)
        use_fused_bilagrid = self.query_one("#use-fused-bilagrid", Checkbox).value

        return GsplatCommandSettings(
            dataset_dir=self.root_dataset.dataset_dir / self.processed_dataset.name,
            output_dir=model_output_dir,
            strategy=strategy,  # type: ignore
            data_factor=data_factor,
            steps_scaler=steps_scaler,
            patch_size=patch_size,
            global_scale=global_scale,
            normalize_world_space=normalize_world_space,
            camera_model=camera_model,  # type: ignore
            batch_size=batch_size,
            init_type=init_type,  # type: ignore
            init_num_pts=init_num_pts,
            init_extent=init_extent,
            sh_degree=sh_degree,
            sh_degree_interval=sh_degree_interval,
            init_opa=init_opa,
            init_scale=init_scale,
            ssim_lambda=ssim_lambda,
            near_plane=near_plane,
            far_plane=far_plane,
            antialiased=antialiased,
            random_bkgd=random_bkgd,
            means_lr=means_lr,
            scales_lr=scales_lr,
            opacities_lr=opacities_lr,
            quats_lr=quats_lr,
            sh0_lr=sh0_lr,
            shN_lr=shN_lr,
            opacity_reg=opacity_reg,
            scale_reg=scale_reg,
            pose_opt=pose_opt,
            pose_opt_lr=pose_opt_lr,
            pose_opt_reg=pose_opt_reg,
            app_opt=app_opt,
            app_embed_dim=app_embed_dim,
            app_opt_lr=app_opt_lr,
            app_opt_reg=app_opt_reg,
            use_fused_bilagrid=use_fused_bilagrid,
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Start training when Enter is pressed on any input."""
        command_settings = self._construct_command()
        self.dismiss(command_settings)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_train(self) -> None:
        """Start training when Enter is pressed."""
        command_settings = self._construct_command()
        self.dismiss(command_settings)
