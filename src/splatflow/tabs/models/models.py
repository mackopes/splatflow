import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List

from rich.json import JSON
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.reactive import var
from textual.widgets import Static, Tree

from splatflow.components.delete_dialog import DeleteDialog
from splatflow.components.viewer_dialog import ViewerDialog
from splatflow.tabs.flow_tab import FlowTab

from .model import ProcessedModel, ProcessedModelState, SplatflowModelData
from .scanner import DatasetModels, scan_models

if TYPE_CHECKING:
    from splatflow.main import SplatflowApp


class ModelsPane(FlowTab):
    models: List[DatasetModels] = []
    selected_node_data: var[DatasetModels | object | None] = var(None)

    BINDINGS = [
        ("d", "delete_model", "Delete model"),
        ("v", "view_model", "View model"),
    ]

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="models-tree-container") as v:
                v.border_title = "Models"
                tree: Tree[DatasetModels] = Tree("Models", id="models-tree")
                tree.show_root = False  # Hide the root, show dataset models directly
                yield tree
            with Vertical(id="models-settings-container") as v:
                v.border_title = "Settings"
                yield Static("", id="models-settings")

    def focus_content(self) -> None:
        """Focus the models tree."""
        self.query_one(Tree).focus()

    def on_mount(self) -> None:
        """Load and display models when the pane is mounted."""
        app: SplatflowApp = self.app  # type: ignore
        config = app.config
        self.models = scan_models(config.splatflow_data_root)

        # Defer populating the tree until after layout
        self.call_after_refresh(self.populate_tree)

    def on_resize(self, event: Resize) -> None:
        """Handle resize events to refresh tree formatting."""
        if self.models:  # Only refresh if we have models
            self.populate_tree()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Update selected data when tree cursor moves."""
        self.refresh_bindings()
        if event.node and event.node.data:
            self.selected_node_data = event.node.data
        else:
            self.selected_node_data = None

    def watch_selected_node_data(
        self, selected_node_data: DatasetModels | object | None
    ) -> None:
        """Update settings display when selection changes."""
        settings_widget = self.query_one("#models-settings", Static)

        if selected_node_data is None:
            settings_widget.update("")
            return

        # Check if it's a ProcessedModel (child node)
        if isinstance(selected_node_data, ProcessedModel):
            # Display the ProcessedModel settings as formatted JSON
            settings_widget.update(
                JSON(json.dumps(selected_node_data.settings, indent=2))
            )
        else:
            # It's a parent DatasetModels node - show placeholder
            settings_widget.update("[dim]Select a model to view settings[/dim]")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check if an action should be enabled based on current state."""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return False  # Disabled when no node selected

        node_data = tree.cursor_node.data

        if action == "delete_model":
            # Only enable delete for ProcessedModel (child) nodes
            if not node_data or not isinstance(node_data, ProcessedModel):
                return False
            return True

        if action == "view_model":
            # Only enable view for READY ProcessedModel nodes with checkpoints
            if not node_data or not isinstance(node_data, ProcessedModel):
                return False
            if node_data.state != ProcessedModelState.READY:
                return False

            # Check if at least one checkpoint exists
            app: SplatflowApp = self.app  # type: ignore
            models_dir = Path(app.config.splatflow_data_root) / "models"
            model_dir = models_dir / node_data.dataset_name / node_data.name
            ckpt_dir = model_dir / "ckpts"

            if not ckpt_dir.exists():
                return False

            # Check if any .pt files exist
            ckpt_files = list(ckpt_dir.glob("*.pt"))
            return len(ckpt_files) > 0

        return True  # Enable all other actions by default

    def action_delete_model(self):
        """Delete the currently selected model."""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return

        # Get the ProcessedModel from the cursor node's data
        model = tree.cursor_node.data
        if not isinstance(model, ProcessedModel):
            return

        # Run the dialog in a worker (required for push_screen_wait)
        self.run_worker(self._delete_model_worker(model))

    def action_view_model(self):
        """View the currently selected model."""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return

        # Get the ProcessedModel from the cursor node's data
        model = tree.cursor_node.data
        if not isinstance(model, ProcessedModel):
            return

        # Run the viewer in a worker
        self.run_worker(self._view_model_worker(model))

    async def _delete_model_worker(self, model: ProcessedModel):
        """Worker method to handle async dialog interaction for deletion."""
        # Show confirmation dialog
        confirmed = await self.app.push_screen_wait(
            DeleteDialog(f"Delete model: {model.dataset_name}/{model.name}?")
        )

        if not confirmed:  # User cancelled
            return

        app: SplatflowApp = self.app  # type: ignore

        # Get the models directory for this dataset
        models_dataset_dir = (
            Path(app.config.splatflow_data_root) / "models" / model.dataset_name
        )
        metadata_path = models_dataset_dir / "splatflow_data.json"
        model_dir = models_dataset_dir / model.name

        try:
            # 1. Remove from JSON metadata
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    data = json.load(f)

                model_data = SplatflowModelData.from_dict(data)

                # Remove the model from the list
                model_data.models = [
                    m
                    for m in model_data.models
                    if not (
                        m.name == model.name and m.dataset_name == model.dataset_name
                    )
                ]

                # Save or delete the metadata file
                if model_data.models:
                    # Still have models, save updated metadata
                    model_data.save(metadata_path)
                else:
                    # No models left, delete the metadata file
                    metadata_path.unlink()

            # 2. Delete the model's directory
            if model_dir.exists():
                shutil.rmtree(model_dir)

            # 3. If no models left, delete the parent dataset directory
            if models_dataset_dir.exists() and not any(models_dataset_dir.iterdir()):
                shutil.rmtree(models_dataset_dir)

            # 4. Refresh the models tab
            self.rescan_and_repopulate()

            app.notify(
                f"Deleted model: {model.dataset_name}/{model.name}",
                title="Model Deleted",
            )

        except Exception as e:
            app.notify(
                f"Error deleting model: {str(e)}", title="Error", severity="error"
            )

    def populate_tree(self) -> None:
        """Populate the tree with models after layout."""
        tree = self.query_one(Tree)

        # Clear existing tree nodes
        tree.root.remove_children()

        for dataset_models in self.models:
            # Create dataset name label (top level)
            dataset_label = Text(dataset_models.dataset_name, style="bold")

            if not dataset_models.models:
                # No models - add as leaf
                tree.root.add_leaf(dataset_label, data=dataset_models)
            else:
                # Has models - add as node with children
                dataset_node = tree.root.add(dataset_label, data=dataset_models)
                for model in dataset_models.models:
                    # Append state if not READY
                    display_name = model.name
                    if model.state != ProcessedModelState.READY:
                        display_name = f"{model.name} ({model.state.value})"
                    dataset_node.add_leaf(display_name, data=model)

    def rescan_and_repopulate(self) -> None:
        """Rescan models directory and repopulate the tree."""
        app: SplatflowApp = self.app  # type: ignore
        config = app.config
        self.models = scan_models(config.splatflow_data_root)
        self.populate_tree()

    async def _view_model_worker(self, model: ProcessedModel):
        """Worker method to launch the viewer for a model."""
        app: SplatflowApp = self.app  # type: ignore

        # Get model directory
        models_dir = Path(app.config.splatflow_data_root) / "models"
        model_dir = models_dir / model.dataset_name / model.name
        ckpt_dir = model_dir / "ckpts"

        # Find the latest checkpoint
        ckpt_files = list(ckpt_dir.glob("ckpt_*_rank*.pt"))
        if not ckpt_files:
            app.notify(
                "No checkpoints found for this model",
                title="Error",
                severity="error",
            )
            return

        # Extract step numbers and find the latest
        def get_step(path: Path) -> int:
            match = re.search(r"ckpt_(\d+)_rank", path.name)
            return int(match.group(1)) if match else 0

        latest_checkpoint = max(ckpt_files, key=get_step)

        # Get data_dir from model settings
        data_dir = Path(model.settings.get("dataset_dir", ""))
        if not data_dir.exists():
            app.notify(
                f"Dataset directory not found: {data_dir}",
                title="Error",
                severity="error",
            )
            return

        # Create and show the viewer dialog
        port = 8080  # Default port
        dialog = ViewerDialog(
            model_name=f"{model.dataset_name}/{model.name}",
            port=port,
        )

        # Start the viewer subprocess
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                "poetry",
                "run",
                "view_model",
                "--checkpoint-path",
                str(latest_checkpoint),
                "--data-dir",
                str(data_dir),
                "--port",
                str(port),
                # No stdout/stderr redirection - inherit from parent to avoid blocking
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "2",
                    "TQDM_MININTERVAL": "1",
                    "TQDM_ASCII": "True",
                    "CC": "gcc-13",
                    "CXX": "g++-13",
                },
            )

            # Store the process in the dialog so it can be killed
            dialog.set_process(process)

            # Show the dialog (blocks until user clicks Stop)
            await self.app.push_screen_wait(dialog)

            # Wait for process to exit (should be quick after kill)
            await asyncio.wait_for(process.wait(), timeout=5.0)

        except asyncio.TimeoutError:
            # Process didn't exit gracefully, force kill
            if process:
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass

        except Exception as e:
            app.notify(
                f"Error launching viewer: {str(e)}",
                title="Error",
                severity="error",
            )
