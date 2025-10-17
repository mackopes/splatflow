from typing import TYPE_CHECKING, List
import json

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.reactive import var
from textual.widgets import Tree, Static
from rich.text import Text
from rich.json import JSON

from splatflow.tabs.flow_tab import FlowTab
from .scanner import DatasetModels, scan_models
from .model import ProcessedModel, ProcessedModelState

if TYPE_CHECKING:
    from splatflow.main import SplatflowApp


class ModelsPane(FlowTab):
    models: List[DatasetModels] = []
    selected_node_data: var[DatasetModels | object | None] = var(None)

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
        if event.node and event.node.data:
            self.selected_node_data = event.node.data
        else:
            self.selected_node_data = None

    def watch_selected_node_data(self, selected_node_data: DatasetModels | object | None) -> None:
        """Update settings display when selection changes."""
        settings_widget = self.query_one("#models-settings", Static)

        if selected_node_data is None:
            settings_widget.update("")
            return

        # Check if it's a ProcessedModel (child node)
        if isinstance(selected_node_data, ProcessedModel):
            # Display the ProcessedModel settings as formatted JSON
            settings_widget.update(JSON(json.dumps(selected_node_data.settings, indent=2)))
        else:
            # It's a parent DatasetModels node - show placeholder
            settings_widget.update("[dim]Select a model to view settings[/dim]")

    def populate_tree(self) -> None:
        """Populate the tree with models after layout."""
        tree = self.query_one(Tree)

        # Clear existing tree nodes
        tree.root.remove_children()

        # Get the available width for formatting
        # Use pane width, subtract space for tree guides/icons
        available_width = self.size.width // 2 - 5

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
