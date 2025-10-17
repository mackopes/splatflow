from typing import TYPE_CHECKING, List

from textual.app import ComposeResult
from textual.events import Resize
from textual.widgets import Tree
from rich.text import Text

from splatflow.tabs.flow_tab import FlowTab
from .scanner import DatasetModels, scan_models
from .model import ProcessedModelState

if TYPE_CHECKING:
    from splatflow.main import SplatflowApp


class ModelsPane(FlowTab):
    models: List[DatasetModels] = []

    def compose(self) -> ComposeResult:
        tree: Tree[DatasetModels] = Tree("Models")
        tree.show_root = False  # Hide the root, show dataset models directly
        yield tree

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

    def populate_tree(self) -> None:
        """Populate the tree with models after layout."""
        tree = self.query_one(Tree)

        # Clear existing tree nodes
        tree.root.remove_children()

        # Get the available width for formatting
        # Use pane width, subtract space for tree guides/icons
        available_width = self.size.width - 2

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
