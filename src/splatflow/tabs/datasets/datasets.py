from typing import TYPE_CHECKING, List

from textual.app import ComposeResult
from textual.events import Resize
from textual.widgets import Tree
from rich.text import Text

from .dataset import Dataset
from .scanner import scan_datasets
from splatflow.tabs.flow_tab import FlowTab


if TYPE_CHECKING:
    from splatflow.main import MyApp


class DatasetsPane(FlowTab):
    datasets: List[Dataset] = []
    BINDINGS = [
        ("i", "action_import_dataset", "Import Dataset"),
        ("p", "action_process_dataset", "Process dataset"),
    ]

    def compose(self) -> ComposeResult:
        tree: Tree[Dataset] = Tree("Datasets")
        tree.show_root = False  # Hide the root, show datasets directly
        yield tree

    def focus_content(self) -> None:
        """Focus the datasets tree."""
        self.query_one(Tree).focus()

    def on_mount(self) -> None:
        """Load and display datasets when the pane is mounted."""
        app: MyApp = self.app  # type: ignore
        config = app.config
        self.datasets = scan_datasets(config.splatflow_data_root)

        # Defer populating the tree until after layout
        self.call_after_refresh(self.populate_tree)

    def on_resize(self, event: Resize) -> None:
        """Handle resize events to refresh tree formatting."""
        if self.datasets:  # Only refresh if we have datasets
            self.populate_tree()

    def populate_tree(self) -> None:
        """Populate the tree with datasets after layout."""
        tree = self.query_one(Tree)

        # Clear existing tree nodes
        tree.root.remove_children()

        # Get the available width for formatting
        # Use pane width, subtract space for tree guides/icons
        available_width = self.size.width - 2

        for dataset in self.datasets:
            # Format the dataset label with styled components
            name_prefix = "* " if not dataset.processed_datasets else ""
            date_str = dataset.created_at.strftime("%Y-%m-%d")
            images_str = f"{dataset.n_images} images"

            # Create left side (name) in bold
            left_text = Text(name_prefix + dataset.name, style="bold")

            # Create right side (date and images) with spacing
            right_text = Text.assemble(
                (date_str, "dim"),
                "  ",
                (images_str, ""),
            )

            # Combine with padding to spread them apart
            # Pad the left text to push right content to the right
            padding = max(0, available_width - len(dataset.name) - 2 - len(right_text))
            left_text.pad_right(padding)

            # Combine both parts
            dataset_label = Text.assemble(left_text, right_text)

            if not dataset.processed_datasets:
                # No processed datasets - add as leaf
                tree.root.add_leaf(dataset_label, data=dataset)
            else:
                # Has processed datasets - add as node with children
                dataset_node = tree.root.add(dataset_label, data=dataset)
                for processed in dataset.processed_datasets:
                    dataset_node.add_leaf(processed.name)

    def action_import_dataset(self):
        pass

    def action_process_dataset(self):
        pass
