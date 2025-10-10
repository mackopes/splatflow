from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.widgets import Tree

from .dataset import Dataset
from .scanner import scan_datasets
from splatflow.tabs.flow_tab import FlowTab


if TYPE_CHECKING:
    from splatflow.main import MyApp


class DatasetsPane(FlowTab):
    BINDINGS = [
        ("i", "action_import_dataset", "Import Dataset"),
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
        datasets = scan_datasets(config.splatflow_data_root)

        tree = self.query_one(Tree)

        for dataset in datasets:
            dataset_label = f"{dataset.name} ({dataset.created_at.strftime('%Y-%m-%d')}, {dataset.n_images} images)"

            if not dataset.processed_datasets:
                # No processed datasets - add as leaf with * prefix
                tree.root.add_leaf(f"* {dataset_label}", data=dataset)
            else:
                # Has processed datasets - add as node with children
                dataset_node = tree.root.add(dataset_label, data=dataset)
                for processed in dataset.processed_datasets:
                    dataset_node.add_leaf(processed.name)

    def action_import_dataset(self):
        pass

    def process_dataset(self):
        pass
