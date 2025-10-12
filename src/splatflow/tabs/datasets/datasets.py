from typing import TYPE_CHECKING, List
from pathlib import Path

from textual.app import ComposeResult
from textual.events import Resize
from textual.widgets import Tree
from rich.text import Text

from .dataset import Dataset
from .scanner import scan_datasets
from splatflow.tabs.flow_tab import FlowTab

from splatflow.scripts.process import HlocCommandSettings
from .process_dialog import ProcessDialog

if TYPE_CHECKING:
    from splatflow.main import MyApp


class DatasetsPane(FlowTab):
    datasets: List[Dataset] = []
    BINDINGS = [
        ("i", "import_dataset", "Import Dataset"),
        ("p", "process_dataset", "Process dataset"),
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
        """Process the currently selected dataset."""
        self.log("=== ACTION PROCESS DATASET CALLED ===")
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            print("not cursor node")
            return

        # Get the dataset from the cursor node's data
        dataset = tree.cursor_node.data
        if not dataset or not isinstance(dataset, Dataset):
            return

        # Run the dialog in a worker (required for push_screen_wait)
        self.run_worker(self._process_dataset_worker(dataset))

    async def _process_dataset_worker(self, dataset: Dataset):
        """Worker method to handle async dialog interaction."""

        # Show dialog to get processed dataset name
        processed_name = await self.app.push_screen_wait(ProcessDialog(dataset.name))

        if processed_name is None:  # User cancelled
            return

        # Get the dataset path
        app: MyApp = self.app  # type: ignore
        dataset_path = Path(app.config.splatflow_data_root) / "datasets" / dataset.name

        builder = HlocCommandSettings(
            images_dir=dataset_path / "input",
            output_dir=dataset_path / processed_name,
            # Using defaults for now, can expose these to UI later
        )

        # Add to queue
        app.add_to_queue(
            name=f"Process: {dataset.name} → {processed_name}",
            command=builder.build(),
            # command=["echo", "processing data"],
        )

        # Switch to queue tab
        app.switch_to_queue_tab()
