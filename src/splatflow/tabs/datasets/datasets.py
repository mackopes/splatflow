from typing import TYPE_CHECKING, List
from pathlib import Path
import json

from splatflow.tabs.datasets.train_dialog import TrainDialog
from textual.app import ComposeResult
from textual.events import Resize
from textual.widgets import Tree
from rich.text import Text

from .dataset import Dataset, ProcessedDataset, ProcessedDatasetState
from .scanner import scan_datasets
from splatflow.tabs.flow_tab import FlowTab

from splatflow.scripts.process import HlocCommandSettings
from splatflow.scripts.train import GsplatCommandSettings
from .process_dialog import ProcessDialog
from splatflow.tabs.models.model import ProcessedModel, ProcessedModelState, SplatflowModelData

if TYPE_CHECKING:
    from splatflow.main import SplatflowApp


class DatasetsPane(FlowTab):
    datasets: List[Dataset] = []
    BINDINGS = [
        ("i", "import_dataset", "Import Dataset"),
        ("p", "process_dataset", "Process dataset"),
        ("m", "train_dataset", "Train dataset"),
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
        app: SplatflowApp = self.app  # type: ignore
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
                    # Append state if not READY
                    display_name = processed.name
                    if processed.state != ProcessedDatasetState.READY:
                        display_name = f"{processed.name} ({processed.state.value})"
                    # Store tuple of (parent_dataset, processed_dataset) for child nodes
                    dataset_node.add_leaf(display_name, data=(dataset, processed))

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Refresh bindings when tree cursor moves."""
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check if an action should be enabled based on current state."""
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            return False  # Disabled when no node selected

        if action == "process_dataset":
            # Check if the cursor is on a top-level Dataset node
            # Top-level nodes have the tree root as their parent
            if tree.cursor_node.parent != tree.root:
                return False  # Disabled for child nodes (ProcessedDatasets)

            # Verify we have valid Dataset data
            dataset = tree.cursor_node.data
            if not dataset or not isinstance(dataset, Dataset):
                return False

            return True  # Enabled for top-level Dataset nodes

        if action == "train_dataset":
            # Check if the cursor is on a child node (ProcessedDataset)
            # Child nodes have tuple data: (dataset, processed_dataset)
            if tree.cursor_node.parent == tree.root:
                return False  # Disabled for top-level Dataset nodes

            # Verify we have valid tuple data
            node_data = tree.cursor_node.data
            if not node_data or not isinstance(node_data, tuple) or len(node_data) != 2:
                return False

            return True  # Enabled for child ProcessedDataset nodes

        return True  # Enable all other actions by default

    def action_import_dataset(self):
        pass

    def action_train_dataset(self):
        """Train the currently selected processed dataset."""
        self.log("=== ACTION TRAIN DATASET CALLED ===")
        tree = self.query_one(Tree)
        if not tree.cursor_node:
            print("not cursor node")
            return

        # Get both dataset and processed_dataset from the cursor node's data
        node_data = tree.cursor_node.data
        if not node_data or not isinstance(node_data, tuple) or len(node_data) != 2:
            return

        dataset, processed_dataset = node_data
        if not isinstance(dataset, Dataset) or not isinstance(
            processed_dataset, ProcessedDataset
        ):
            return

        # Run the dialog in a worker (required for push_screen_wait)
        self.run_worker(self._train_dataset_worker(dataset, processed_dataset))

    async def _train_dataset_worker(
        self, dataset: Dataset, processed_dataset: ProcessedDataset
    ):
        """Worker method to handle async dialog interaction for training."""
        # Show dialog and get command settings
        command_settings = await self.app.push_screen_wait(
            TrainDialog(dataset, processed_dataset)
        )

        if command_settings is None:  # User cancelled
            return

        app: SplatflowApp = self.app  # type: ignore

        # Extract model name from output_dir path
        model_name = command_settings.output_dir.name

        # Get the models directory for this dataset
        models_dataset_dir = Path(app.config.splatflow_data_root) / "models" / dataset.name
        models_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Path to the metadata JSON file
        metadata_path = models_dataset_dir / "splatflow_data.json"

        # Load or create metadata
        def load_model_metadata() -> SplatflowModelData:
            if not metadata_path.exists():
                return SplatflowModelData([])
            try:
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                return SplatflowModelData.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.log(f"Warning: Could not parse {metadata_path}: {e}")
                return SplatflowModelData([])

        # Create ProcessedModel with PENDING state
        processed_model = ProcessedModel(
            name=model_name,
            dataset_name=dataset.name,
            state=ProcessedModelState.PENDING,
            settings=command_settings.to_dict(),
        )

        # Add to metadata and save
        model_metadata = load_model_metadata()
        model_metadata.models.append(processed_model)
        model_metadata.save(metadata_path)

        # Define callbacks to update the state
        def before_exec_callback():
            """Update state to PROCESSING before execution."""
            metadata = load_model_metadata()
            for m in metadata.models:
                if m.name == model_name and m.dataset_name == dataset.name:
                    m.state = ProcessedModelState.PROCESSING
                    break
            metadata.save(metadata_path)

        def on_success_callback():
            """Update state to READY on success."""
            metadata = load_model_metadata()
            for m in metadata.models:
                if m.name == model_name and m.dataset_name == dataset.name:
                    m.state = ProcessedModelState.READY
                    break
            metadata.save(metadata_path)

        def on_error_callback():
            """Update state to FAILED on error."""
            metadata = load_model_metadata()
            for m in metadata.models:
                if m.name == model_name and m.dataset_name == dataset.name:
                    m.state = ProcessedModelState.FAILED
                    break
            metadata.save(metadata_path)

        # Build the command
        command = command_settings.build()

        # Display the command in a toast notification
        command_str = " ".join(str(c) for c in command)
        app.notify(f"Training command: {command_str}", title="Training Command")

        # Add to queue with callbacks
        app.add_to_queue(
            name=f"Train: {dataset.name}/{processed_dataset.name} → {model_name}",
            command=command,
            before_exec_callback=before_exec_callback,
            on_success_callback=on_success_callback,
            on_error_callback=on_error_callback,
        )

        # Switch to queue tab
        app.switch_to_queue_tab()

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
        app: SplatflowApp = self.app  # type: ignore
        dataset_path = Path(app.config.splatflow_data_root) / "datasets" / dataset.name

        builder = HlocCommandSettings(
            dataset_dir=dataset_path,
            output_dir=dataset_path / processed_name,
            # Using defaults for now, can expose these to UI later
        )

        # Create ProcessedDataset with PENDING state and save to JSON
        processed_dataset = ProcessedDataset(
            name=processed_name,
            state=ProcessedDatasetState.PENDING,
            settings=builder.to_dict(),
        )
        dataset.add_processed_dataset(processed_dataset)

        # TODO: Extract and refactor these callbacks
        # Define callbacks to update the state
        def before_exec_callback():
            """Update state to PROCESSING before execution."""
            splatflow_data = dataset._load_splatflow_data()
            if splatflow_data:
                for pd in splatflow_data.datasets:
                    if pd.name == processed_name:
                        pd.state = ProcessedDatasetState.PROCESSING
                        break
                splatflow_data.save(dataset._splatflow_data_path)

        def on_success_callback():
            """Update state to READY on success."""
            splatflow_data = dataset._load_splatflow_data()
            if splatflow_data:
                for pd in splatflow_data.datasets:
                    if pd.name == processed_name:
                        pd.state = ProcessedDatasetState.READY
                        break
                splatflow_data.save(dataset._splatflow_data_path)

        def on_error_callback():
            """Update state to FAILED on error."""
            splatflow_data = dataset._load_splatflow_data()
            if splatflow_data:
                for pd in splatflow_data.datasets:
                    if pd.name == processed_name:
                        pd.state = ProcessedDatasetState.FAILED
                        break
                splatflow_data.save(dataset._splatflow_data_path)

        # TODO: Check if the name is unique
        # Add to queue with callbacks
        app.add_to_queue(
            name=f"Process: {dataset.name} → {processed_name}",
            command=builder.build(),
            before_exec_callback=before_exec_callback,
            on_success_callback=on_success_callback,
            on_error_callback=on_error_callback,
        )

        # Switch to queue tab
        app.switch_to_queue_tab()
