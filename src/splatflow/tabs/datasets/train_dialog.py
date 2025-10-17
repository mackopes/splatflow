import pathlib
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

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
        with Container(id="dialog"):
            yield Label(
                f"Traint dataset: {self.root_dataset.name}/{self.processed_dataset.name}"
            )
            yield Label("Enter name for the model:")
            yield Input(
                placeholder="e.g., hloc_superpoint",
                value="model_name",
                id="model-name",
            )
            with Vertical(id="buttons"):
                yield Button("Process", variant="primary", id="train")
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
        return GsplatCommandSettings(
            dataset_dir=self.root_dataset.dataset_dir / self.processed_dataset.name,
            output_dir=model_output_dir,
        )

    # def on_input_submitted(self, event: Input.Submitted) -> None:
    #     """Allow Enter key to submit."""
    #     self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)
