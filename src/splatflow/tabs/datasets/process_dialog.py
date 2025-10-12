from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Container, Vertical
from textual.widgets import Input, Button, Label
from textual.binding import Binding


class ProcessDialog(ModalScreen[str | None]):
    """Modal dialog to get processed dataset name."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, dataset_name: str):
        super().__init__()
        self.dataset_name = dataset_name

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(f"Process dataset: {self.dataset_name}")
            yield Label("Enter name for processed dataset:")
            yield Input(
                placeholder="e.g., hloc_superpoint",
                value="processed",
                id="name-input",
            )
            with Vertical(id="buttons"):
                yield Button("Process", variant="primary", id="process")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        """Focus the input when dialog opens."""
        self.query_one("#name-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "process":
            input_widget = self.query_one("#name-input", Input)
            self.dismiss(input_widget.value)
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Allow Enter key to submit."""
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)
