from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class DeleteDialog(ModalScreen[bool]):
    """Generic modal dialog to confirm deletion."""

    CSS_PATH = "delete_dialog.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("left", "focus_delete", "Focus Delete", show=False),
        Binding("right", "focus_cancel", "Focus Cancel", show=False),
    ]

    def __init__(self, title: str):
        super().__init__()
        self.title = title

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(self.title)
            yield Label("[bold red]This action cannot be undone![/bold red]")
            with Horizontal(id="buttons"):
                yield Button("Delete", variant="error", id="delete")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        """Focus the delete button by default."""
        self.query_one("#delete", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "delete":
            self.dismiss(True)  # Confirmed
        else:
            self.dismiss(False)  # Cancelled

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_focus_delete(self) -> None:
        """Focus the delete button."""
        self.query_one("#delete", Button).focus()

    def action_focus_cancel(self) -> None:
        """Focus the cancel button."""
        self.query_one("#cancel", Button).focus()
