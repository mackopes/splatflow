"""Modal dialog for viewing a trained Gaussian Splatting model."""

import asyncio
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static
from textual.reactive import var


class ViewerDialog(ModalScreen[bool]):
    """Modal dialog that shows viewer status and allows stopping it."""

    CSS_PATH = "viewer_dialog.tcss"

    BINDINGS = [
        Binding("escape", "stop", "Stop"),
    ]

    # Reactive state
    status: var[str] = var("Starting...")

    def __init__(self, model_name: str, port: int = 8080):
        super().__init__()
        self.model_name = model_name
        self.port = port
        self.process: asyncio.subprocess.Process | None = None

    def compose(self) -> ComposeResult:
        with Container(id="viewer-dialog"):
            yield Label(f"[bold]Viewing:[/bold] {self.model_name}", id="viewer-title")

            with Vertical(id="viewer-status-container"):
                yield Static("", id="viewer-status")
                yield Static("", id="viewer-url")

            with Horizontal(id="viewer-buttons"):
                yield Button("Stop", variant="error", id="stop")

    def on_mount(self) -> None:
        """Focus the stop button and update initial status."""
        self.query_one("#stop", Button).focus()
        self.update_status_display()

    def watch_status(self, status: str) -> None:
        """Update the display when status changes."""
        self.update_status_display()

    def update_status_display(self) -> None:
        """Update the status and URL labels."""
        # Only update if dialog is mounted (widgets exist)
        if not self.is_mounted:
            return

        status_widget = self.query_one("#viewer-status", Static)
        url_widget = self.query_one("#viewer-url", Static)

        # Update status with color coding
        if self.status == "Starting...":
            status_widget.update(f"[yellow]● {self.status}[/yellow]")
            url_widget.update("")
        elif self.status == "Running":
            status_widget.update(f"[green]● {self.status}[/green]")
            url_widget.update(f"[cyan]http://localhost:{self.port}[/cyan]")
        elif self.status == "Stopped":
            status_widget.update(f"[red]● {self.status}[/red]")
            url_widget.update("")
        else:
            status_widget.update(f"[dim]● {self.status}[/dim]")
            url_widget.update("")

    def set_process(self, process: asyncio.subprocess.Process) -> None:
        """Store the subprocess handle."""
        self.process = process
        self.status = "Running"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop":
            self.action_stop()

    def action_stop(self) -> None:
        """Stop the viewer and dismiss the dialog."""
        self.status = "Stopped"
        if self.process:
            try:
                self.process.kill()
            except ProcessLookupError:
                # Process already terminated
                pass
        self.dismiss(True)
