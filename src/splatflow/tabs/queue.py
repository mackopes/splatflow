from textual.app import ComposeResult
from textual.widgets import Label

from .flow_tab import FlowTab


class QueuePane(FlowTab):
    def compose(self) -> ComposeResult:
        yield Label("Queue content goes here")

    def focus_content(self) -> None:
        """No-op: Queue pane has no focusable content yet."""
        pass
