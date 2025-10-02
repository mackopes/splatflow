from textual.app import ComposeResult
from textual.widgets import Label

from .flow_tab import FlowTab


class ModelsPane(FlowTab):
    def compose(self) -> ComposeResult:
        yield Label("Models content goes here")

    def focus_content(self) -> None:
        """No-op: Models pane has no focusable content yet."""
        pass
