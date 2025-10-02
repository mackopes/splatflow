from abc import abstractmethod

from textual.widgets import Static


class FlowTab(Static):
    """Abstract base class for tab content panes."""

    @abstractmethod
    def focus_content(self) -> None:
        """Focus the main content of this tab."""
        ...
