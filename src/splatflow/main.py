from typing import List, Tuple, Type

from textual.app import App, ComposeResult
from textual.widgets import Footer, TabbedContent, TabPane

from splatflow.config import Config, load_config
from splatflow.initialise import initialise
from splatflow.tabs.datasets.datasets import DatasetsPane
from splatflow.tabs.models import ModelsPane
from splatflow.tabs.queue import QueuePane
from splatflow.tabs.flow_tab import FlowTab

TABS: List[Tuple[str, Type[FlowTab]]] = [
    ("Datasets", DatasetsPane),
    ("Models", ModelsPane),
    ("Queue", QueuePane),
]


class MyApp(App):
    CSS_PATH = ["app.tcss", "tabs/datasets/datasets.tcss"]

    BINDINGS = [
        ("t", "next_tab", "Next tab"),
    ]

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        with TabbedContent():
            for tab_name, tab_pane in TABS:
                with TabPane(tab_name, id=tab_name.lower()):
                    yield tab_pane()
        yield Footer()

    def action_next_tab(self) -> None:
        """Switch to the next tab and focus its content."""
        tabbed_content = self.query_one(TabbedContent)
        active_id = tabbed_content.active
        active_index = next(i for i, t in enumerate(TABS) if t[0].lower() == active_id)
        next_index = (active_index + 1) % len(TABS)
        next_id = TABS[next_index][0].lower()

        tabbed_content.active = next_id

        # Focus the active tab's content using the FlowTab interface
        active_pane = tabbed_content.get_pane(next_id)
        if active_pane:
            # The pane's first child is the FlowTab widget
            flow_tab = active_pane.query_one(FlowTab)
            flow_tab.focus_content()


def main():
    config = load_config()
    initialise(config)
    app = MyApp(config)
    app.theme = "catppuccin-mocha"
    app.run()


if __name__ == "__main__":
    main()
