from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.widgets import ListView

from .dataset_list_item import DatasetListItem
from .scanner import scan_datasets
from splatflow.tabs.flow_tab import FlowTab

if TYPE_CHECKING:
    from splatflow.main import MyApp


class DatasetsPane(FlowTab):
    BINDINGS = [
        ("i", "action_import_dataset", "Import Dataset"),
    ]

    def compose(self) -> ComposeResult:
        yield ListView()

    def focus_content(self) -> None:
        """Focus the first dataset item."""
        self.query(ListView).focus()

    def on_mount(self) -> None:
        """Load and display datasets when the pane is mounted."""
        app: MyApp = self.app  # type: ignore
        config = app.config
        datasets = scan_datasets(config.splatflow_data_root)

        list_view = self.query_one(ListView)
        for dataset in datasets:
            list_view.append(DatasetListItem(dataset))

    def action_import_dataset(self):
        pass
