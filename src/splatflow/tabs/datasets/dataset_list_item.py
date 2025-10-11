from textual.app import ComposeResult
from textual.containers import HorizontalGroup
from textual.widgets import ListItem, Label, Tree

from .dataset import Dataset


class DatasetListItem(ListItem):
    """A widget to display dataset information in the list."""

    can_focus = True

    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def compose(self) -> ComposeResult:
        with HorizontalGroup():
            name_prefix = "" if self.dataset.processed_datasets else "* "
            yield Label(name_prefix + self.dataset.name, classes="dataset-name")
            yield Label(
                self.dataset.created_at.strftime("%Y-%m-%d"),
                classes="dataset-date",
            )
            yield Label(
                f"{self.dataset.n_images} images",
                classes="dataset-n-images",
            )
