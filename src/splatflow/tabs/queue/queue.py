from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import ListView, ListItem, Label, RichLog
from textual.reactive import var
from textual.strip import Strip
from rich.segment import Segment

from splatflow.tabs.flow_tab import FlowTab

if TYPE_CHECKING:
    from splatflow.main import SplatflowApp


class QueuePane(FlowTab):
    """Queue pane showing processing tasks and their output."""

    selected_item_id: var[str | None] = var(None)
    # _last_output_length: int = 0

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="queue-list-container") as v:
                v.border_title = "Queue"
                yield ListView(id="queue-list")
            with Vertical(id="queue-output-container") as v:
                v.border_title = "Output"
                log = RichLog(id="queue-output", wrap=True, highlight=True)
                log.can_focus = False
                yield log

    def focus_content(self) -> None:
        """Focus the queue list."""
        self.query_one("#queue-list", ListView).focus()

    def on_mount(self) -> None:
        """Set up queue watching."""
        app: SplatflowApp = self.app  # type: ignore
        # Watch for queue changes
        self.watch(app, "queue", self.on_queue_changed)

    def on_queue_changed(self, queue) -> None:
        """Handle queue changes."""
        list_view = self.query_one("#queue-list", ListView)

        # Create a mapping of queue item IDs to their data
        queue_items_map = {item.id: item for item in queue}

        # Track which list items to keep
        existing_ids = set()

        # Update existing items or mark for removal
        for idx in range(len(list_view) - 1, -1, -1):
            list_item = list_view.children[idx]
            item_id = getattr(list_item, "item_id", None)

            if item_id in queue_items_map:
                # Update the label if the item still exists
                queue_item = queue_items_map[item_id]
                status_symbols = {
                    "pending": "⏳",
                    "running": "▶️",
                    "completed": "✅",
                    "failed": "❌",
                }
                symbol = status_symbols.get(queue_item.status, "?")
                new_label_text = f"{symbol} {queue_item.name}"

                # Update the label text
                label = list_item.query_one(Label)
                label.update(new_label_text)

                existing_ids.add(item_id)
            else:
                # Item no longer in queue, remove it
                list_item.remove()

        # Add new items that aren't in the ListView yet
        for item in queue:
            if item.id not in existing_ids:
                status_symbols = {
                    "pending": "⏳",
                    "running": "▶️",
                    "completed": "✅",
                    "failed": "❌",
                }
                symbol = status_symbols.get(item.status, "?")
                label = f"{symbol} {item.name}"

                list_item = ListItem(Label(label))
                list_item.item_id = item.id  # type: ignore
                list_view.append(list_item)

        # Update output if we're viewing a selected item
        if self.selected_item_id:
            self.update_output()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle queue item selection."""
        if event.item and hasattr(event.item, "item_id"):
            self.selected_item_id = event.item.item_id  # type: ignore
            # self._last_output_length = 0  # Reset when selecting a new item
            self.update_output()

    def update_output(self) -> None:
        """Update the output log for the selected item."""
        if not self.selected_item_id:
            return

        app: SplatflowApp = self.app  # type: ignore
        queue_item = next(
            (item for item in app.queue if item.id == self.selected_item_id), None
        )

        if not queue_item:
            return

        log = self.query_one("#queue-output", RichLog)

        current_length = len(queue_item.output)
        log_length = len(log.lines)

        if current_length < log_length:
            log.clear()
            for line in queue_item.output:
                log.write(line)
        else:
            for i in range(log_length):
                log.lines[i] = Strip([Segment(queue_item.output[i])])
            for line in queue_item.output[log_length:]:
                log.write(line)
