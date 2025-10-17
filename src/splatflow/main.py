from typing import Callable, List, Tuple, Type
import asyncio
import os
import time

from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Footer, TabbedContent, TabPane

from splatflow.config import Config, load_config
from splatflow.initialise import initialise
from splatflow.worker.queue_manager import QueueItem
from splatflow.tabs.datasets.datasets import DatasetsPane
from splatflow.tabs.models.models import ModelsPane
from splatflow.tabs.queue.queue import QueuePane
from splatflow.tabs.flow_tab import FlowTab

TABS: List[Tuple[str, Type[FlowTab]]] = [
    ("Datasets", DatasetsPane),
    ("Models", ModelsPane),
    ("Queue", QueuePane),
]


class SplatflowApp(App):
    CSS_PATH = [
        "app.tcss",
        "tabs/datasets/datasets.tcss",
        "tabs/queue/queue.tcss",
        "tabs/models/models.tcss",
        "components/delete_dialog.tcss",
    ]

    BINDINGS = [
        ("t", "next_tab", "Next tab"),
    ]

    # Queue management
    queue: reactive[List[QueueItem]] = reactive(list, init=False)
    _processing: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.queue = []

    def compose(self) -> ComposeResult:
        with TabbedContent():
            for tab_name, tab_pane in TABS:
                with TabPane(tab_name, id=tab_name.lower()):
                    yield tab_pane()
        yield Footer()

    def on_mount(self) -> None:
        """Start the queue processing worker when app mounts."""
        self.run_worker(self.process_queue_worker, exclusive=True)

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

    # TODO: refactor queue items to a standalone file
    def add_to_queue(
        self,
        name: str,
        command: List[str],
        before_exec_callback: Callable[[], None] | None = None,
        on_success_callback: Callable[[], None] | None = None,
        on_error_callback: Callable[[], None] | None = None,
    ) -> QueueItem:
        """Add a new item to the processing queue."""
        print("Adding item to queue")
        item = QueueItem.create(
            name=name,
            command=command,
            before_exec_callback=before_exec_callback,
            on_success_callback=on_success_callback,
            on_error_callback=on_error_callback,
        )
        self.queue.append(item)
        self.mutate_reactive(SplatflowApp.queue)
        return item

    async def _schedule_delayed_update(self, delay: float, state: dict) -> None:
        """Schedule a delayed UI update that can be cancelled."""
        try:
            await asyncio.sleep(delay)
            # Only update if this task wasn't cancelled
            if state.get("pending_task") == asyncio.current_task():
                self.mutate_reactive(SplatflowApp.queue)
                state["last_update_time"] = time.time()
                state["pending_task"] = None
        except asyncio.CancelledError:
            pass  # Task was cancelled, no update needed

    async def process_queue_worker(self) -> None:
        """Background worker that processes queue items."""
        while True:
            # Find next pending item
            pending_item = next(
                (item for item in self.queue if item.status == "pending"), None
            )

            if pending_item and not self._processing:
                print("found pending item!")
                self._processing = True
                await self.process_queue_item(pending_item)
                self._processing = False
            else:
                # No pending items, sleep a bit
                await asyncio.sleep(1)

    async def process_queue_item(self, item: QueueItem) -> None:
        """Process a single queue item using subprocess."""
        item.mark_running()
        self.mutate_reactive(SplatflowApp.queue)

        # Call before execution callback
        if item.before_exec_callback:
            item.before_exec_callback()

        try:
            # Create subprocess with pipes for stdout and stderr
            process = await asyncio.create_subprocess_exec(
                *item.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # TODO: Specify the window width (ncols) for TQDM
                # TODO Remove the CC and CXX from here
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "2",
                    "TQDM_MININTERVAL": "1",
                    "TQDM_ASCII": "True",
                    "CC": "gcc-13",
                    "CXX": "g++-13",
                },
            )

            # Helper to read stream byte-by-byte for real-time output
            async def read_stream(stream, prefix=""):
                buffer = ""
                min_interval = 1.0  # 1 second between UI updates
                # Shared state that can be modified by both trigger_update and scheduled task
                state = {"last_update_time": 0, "pending_task": None}

                async def trigger_update():
                    """Trigger UI update with debouncing."""
                    current_time = time.time()

                    # Cancel any pending update
                    if state["pending_task"]:
                        state["pending_task"].cancel()
                        state["pending_task"] = None

                    # Check if we can update immediately
                    if current_time - state["last_update_time"] >= min_interval:
                        self.mutate_reactive(SplatflowApp.queue)
                        state["last_update_time"] = current_time
                    else:
                        # Schedule delayed update
                        delay = min_interval - (
                            current_time - state["last_update_time"]
                        )
                        task = asyncio.create_task(
                            self._schedule_delayed_update(delay, state)
                        )
                        state["pending_task"] = task

                while True:
                    # Read one byte at a time for immediate output
                    byte = await stream.read(1)
                    if not byte:
                        # Flush any remaining buffer content
                        if buffer.strip():
                            output_line = f"{prefix}{buffer}" if prefix else buffer
                            item.add_output(output_line)
                        # Cancel pending and force final update
                        if state["pending_task"]:
                            state["pending_task"].cancel()
                        self.mutate_reactive(SplatflowApp.queue)
                        break

                    try:
                        char = byte.decode()
                    except UnicodeDecodeError:
                        # Skip invalid bytes
                        continue

                    # Emit on newline or carriage return (for tqdm)
                    if char == "\n":
                        if buffer.strip():
                            output_line = f"{prefix}{buffer}" if prefix else buffer
                            item.add_output(output_line)
                            await trigger_update()
                        buffer = ""
                    elif char == "\r":
                        if buffer.strip():
                            output_line = f"{prefix}{buffer}" if prefix else buffer
                            # for \r we want to overwrite the last line
                            item.output[-1] = output_line
                            await trigger_update()
                        buffer = ""
                    else:
                        buffer += char

            # Read stdout and stderr concurrently
            await asyncio.gather(
                read_stream(process.stdout),
                read_stream(process.stderr, "[STDERR] "),
            )

            # Wait for process to complete
            returncode = await process.wait()

            if returncode == 0:
                item.mark_completed()
                # Call success callback
                if item.on_success_callback:
                    item.on_success_callback()
            else:
                print(f"Process exited with code {returncode}")
                item.mark_failed(f"Process exited with code {returncode}")
                # Call error callback
                if item.on_error_callback:
                    item.on_error_callback()

        except Exception as e:
            item.mark_failed(f"Error: {str(e)}")
            # Call error callback
            if item.on_error_callback:
                item.on_error_callback()

        self.mutate_reactive(SplatflowApp.queue)

    def switch_to_queue_tab(self) -> None:
        """Switch to the queue tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "queue"

    def refresh_models_tab(self) -> None:
        """Refresh the models tab to show updated data.

        Note: Alternative approach would be using Textual's message system:
        - Define custom ModelsChanged(Message) class
        - Post message: self.app.post_message(ModelsChanged())
        - Handle in ModelsPane: def on_models_changed(self, message: ModelsChanged)
        """
        from splatflow.tabs.models.models import ModelsPane

        try:
            models_pane = self.query_one(ModelsPane)
            models_pane.rescan_and_repopulate()
        except NoMatches:
            pass  # Models tab not mounted yet


def main():
    config = load_config()
    initialise(config)
    app = SplatflowApp(config)
    app.theme = "catppuccin-mocha"
    app.run()


if __name__ == "__main__":
    main()
