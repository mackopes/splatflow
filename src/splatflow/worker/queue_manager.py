"""Queue manager for handling background processing tasks."""

import dataclasses
import datetime
import uuid
from typing import Callable, Literal


@dataclasses.dataclass
class QueueItem:
    """Represents a single item in the processing queue."""

    id: str
    name: str  # Display name (e.g., "Process: dataset_name")
    command: list[str]  # Command to run as subprocess
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    output: list[str] = dataclasses.field(default_factory=list)  # Captured output lines
    error: str | None = None  # Error message if failed
    created_at: datetime.datetime = dataclasses.field(
        default_factory=datetime.datetime.now
    )
    started_at: datetime.datetime | None = None
    completed_at: datetime.datetime | None = None

    # Lifecycle callbacks
    before_exec_callback: Callable[[], None] | None = None
    on_success_callback: Callable[[], None] | None = None
    on_error_callback: Callable[[], None] | None = None

    @classmethod
    def create(
        cls,
        name: str,
        command: list[str],
        before_exec_callback: Callable[[], None] | None = None,
        on_success_callback: Callable[[], None] | None = None,
        on_error_callback: Callable[[], None] | None = None,
    ) -> "QueueItem":
        """Create a new queue item with a generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            command=command,
            before_exec_callback=before_exec_callback,
            on_success_callback=on_success_callback,
            on_error_callback=on_error_callback,
        )

    def add_output(self, line: str) -> None:
        """Add a line to the output."""
        self.output.append(line)

    def mark_running(self) -> None:
        """Mark the item as running."""
        self.status = "running"
        self.started_at = datetime.datetime.now()

    def mark_completed(self) -> None:
        """Mark the item as completed."""
        self.status = "completed"
        self.completed_at = datetime.datetime.now()

    def mark_failed(self, error: str) -> None:
        """Mark the item as failed with an error message."""
        self.status = "failed"
        self.error = error
        self.completed_at = datetime.datetime.now()

    @property
    def duration(self) -> float | None:
        """Get the duration in seconds if the item has completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
