import dataclasses
import enum
import json
import pathlib
from typing import Any, Dict, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ProcessedModelState(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"
    READY = "ready"


@dataclasses.dataclass
class ProcessedModel:
    name: str
    dataset_name: str
    state: ProcessedModelState
    settings: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "state": self.state.value,  # Convert enum to string
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedModel":
        """Create from dictionary loaded from JSON."""
        return cls(
            name=data["name"],
            dataset_name=data["dataset_name"],
            state=ProcessedModelState(data["state"]),  # Convert string to enum
            settings=data["settings"],
        )


@dataclasses.dataclass
class SplatflowModelData:
    """Structure for splatflow_data.json file in models directory."""

    models: List[ProcessedModel] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {"models": [m.to_dict() for m in self.models]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplatflowModelData":
        """Create from dictionary loaded from JSON."""
        models = [ProcessedModel.from_dict(m) for m in data.get("models", [])]
        return cls(models=models)

    def save(self, path: pathlib.Path):
        """Save to JSON file."""
        with open(path, mode="w") as f:
            json.dump(self.to_dict(), f, indent=2)
