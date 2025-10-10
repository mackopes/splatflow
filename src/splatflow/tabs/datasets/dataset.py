import dataclasses
import datetime
from typing import Any, Dict, List


@dataclasses.dataclass
class ProcessedDataset:
    name: str
    settings: Dict[str, Any]


@dataclasses.dataclass
class SplatflowData:
    """Structure for splatflow_data.json file."""

    datasets: List[ProcessedDataset] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "datasets": [
                {"name": d.name, "settings": d.settings} for d in self.datasets
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplatflowData":
        """Create from dictionary loaded from JSON."""
        datasets = [
            ProcessedDataset(name=d["name"], settings=d["settings"])
            for d in data.get("datasets", [])
        ]
        return cls(datasets=datasets)


@dataclasses.dataclass
class Dataset:
    name: str
    created_at: datetime.datetime
    n_images: int = dataclasses.field(default=0)
    processed_datasets: List[ProcessedDataset] = dataclasses.field(default_factory=list)
    valid: bool = dataclasses.field(default=False)
