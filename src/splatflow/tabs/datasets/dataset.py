import dataclasses
import datetime
import json
import os
import pathlib
from typing import Any, Dict, List, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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

    def save(self, path: pathlib.Path):
        with open(path, mode="w") as f:
            json.dump(self.to_dict(), f)


@dataclasses.dataclass
class Dataset:
    name: str
    dataset_dir: pathlib.Path
    created_at: datetime.datetime
    n_images: int = dataclasses.field(default=0)
    valid: bool = dataclasses.field(default=False)

    @property
    def _splatflow_data_path(self):
        return self.dataset_dir / "splatflow_data.json"

    def _load_splatflow_data(self) -> SplatflowData | None:
        if not self._splatflow_data_path.is_file():
            return None

        try:
            with open(self._splatflow_data_path, "r") as f:
                data = json.load(f)

            splatflow_data = SplatflowData.from_dict(data)
            return splatflow_data

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Log error and return empty list if JSON is malformed
            print(f"Warning: Could not parse {self._splatflow_data_path}: {e}")
            return None

    def _load_or_create_splatflow_data(self) -> SplatflowData:
        splatflow_data = self._load_splatflow_data()
        if splatflow_data is not None:
            return splatflow_data

        splatflow_data = SplatflowData([])
        splatflow_data.save(self._splatflow_data_path)
        return splatflow_data

    @classmethod
    def load_from_directory(
        cls, dataset_dir: pathlib.Path, name: Optional[str] = None
    ) -> "Dataset":
        if not dataset_dir.is_dir():
            raise ValueError(f"Directory {dataset_dir} does not exist")

        _name = name if name else os.path.basename(dataset_dir)
        created_at = datetime.datetime.fromtimestamp(dataset_dir.stat().st_ctime)

        input_dir = dataset_dir / "input"
        valid = False
        n_images = 0

        if input_dir.is_dir():
            n_images = sum(
                1
                for f in input_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            valid = n_images > 0

        return cls(_name, dataset_dir, created_at, n_images, valid)

    @property
    def processed_datasets(self) -> List[ProcessedDataset]:
        splatflow_data = self._load_splatflow_data()
        if splatflow_data is None:
            return []

        return splatflow_data.datasets

    def add_processed_dataset(self, processed_dataset: ProcessedDataset):
        splatflow_data = self._load_or_create_splatflow_data()
        splatflow_data.datasets.append(processed_dataset)
        splatflow_data.save(self._splatflow_data_path)
