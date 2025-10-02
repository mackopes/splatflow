import dataclasses
import datetime
from typing import List


@dataclasses.dataclass
class ProcessedDataset:
    name: str


@dataclasses.dataclass
class Dataset:
    name: str
    created_at: datetime.datetime
    n_images: int = dataclasses.field(default=0)
    processed_datasets: List[ProcessedDataset] = dataclasses.field(default_factory=list)
    valid: bool = dataclasses.field(default=False)
