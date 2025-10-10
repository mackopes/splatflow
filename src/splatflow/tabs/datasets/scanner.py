import datetime
import pathlib
from typing import List, Union
import json

from .dataset import Dataset, ProcessedDataset, SplatflowData

# Image file extensions to count
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def scan_datasets(data_root: Union[str, pathlib.Path]) -> List[Dataset]:
    """Scan the datasets directory and return list of datasets with metadata.

    Args:
        data_root: Path to splatflow data root directory

    Returns:
        List of Dataset objects with validation and image counts
    """
    datasets_dir = pathlib.Path(data_root) / "datasets"

    if not datasets_dir.exists():
        return []

    datasets: List[Dataset] = []

    for dataset_dir in datasets_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        input_dir = dataset_dir / "input"
        n_images = 0
        valid = False

        # Check if input directory exists and count images
        if input_dir.exists() and input_dir.is_dir():
            # Count image files
            n_images = sum(
                1
                for f in input_dir.iterdir()
                if f.is_file() and f.suffix in IMAGE_EXTENSIONS
            )
            valid = n_images > 0

        # Get creation time from directory
        created_at = datetime.datetime.fromtimestamp(dataset_dir.stat().st_ctime)

        processed_datasets = _collect_processed_datasets(dataset_dir)

        datasets.append(
            Dataset(
                name=dataset_dir.name,
                created_at=created_at,
                n_images=n_images,
                valid=valid,
                processed_datasets=processed_datasets,
            )
        )

    return datasets


def _collect_processed_datasets(dataset_dir: pathlib.Path) -> List[ProcessedDataset]:
    """Collect processed dataset variants from splatflow_data.json.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        List of ProcessedDataset objects representing different processing configurations
    """
    data_file_path = dataset_dir / "splatflow_data.json"

    if not data_file_path.is_file():
        return []

    try:
        with open(data_file_path, "r") as f:
            data = json.load(f)

        splatflow_data = SplatflowData.from_dict(data)
        return splatflow_data.datasets

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Log error and return empty list if JSON is malformed
        print(f"Warning: Could not parse {data_file_path}: {e}")
        return []
