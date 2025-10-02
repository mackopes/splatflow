from pathlib import Path
from typing import List
from .dataset import Dataset
import datetime


# Image file extensions to count
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def scan_datasets(data_root: str) -> List[Dataset]:
    """Scan the datasets directory and return list of datasets with metadata.

    Args:
        data_root: Path to splatflow data root directory

    Returns:
        List of Dataset objects with validation and image counts
    """
    datasets_dir = Path(data_root) / "datasets"

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

        datasets.append(
            Dataset(
                name=dataset_dir.name,
                created_at=created_at,
                n_images=n_images,
                valid=valid,
            )
        )

    return datasets
