import pathlib
from typing import List, Union

from .dataset import Dataset

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
        dataset = Dataset.load_from_directory(dataset_dir)
        datasets.append(dataset)

    return datasets
