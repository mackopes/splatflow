import json
import pathlib
from dataclasses import dataclass
from typing import List, Union

from .model import ProcessedModel, SplatflowModelData


@dataclass
class DatasetModels:
    """Container for a dataset's trained models."""

    dataset_name: str
    models: List[ProcessedModel]


def scan_models(data_root: Union[str, pathlib.Path]) -> List[DatasetModels]:
    """Scan the models directory and return list of dataset models.

    Args:
        data_root: Path to splatflow data root directory

    Returns:
        List of DatasetModels objects containing trained models per dataset
    """
    models_dir = pathlib.Path(data_root) / "models"

    if not models_dir.exists():
        return []

    dataset_models: List[DatasetModels] = []

    for dataset_dir in models_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        # Check for splatflow_data.json
        metadata_path = dataset_dir / "splatflow_data.json"
        if not metadata_path.exists():
            continue

        # Load the metadata
        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)

            model_data = SplatflowModelData.from_dict(data)

            dataset_models.append(
                DatasetModels(
                    dataset_name=dataset_dir.name,
                    models=model_data.models,
                )
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Skip datasets with malformed JSON
            print(f"Warning: Could not parse {metadata_path}: {e}")
            continue

    return dataset_models
