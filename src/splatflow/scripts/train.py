from splatflow.train.gsplat_simple_trainer import entrypoint
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class GsplatCommandSettings:
    """Type-safe builder for hloc processing commands."""

    dataset_dir: Path  # data-dir
    output_dir: Path  # result-dir

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_dir": str(self.dataset_dir),
            "output_dir": str(self.output_dir),
        }

    def build(self) -> List[str]:
        """Build the CLI command list."""
        cmd = [
            "poetry",
            "run",
            "train",
            "default",
            "--data-dir",
            self.dataset_dir,
            "--result-dir",
            self.output_dir,
            "--save-ply",
            "--steps-scaler",
            "0.1",
        ]

        return cmd


def main():
    entrypoint()
