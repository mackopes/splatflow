import typer
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from splatflow.process import hloc
from splatflow.process.hloc import Feature, Matcher, MatchingMethod, CameraModel


@dataclass
class HlocCommandSettings:
    """Type-safe builder for hloc processing commands."""

    dataset_dir: Path
    output_dir: Path
    matching_method: MatchingMethod = "vocab_tree"
    feature_type: Feature = "superpoint_max"
    matcher_type: Matcher = "superglue"
    num_matched: int = 200
    use_single_camera_mode: bool = False
    camera_model: CameraModel = CameraModel.OPENCV

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_dir": str(self.dataset_dir),
            "output_dir": str(self.output_dir),
            "matching_method": self.matching_method,
            "feature_type": self.feature_type,
            "matcher_type": self.matcher_type,
            "num_matched": self.num_matched,
            "use_single_camera_mode": self.use_single_camera_mode,
            "camera_model": self.camera_model.value,
        }

    def build(self) -> List[str]:
        """Build the CLI command list."""
        cmd = [
            "poetry",
            "run",
            "process",
            str(self.dataset_dir),
            str(self.output_dir),
            "--matching-method",
            self.matching_method,
            "--feature-type",
            self.feature_type,
            "--matcher-type",
            self.matcher_type,
            "--num-matched",
            str(self.num_matched),
        ]

        if self.use_single_camera_mode:
            cmd.append("--use-single-camera-mode")

        cmd.extend(["--camera-model", self.camera_model.value])

        return cmd


def main():
    typer.run(hloc.run_hloc)
