import typer
from dataclasses import dataclass
from pathlib import Path
from typing import List

from splatflow.process import hloc
from splatflow.process.hloc import Feature, Matcher, MatchingMethod, CameraModel


@dataclass
class HlocCommandSettings:
    """Type-safe builder for hloc processing commands."""

    images_dir: Path
    output_dir: Path
    matching_method: MatchingMethod = "vocab_tree"
    feature_type: Feature = "superpoint_max"
    matcher_type: Matcher = "superglue"
    num_matched: int = 200
    use_single_camera_mode: bool = False
    camera_model: CameraModel = CameraModel.OPENCV

    def build(self) -> List[str]:
        """Build the CLI command list."""
        cmd = [
            "poetry",
            "run",
            "process",
            str(self.images_dir),
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
