import pathlib
from typing import Literal
from enum import Enum

from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    pairs_from_retrieval,
    reconstruction,
)

import pycolmap

Feature = Literal[
    "sift",
    "superpoint_aachen",
    "superpoint_max",
    "superpoint_inloc",
    "r2d2",
    "d2net-ss",
    "sosnet",
    "disk",
]

Matcher = Literal[
    "superglue",
    "superglue-fast",
    "NN-superpoint",
    "NN-ratio",
    "NN-mutual",
    "adalam",
    "disk+lightglue",
    "superpoint+lightglue",
]

MatchingMethod = Literal["vocab_tree", "exhaustive", "sequential"]


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"
    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"


def run_hloc(
    images_dir: pathlib.Path,
    output_dir: pathlib.Path,
    matching_method: MatchingMethod = "vocab_tree",
    feature_type: Feature = "superpoint_max",
    matcher_type: Matcher = "superglue",
    num_matched: int = 200,
    use_single_camera_mode: bool = False,
    camera_model: CameraModel = CameraModel.OPENCV,
):
    sfm_pairs = output_dir / "pairs-netvlad.txt"
    sfm_dir = output_dir / "sparse" / "0"
    features = output_dir / "features.h5"
    matches = output_dir / "matches.h5"

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs[feature_type]
    matcher_conf = match_features.confs[matcher_type]

    references = [p.relative_to(images_dir).as_posix() for p in images_dir.iterdir()]
    extract_features.main(
        feature_conf, images_dir, image_list=references, feature_path=features
    )

    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    else:
        retrieval_path = extract_features.main(retrieval_conf, images_dir, output_dir)
        num_matched = min(len(references), num_matched)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)

    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    if use_single_camera_mode:  # one camera per all frames
        camera_mode = pycolmap.CameraMode.SINGLE  # type: ignore
    else:  # one camera per frame
        camera_mode = pycolmap.CameraMode.PER_IMAGE  # type: ignore

    image_options = pycolmap.ImageReaderOptions(camera_model=camera_model.value)

    reconstruction.main(
        sfm_dir,
        images_dir,
        sfm_pairs,
        features,
        matches,
        camera_mode=camera_mode,
        image_options=image_options,  # type: ignore Hloc has a wrong type definition.
        verbose=False,
    )
