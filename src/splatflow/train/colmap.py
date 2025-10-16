import functools
import os
import pathlib
import shutil
from typing import Any, Dict, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import tqdm
from PIL import Image

from splatflow.colmap_utils import ColmapTransforms
from splatflow.train.math_utils import rot_x


class TransformsDataset(torch.utils.data.Dataset):
    """A Dataset class to load images from transforms.json"""

    def __init__(
        self,
        transforms_path: pathlib.Path,
        patch_size: Optional[int] = None,
        factor: int = 1,
    ):
        self.patch_size = patch_size
        self.transforms = ColmapTransforms.load(transforms_path)
        self.transforms_path = transforms_path
        self.transforms_dir = transforms_path.parent
        self.factor = factor
        self._c2w_cache: Dict[int, torch.Tensor] = {}

        self._prepare_dataset()

    @functools.cached_property
    def _undistorted_image_dir(self) -> pathlib.Path:
        # TODO: maybe create our own transforms.json where the image dir is a standalone top level field
        abs_first_image_path = (
            self.transforms_path.parents[0] / self.transforms.frames[0].file_path
        )

        image_dir = abs_first_image_path.parents[0]

        undistorted_image_dir = image_dir.parents[0] / f"undistorted_{self.factor}"
        return undistorted_image_dir

    @functools.cached_property
    def K(self):
        return np.array(
            [
                [self.transforms.fl_x, 0, self.transforms.cx],
                [0, self.transforms.fl_y, self.transforms.cy],
                [0, 0, 1],
            ]
        )

    @functools.cached_property
    def _params(self):
        return np.array(
            [
                self.transforms.k1,
                self.transforms.k2,
                self.transforms.p1,
                self.transforms.p2,
            ],
            dtype=np.float32,
        )

    def c2w(self, index: int) -> torch.Tensor:
        if index in self._c2w_cache:
            return self._c2w_cache[index]

        frame = self.transforms.frames[index]
        align_matrix = np.eye(4)
        align_matrix[:3, :3] = rot_x(-np.pi / 2)
        c2w = torch.from_numpy(
            align_matrix @ frame.transform_matrix @ np.diag([1, -1, -1, 1])
        ).float()

        self._c2w_cache[index] = c2w

        return c2w

    @functools.cache
    def _K_undist_roi_undist(self):
        width = self.transforms.width // self.factor
        height = self.transforms.height // self.factor

        # TODO: Handle params not present and different camera types
        # TODO: Handle different camera sizes

        K = self.K
        K[:2, :] /= self.factor

        K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
            K, self._params, (width, height), 0
        )

        return K_undist, roi_undist

    def _prepare_dataset(self):
        if self._undistorted_image_dir.is_dir():
            # Check if undistorted dataset has the correct number of images
            num_undistorted = len(list(self._undistorted_image_dir.iterdir()))
            num_frames = len(self.transforms.frames)
            if num_undistorted < num_frames:
                print(
                    f"Undistorted dataset has {num_undistorted} images but expected {num_frames}. Deleting and re-undistorting."
                )
                shutil.rmtree(self._undistorted_image_dir)
                self._undistort_images()
            else:
                print(
                    f"Undistorted dataset already exists with {num_undistorted} images"
                )
        else:
            self._undistort_images()
        print("undistorting done")
        print(self._undistorted_image_dir)

    def _undistort_images(self):
        if self._undistorted_image_dir.is_dir():
            raise ValueError(f"{self._undistorted_image_dir} already exists")

        print(f"undistoring images to {self._undistorted_image_dir}")
        os.makedirs(self._undistorted_image_dir)

        width = self.transforms.width
        height = self.transforms.height

        K_undist, roi_undist = self._K_undist_roi_undist()

        mapx, mapy = cv2.initUndistortRectifyMap(
            self.K,
            self._params,
            None,  # type: ignore
            K_undist,
            (width, height),
            cv2.CV_32FC1,  # type: ignore
        )

        # TODO: Paralelise this
        for frame in tqdm.tqdm(self.transforms.frames):
            image_path = self.transforms_dir / frame.file_path
            image = imageio.imread(image_path)[..., :3]
            if self.factor != 1:
                resized_size = (
                    int(round(image.shape[1] // self.factor)),
                    int(round(image.shape[0] // self.factor)),
                )
                image = np.array(
                    Image.fromarray(image).resize(resized_size, Image.BICUBIC)  # type: ignore
                )

            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi_undist
            image = image[y : y + h, x : x + w]
            undistorted_image_path = self._undistorted_image_dir / image_path.name

            imageio.imwrite(undistorted_image_path, image)

    def __len__(self):
        return len(self.transforms.frames)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        frame = self.transforms.frames[item]
        undistorted_image_path = self._undistorted_image_dir / frame.file_path.name
        image = imageio.imread(undistorted_image_path)

        K_undist, _ = self._K_undist_roi_undist()

        data = {
            "K": torch.from_numpy(K_undist).float(),
            "camtoworld": self.c2w(item),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }

        return data
