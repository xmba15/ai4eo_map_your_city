import random
from typing import Tuple

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

__all__ = ("S2RandomRotation", "S2PermuteConcat")


class S2RandomRotation(ImageOnlyTransform):
    def __init__(
        self,
        limits: Tuple[float, float],
        always_apply=False,
        p=0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.limits = limits

    def apply(self, image, angle=0, **params):
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(
            (width // 2, height // 2),
            angle,
            1.0,
        )

        for channel in range(image.shape[2]):
            image[:, :, channel] = cv2.warpAffine(
                image[:, :, channel],
                rotation_matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        return image

    def get_params(self):
        return {"angle": random.uniform(self.limits[0], self.limits[1])}

    def get_transform_init_args_names(self):
        return ("limits",)


class S2PermuteConcat(ImageOnlyTransform):
    def __init__(
        self,
        random_permutation=False,
    ):
        super().__init__(p=1.0, always_apply=True)
        self.random_permutation = random_permutation

    def apply(self, image, **params):
        height, width, channel = image.shape[:3]
        assert (height, width, channel) == (64, 64, 12)

        channel_indices = np.arange(channel)
        if self.random_permutation:
            np.random.shuffle(channel_indices)
        group_indices = [
            channel_indices[:4],
            channel_indices[4:8],
            channel_indices[8:],
        ]

        def concatenate_group(g_indices):
            top_row = np.concatenate((image[..., g_indices[0]], image[..., g_indices[1]]), axis=1)
            bottom_row = np.concatenate((image[..., g_indices[2]], image[..., g_indices[3]]), axis=1)
            return np.concatenate((top_row, bottom_row), axis=0)

        output = np.empty((256, 256, 3), dtype=image.dtype)
        for i in range(3):
            cur_indices = group_indices[i]
            output[:, :, i] = np.tile(
                concatenate_group(cur_indices),
                (2, 2),
            )

        return output

    def get_transform_init_args_names(self):
        return ("random_permutation",)
