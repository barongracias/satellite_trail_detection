"""Default image and mask transforms for patch-based training."""

from __future__ import annotations

from typing import Any, Callable

import torchvision.transforms as T


def get_patch_transforms() -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """
    Create the default transforms for grayscale image and mask patches.

    Returns
    -------
    tuple[Callable[[Any], Any], Callable[[Any], Any]]
        Transform pair for image patches and mask patches.
    """
    image_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    mask_transform = T.ToTensor()
    return image_transform, mask_transform
