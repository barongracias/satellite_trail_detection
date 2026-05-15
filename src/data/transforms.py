"""Image and mask transforms for patch-based training."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from PIL import Image


class _RandomRotate90(v2.Transform):
    """Rotate by a uniformly random multiple of 90 degrees."""

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"angle": int(torch.randint(0, 4, ())) * 90}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return v2.functional.rotate(inpt, params["angle"])


_GEO_PIPELINE = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        _RandomRotate90(),
    ]
)


class JointTransform:
    """Apply identical random geometric transforms to a (image, mask) PIL pair.

    Both inputs receive the same random flip and rotation decisions because
    torchvision.transforms.v2 draws parameters once and reuses them for every
    element passed in a single forward call.
    """

    def __init__(self, augment: bool = True) -> None:
        self.augment = augment

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_t = TF.pil_to_tensor(image)   # [C, H, W] uint8
        mask_t = TF.pil_to_tensor(mask)   # [C, H, W] uint8

        img_tv = tv_tensors.Image(img_t)
        mask_tv = tv_tensors.Mask(mask_t)

        if self.augment:
            img_tv, mask_tv = _GEO_PIPELINE(img_tv, mask_tv)

        image_out = img_tv.as_subclass(torch.Tensor).float() / 255.0
        mask_out = (mask_tv.as_subclass(torch.Tensor) > 0).float()
        return image_out, mask_out


def get_train_transforms() -> JointTransform:
    """Return a joint transform applying random geometric augmentation to image and mask."""
    return JointTransform(augment=True)


def get_eval_transforms() -> JointTransform:
    """Return a joint transform that converts PIL to tensor with no augmentation."""
    return JointTransform(augment=False)


def get_patch_transforms() -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """Return the legacy separate image and mask transforms for SatelliteTrailPatchDataset."""
    image_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    mask_transform = T.ToTensor()
    return image_transform, mask_transform
