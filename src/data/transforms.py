"""Image and mask transforms for patch-based training."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from PIL import Image


NORMALISATION_MODES = ("fixed", "per_image", "full_image")


class SignalDependentNoise:
    """Add calibrated signal-dependent Gaussian noise to a [0, 1] image tensor."""

    def __init__(self, alpha: float, beta: float, multiplier: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.multiplier = float(multiplier)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image_nonnegative = image.clamp_min(0.0)
        variance = self.alpha * image_nonnegative + self.beta ** 2
        std = self.multiplier * torch.sqrt(variance)
        noisy = image + torch.randn_like(image) * std
        return noisy.clamp(0.0, 1.0)


def normalise_tensor(
    image: torch.Tensor,
    mode: str,
    full_image_mean: float | None = None,
    full_image_std: float | None = None,
) -> torch.Tensor:
    """Apply the requested normalisation to a [0, 1] image tensor.

    ``fixed`` is the paper-agnostic default ``(x - 0.5) / 0.5``. ``per_image``
    is a per-patch z-score. ``full_image`` is a per-source-image z-score from
    pre-computed stats; if either stat is missing or non-finite it falls back
    to the per-patch z-score so a partial manifest cannot silently produce
    NaN outputs.
    """
    if mode == "per_image":
        return (image - image.mean()) / (image.std() + 1e-6)
    if mode == "full_image":
        if (
            full_image_mean is not None
            and full_image_std is not None
            and not pd.isna(full_image_mean)
            and not pd.isna(full_image_std)
        ):
            return (image - float(full_image_mean)) / (float(full_image_std) + 1e-6)
        return (image - image.mean()) / (image.std() + 1e-6)
    return (image - 0.5) / 0.5


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


