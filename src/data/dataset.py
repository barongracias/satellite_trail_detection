"""Patch-based dataset utilities for satellite trail segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.indexing import (
    PatchDatasetConfig,
    PatchIndexEntry,
    ProcessedImagePair,
    build_patch_index,
    discover_image_mask_pairs,
)


Image.MAX_IMAGE_PIXELS = None


class SatelliteTrailPatchDataset(Dataset[dict[str, Any]]):
    """Load paired image and mask patches from large astronomical frames."""

    def __init__(
        self,
        root_dir: str | Path | PatchDatasetConfig,
        patch_size: int = 512,
        stride: int | None = None,
        strict_pairing: bool = True,
        image_transform: Callable[[Image.Image], Any] | None = None,
        mask_transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        """
        Initialise the dataset from a directory of processed PNG pairs.

        Parameters
        ----------
        root_dir:
            Directory containing `*_red.fits_full.png` images and matching
            `*_red_mask.png` masks, or a pre-built dataset configuration.
        patch_size:
            Square patch size in pixels.
        stride:
            Patch stride in pixels. Defaults to `patch_size`.
        strict_pairing:
            When true, raise an error if any processed image is missing its mask.
        image_transform:
            Optional transform applied to image patches.
        mask_transform:
            Optional transform applied to mask patches.
        """
        if isinstance(root_dir, PatchDatasetConfig):
            self.config = root_dir
        else:
            self.config = PatchDatasetConfig(
                root_dir=Path(root_dir),
                patch_size=patch_size,
                stride=stride,
                strict_pairing=strict_pairing,
            )

        self.root_dir = self.config.root_dir
        self.patch_size = self.config.patch_size
        self.stride = self.config.resolved_stride
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.pairs = discover_image_mask_pairs(self.config)
        self.samples: list[PatchIndexEntry] = build_patch_index(
            pairs=self.pairs,
            patch_size=self.patch_size,
            stride=self.stride,
        )

    def __len__(self) -> int:
        """Return the number of indexed patches."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Retrieve a single image and mask patch pair.

        Parameters
        ----------
        idx:
            Patch index in the deterministic patch list.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the image patch, mask patch, patch
            coordinates, and source image path.
        """
        sample = self.samples[idx]

        with Image.open(sample.image_path) as img:
            image_patch = img.crop(
                (
                    sample.x,
                    sample.y,
                    sample.x + self.patch_size,
                    sample.y + self.patch_size,
                )
            ).convert("L")

        with Image.open(sample.mask_path) as mask_image:
            mask_patch = mask_image.crop(
                (
                    sample.x,
                    sample.y,
                    sample.x + self.patch_size,
                    sample.y + self.patch_size,
                )
            ).convert("L")

        if self.image_transform is not None:
            image_patch = self.image_transform(image_patch)

        if self.mask_transform is not None:
            mask_patch = self.mask_transform(mask_patch)

        return {
            "image": image_patch,
            "mask": mask_patch,
            "coords": torch.tensor([sample.y, sample.x], dtype=torch.int32),
            "grid_coords": torch.tensor(
                [sample.grid_y, sample.grid_x], dtype=torch.int32
            ),
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }

    def get_patch_records(self) -> list[PatchIndexEntry]:
        """Return the indexed patch records."""
        return list(self.samples)

    def get_image_pairs(self) -> list[ProcessedImagePair]:
        """Return the validated image and mask pairs."""
        return list(self.pairs)
