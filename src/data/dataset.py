"""Patch-based dataset utilities for satellite trail segmentation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import pandas as pd
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
from src.data.transforms import JointTransform, normalise_tensor


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
        """Return image patch, mask patch, and pixel coordinates at index idx."""
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
        }

    def get_patch_records(self) -> list[PatchIndexEntry]:
        """Return the indexed patch records."""
        return list(self.samples)

    def get_image_pairs(self) -> list[ProcessedImagePair]:
        """Return the validated image and mask pairs."""
        return list(self.pairs)


class PatchDirectoryDataset(Dataset[dict[str, Any]]):
    """Load pre-built patches from a single split directory and a manifest CSV.

    Patches are expected to have been written by ``scripts/build_patch_dataset.py``.
    Image patches are normalised at load time; mask patches are converted to float
    tensors.  The returned dict matches the format of
    :class:`SatelliteTrailPatchDataset`.
    """

    def __init__(
        self,
        patch_dir: str | Path,
        manifest_path: str | Path | None = None,
        normalisation: str = "fixed",
        augment_train: bool = True,
    ) -> None:
        self.patch_dir = Path(patch_dir)
        self.normalisation = normalisation
        split = self.patch_dir.name
        if manifest_path is None:
            manifest_path = self.patch_dir.parent / "manifest.csv"
        manifest = pd.read_csv(manifest_path)
        self.records = manifest[manifest["split"] == split].reset_index(drop=True)
        self._transform = JointTransform(augment=augment_train and split == "train")

        if normalisation == "full_image":
            has_stats = {"image_mean", "image_std"}.issubset(self.records.columns)
            n_missing = (
                int(self.records[["image_mean", "image_std"]].isna().any(axis=1).sum())
                if has_stats else len(self.records)
            )
            if n_missing:
                warnings.warn(
                    f"PatchDirectoryDataset({split}): normalisation=full_image but "
                    f"{n_missing}/{len(self.records)} rows lack image_mean/image_std; "
                    "falling back to per-patch z-score for those rows.",
                    stacklevel=2,
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records.iloc[idx]
        with Image.open(row["patch_path"]) as img:
            img_pil = img.convert("L")
        with Image.open(row["mask_path"]) as msk:
            mask_pil = msk.convert("L")

        image, mask = self._transform(img_pil, mask_pil)
        image = normalise_tensor(
            image,
            self.normalisation,
            full_image_mean=row.get("image_mean"),
            full_image_std=row.get("image_std"),
        )

        return {
            "image": image,
            "mask": mask,
            "coords": torch.tensor([0, 0], dtype=torch.int32),
        }
