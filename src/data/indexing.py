"""Torch-free helpers for validating processed image pairs and patch indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class PatchDatasetConfig:
    """Configuration for deterministic patch extraction from processed images."""

    root_dir: Path
    patch_size: int = 512
    stride: int | None = None
    strict_pairing: bool = True

    def __post_init__(self) -> None:
        """Validate the dataset configuration and normalise the root path."""
        object.__setattr__(self, "root_dir", Path(self.root_dir))
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"Dataset root is not a directory: {self.root_dir}"
            )
        if self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if self.resolved_stride <= 0:
            raise ValueError("stride must be a positive integer")

    @property
    def resolved_stride(self) -> int:
        """Return the stride used for patch indexing."""
        return self.patch_size if self.stride is None else self.stride


@dataclass(frozen=True)
class ProcessedImagePair:
    """Describe a processed image, its matching mask, and their shared size."""

    image_path: Path
    mask_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class PatchIndexEntry:
    """Describe a single patch location within a paired image and mask."""

    image_path: Path
    mask_path: Path
    y: int
    x: int
    grid_y: int
    grid_x: int


def discover_image_mask_pairs(
    config: PatchDatasetConfig,
) -> list[ProcessedImagePair]:
    """Return valid processed image and mask pairs from a directory."""
    images = sorted(config.root_dir.glob("*_red.fits_full.png"))
    if not images:
        raise RuntimeError(f"No processed images found in {config.root_dir}")

    pairs: list[ProcessedImagePair] = []
    missing_masks: list[str] = []

    for image_path in images:
        mask_path = image_path.with_name(
            image_path.name.replace("_red.fits_full.png", "_red_mask.png")
        )

        if not mask_path.exists():
            if config.strict_pairing:
                missing_masks.append(mask_path.name)
            continue

        with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
            image_size = image.size
            mask_size = mask_image.size

        if image_size != mask_size:
            raise ValueError(
                f"Image and mask sizes do not match for {image_path.name}"
            )

        width, height = image_size
        pairs.append(
            ProcessedImagePair(
                image_path=image_path,
                mask_path=mask_path,
                width=width,
                height=height,
            )
        )

    if missing_masks:
        raise FileNotFoundError(f"Missing masks: {', '.join(missing_masks)}")

    if not pairs:
        raise RuntimeError(f"No valid image and mask pairs found in {config.root_dir}")

    return pairs


def build_patch_index(
    pairs: Iterable[ProcessedImagePair],
    patch_size: int,
    stride: int,
) -> list[PatchIndexEntry]:
    """Build a deterministic list of patch coordinates across image pairs."""
    samples: list[PatchIndexEntry] = []

    for pair in pairs:
        if pair.width < patch_size or pair.height < patch_size:
            raise ValueError(
                f"Image is smaller than patch_size: {pair.image_path.name}"
            )

        for grid_y, y in enumerate(range(0, pair.height - patch_size + 1, stride)):
            for grid_x, x in enumerate(range(0, pair.width - patch_size + 1, stride)):
                samples.append(
                    PatchIndexEntry(
                        image_path=pair.image_path,
                        mask_path=pair.mask_path,
                        y=y,
                        x=x,
                        grid_y=grid_y,
                        grid_x=grid_x,
                    )
                )

    if not samples:
        raise RuntimeError("No patches generated from the processed image pairs")

    return samples
