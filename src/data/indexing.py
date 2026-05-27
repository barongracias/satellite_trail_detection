"""Torch-free helpers for validating processed image pairs and patch indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from PIL import Image

from src.config.constants import PATCH_SIZE


Image.MAX_IMAGE_PIXELS = None


_OBSERVATION_FILENAME_PATTERN = re.compile(r"ML1_(\d{8})_(\d{6})")


def parse_observation_datetime(image_path: str | Path) -> datetime | None:
    """Parse the YYYYMMDD HHMMSS observation timestamp from a MeerLICHT filename.

    The MeerLICHT filename convention is `ML1_YYYYMMDD_HHMMSS_red.fits_full.png`.
    Returns ``None`` when the filename does not match the expected pattern, so
    callers can decide whether unknown timestamps are an error.
    """
    match = _OBSERVATION_FILENAME_PATTERN.search(Path(image_path).name)
    if match is None:
        return None
    date_str, time_str = match.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")


@dataclass(frozen=True)
class PatchDatasetConfig:
    """Configuration for deterministic patch extraction from processed images."""

    root_dir: Path
    patch_size: int = PATCH_SIZE
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


