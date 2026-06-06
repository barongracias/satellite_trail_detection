"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def write_pair(root_dir: Path, stem: str, image_array: np.ndarray, mask_array: np.ndarray) -> None:
    """Write a paired image/mask PNG using the MeerLICHT ``*_red`` naming convention."""
    Image.fromarray(image_array).save(root_dir / f"{stem}_red.fits_full.png")
    Image.fromarray(mask_array).save(root_dir / f"{stem}_red_mask.png")
