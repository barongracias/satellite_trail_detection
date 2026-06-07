#!/usr/bin/env python
"""Shared locked-winner canvas reconstruction helpers for M6.1 analyses.

These helpers intentionally live under ``scripts/`` because they reuse the
existing locked-winner inference and Hough helpers. They are for
post-hoc locked-winner analysis only: no model selection, no threshold tuning.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from scripts.evaluation.hough_postprocess import (
    _HOUGH_MAX_BATCH,
    _chunks,
    _infer_batch,
    _load_normalised_patch,
    _parse_yx,
)
from src.classical.hough_runner import _apply_hough
from src.config.constants import PATCH_SIZE
from src.models.loading import load_segmentation_model

Image.MAX_IMAGE_PIXELS = None

LOCKED_CHECKPOINT = "results/checkpoints/model-best.pth"
LOCKED_THRESHOLD = 0.45
LOCKED_NORMALISATION = "full_image"
LOCKED_HOUGH_INPUT_THRESHOLD = 0.10
LOCKED_HOUGH_THRESHOLD = 50
LOCKED_MIN_LINE_LENGTH = 100
LOCKED_MAX_LINE_GAP = 250
LOCKED_LINE_THICKNESS = 3


@dataclass(frozen=True)
class LockedCanvases:
    """Per-source-image canvases for the locked winner."""

    source_image: str
    raw_image: np.ndarray
    target_canvas: np.ndarray
    support_canvas: np.ndarray
    prob_canvas: np.ndarray
    binary_canvas: np.ndarray
    hough_canvas: np.ndarray
    normalisation: str
    threshold: float
    hough_params: dict[str, int | float]

    @property
    def combined_canvas(self) -> np.ndarray:
        return np.logical_or(self.binary_canvas, self.hough_canvas)


def read_test_manifest(patch_dir: str | Path = "data/patches") -> pd.DataFrame:
    manifest_path = Path(patch_dir) / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    return manifest[manifest["split"] == "test"].reset_index(drop=True)


def source_to_mask_path(source_image: str | Path) -> Path:
    p = Path(source_image)
    name = p.name
    if name.endswith(".fits_full.png"):
        return p.with_name(name.replace(".fits_full.png", "_mask.png"))
    if name.endswith("_full.png"):
        return p.with_name(name.replace("_full.png", "_mask.png"))
    raise ValueError(f"Cannot derive mask path from source image: {source_image}")


def load_raw_image(source_image: str | Path) -> np.ndarray:
    with Image.open(source_image) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def load_raw_mask(source_image: str | Path) -> np.ndarray:
    mask_path = source_to_mask_path(source_image)
    with Image.open(mask_path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8) > 127


def build_support_canvas(
    records: pd.DataFrame,
    shape: tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """Return sampled-test patch support in raw image shape."""
    support = np.zeros(shape, dtype=bool)
    h, w = shape
    for patch_path in records["patch_path"]:
        y, x = _parse_yx(str(patch_path))
        y2 = min(y + patch_size, h)
        x2 = min(x + patch_size, w)
        if y < h and x < w:
            support[y:y2, x:x2] = True
    return support


def place_patch_max(
    canvas: np.ndarray,
    patch: np.ndarray,
    y: int,
    x: int,
) -> None:
    """Place a patch into a raw-shape canvas using max-composition."""
    h, w = canvas.shape
    ph, pw = patch.shape
    y2 = min(y + ph, h)
    x2 = min(x + pw, w)
    if y >= h or x >= w or y2 <= y or x2 <= x:
        return
    canvas[y:y2, x:x2] = np.maximum(canvas[y:y2, x:x2], patch[: y2 - y, : x2 - x])


def reconstruct_prob_canvas(
    records: pd.DataFrame,
    raw_shape: tuple[int, int],
    model: torch.nn.Module,
    device: torch.device,
    normalisation: str,
    max_batch: int = _HOUGH_MAX_BATCH,
) -> np.ndarray:
    prob_canvas = np.zeros(raw_shape, dtype=np.float32)
    has_full_image_stats = {"image_mean", "image_std"}.issubset(records.columns)
    rows = list(records.itertuples(index=False))
    for chunk in _chunks(rows, max_batch):
        patches = []
        coords = []
        for row in chunk:
            mean = getattr(row, "image_mean", None) if has_full_image_stats else None
            std = getattr(row, "image_std", None) if has_full_image_stats else None
            patches.append(_load_normalised_patch(row.patch_path, normalisation, mean, std))
            coords.append(_parse_yx(row.patch_path))
        probs = _infer_batch(patches, model, device)
        for (y, x), prob in zip(coords, probs):
            place_patch_max(prob_canvas, prob.astype(np.float32, copy=False), y, x)
    return prob_canvas


def load_locked_model(
    checkpoint: str | Path = LOCKED_CHECKPOINT,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, str, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalisation = load_segmentation_model(str(checkpoint), device)
    if normalisation != LOCKED_NORMALISATION:
        raise ValueError(
            f"Locked-winner analyses require normalisation={LOCKED_NORMALISATION!r}; "
            f"checkpoint returned {normalisation!r}"
        )
    return model, normalisation, device


def reconstruct_locked_canvases(
    source_image: str,
    records: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    normalisation: str,
    threshold: float = LOCKED_THRESHOLD,
    hough_input_threshold: float = LOCKED_HOUGH_INPUT_THRESHOLD,
    hough_threshold: int = LOCKED_HOUGH_THRESHOLD,
    min_line_length: int = LOCKED_MIN_LINE_LENGTH,
    max_line_gap: int = LOCKED_MAX_LINE_GAP,
    line_thickness: int = LOCKED_LINE_THICKNESS,
) -> LockedCanvases:
    if abs(threshold - LOCKED_THRESHOLD) > 1e-12:
        raise ValueError(f"Locked-winner analyses require threshold={LOCKED_THRESHOLD}; got {threshold}")
    raw_image = load_raw_image(source_image)
    target_canvas = load_raw_mask(source_image)
    support_canvas = build_support_canvas(records, raw_image.shape)
    prob_canvas = reconstruct_prob_canvas(records, raw_image.shape, model, device, normalisation)
    binary_canvas = prob_canvas >= threshold
    hough_input = (prob_canvas >= hough_input_threshold).astype(np.uint8) * 255
    hough_canvas = _apply_hough(
        hough_input,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
        line_thickness=line_thickness,
    ) > 0
    return LockedCanvases(
        source_image=str(source_image),
        raw_image=raw_image,
        target_canvas=target_canvas,
        support_canvas=support_canvas,
        prob_canvas=prob_canvas,
        binary_canvas=binary_canvas,
        hough_canvas=hough_canvas,
        normalisation=normalisation,
        threshold=threshold,
        hough_params={
            "hough_input_threshold": hough_input_threshold,
            "hough_threshold": hough_threshold,
            "min_line_length": min_line_length,
            "max_line_gap": max_line_gap,
            "line_thickness": line_thickness,
        },
    )


def iter_positive_source_groups(test_df: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    for source_image, group in test_df.groupby("source_image"):
        if bool((group["positive_pixel_fraction"] > 0).any()):
            yield str(source_image), group.reset_index(drop=True)


def manifest_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def provenance(
    *,
    checkpoint: str | Path = LOCKED_CHECKPOINT,
    threshold: float = LOCKED_THRESHOLD,
    normalisation: str = LOCKED_NORMALISATION,
    patch_dir: str | Path = "data/patches",
) -> dict[str, Any]:
    manifest_path = Path(patch_dir) / "manifest.csv"
    return {
        "checkpoint": str(checkpoint),
        "threshold": threshold,
        "normalisation": normalisation,
        "split": "sampled_test",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash(manifest_path) if manifest_path.exists() else None,
        "git_commit": git_commit(),
        "post_hoc_locked_winner_analysis": True,
        "display_intensity_caveat": (
            "Raw intensities are 8-bit display-scaled PNG values; contrast metrics are "
            "relative display-space proxies, not calibrated flux/SNR."
        ),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False))
