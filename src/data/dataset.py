"""Patch-based dataset utilities for satellite trail segmentation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.config.constants import GLOBAL_SEED
from src.data.transforms import JointTransform, SignalDependentNoise, normalise_tensor


Image.MAX_IMAGE_PIXELS = None

TARGET_MODES = ("hard", "dilated_soft")


def dilated_soft_target(
    mask: torch.Tensor, dilation_px: int, band_value: float
) -> torch.Tensor:
    """Turn a hard {0,1} mask into a soft target with a dilated boundary band.

    The trail core keeps value 1.0; pixels within ``dilation_px`` (Chebyshev) of
    the core are set to ``band_value`` (e.g. 0.5); background stays 0.0. Encodes
    boundary uncertainty so the loss is not penalised for near-miss edge pixels.
    Operates on a ``(1, H, W)`` mask tensor. Train-only by construction.
    """
    hard = (mask > 0.5).float()
    k = 2 * dilation_px + 1
    dilated = F.max_pool2d(
        hard.unsqueeze(0), kernel_size=k, stride=1, padding=dilation_px
    ).squeeze(0)
    return torch.maximum(hard, dilated * band_value)


class PatchDirectoryDataset(Dataset[dict[str, Any]]):
    """Load pre-built patches from a single split directory and a manifest CSV.

    Patches are expected to have been written by ``scripts/data/build_patch_dataset.py``.
    Image patches are normalised at load time; mask patches are converted to float
    tensors. ``__getitem__`` returns ``{"image": ..., "mask": ...}``.
    """

    def __init__(
        self,
        patch_dir: str | Path,
        manifest_path: str | Path | None = None,
        normalisation: str = "fixed",
        augment_train: bool = True,
        noise_augment: bool = False,
        noise_std_multiplier: float = 1.0,
        noise_calibration_path: str | Path = "results/classical/background_noise_calibration.json",
        train_image_fraction: float = 1.0,
        target_mode: str = "hard",
        soft_dilation_px: int = 1,
        soft_band_value: float = 0.5,
    ) -> None:
        self.patch_dir = Path(patch_dir)
        self.normalisation = normalisation
        split = self.patch_dir.name
        if target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}")
        if manifest_path is None:
            manifest_path = self.patch_dir.parent / "manifest.csv"
        manifest = pd.read_csv(manifest_path)
        self.records = manifest[manifest["split"] == split].reset_index(drop=True)
        # Data-efficiency study: subset the TRAIN split to a fraction of source
        # images (image-level, preserving the no-leakage discipline). A fixed
        # GLOBAL_SEED permutation makes the subsets nested across fractions
        # (30% ⊂ 50% ⊂ 70%) and independent of the training seed.
        if split == "train" and train_image_fraction < 1.0:
            ordered = (
                self.records["source_image"]
                .drop_duplicates()
                .sample(frac=1.0, random_state=GLOBAL_SEED)
                .tolist()
            )
            n_keep = max(1, round(train_image_fraction * len(ordered)))
            keep = set(ordered[:n_keep])
            self.records = self.records[
                self.records["source_image"].isin(keep)
            ].reset_index(drop=True)
        # Pre-materialise rows as plain dicts to avoid pandas Series construction
        # in __getitem__, which dominates the DataLoader CPU path otherwise.
        self._rows = self.records.to_dict("records")
        self._transform = JointTransform(augment=augment_train and split == "train")
        # Soft/dilated targets apply to TRAIN ONLY — val/test masks stay hard so
        # evaluation is never self-serving (mirrors the noise-augment discipline).
        self._apply_soft_target = target_mode == "dilated_soft" and split == "train"
        self._soft_dilation_px = soft_dilation_px
        self._soft_band_value = soft_band_value
        self._noise: SignalDependentNoise | None = None
        if noise_augment and split == "train":
            calibration_path = Path(noise_calibration_path)
            if not calibration_path.exists():
                raise FileNotFoundError(
                    f"noise calibration file not found: {calibration_path}"
                )
            calibration = json.loads(calibration_path.read_text())
            self._noise = SignalDependentNoise(
                alpha=float(calibration["alpha"]),
                beta=float(calibration["beta"]),
                multiplier=noise_std_multiplier,
            )

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
        row = self._rows[idx]
        with Image.open(row["patch_path"]) as img:
            img_pil = img.convert("L")
        with Image.open(row["mask_path"]) as msk:
            mask_pil = msk.convert("L")

        image, mask = self._transform(img_pil, mask_pil)
        if self._apply_soft_target:
            mask = dilated_soft_target(mask, self._soft_dilation_px, self._soft_band_value)
        if self._noise is not None:
            image = self._noise(image)
        image = normalise_tensor(
            image,
            self.normalisation,
            full_image_mean=row.get("image_mean"),
            full_image_std=row.get("image_std"),
        )

        return {"image": image, "mask": mask}
