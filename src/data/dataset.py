"""Patch-based dataset utilities for satellite trail segmentation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import JointTransform, SignalDependentNoise, normalise_tensor


Image.MAX_IMAGE_PIXELS = None


class PatchDirectoryDataset(Dataset[dict[str, Any]]):
    """Load pre-built patches from a single split directory and a manifest CSV.

    Patches are expected to have been written by ``scripts/build_patch_dataset.py``.
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
    ) -> None:
        self.patch_dir = Path(patch_dir)
        self.normalisation = normalisation
        split = self.patch_dir.name
        if manifest_path is None:
            manifest_path = self.patch_dir.parent / "manifest.csv"
        manifest = pd.read_csv(manifest_path)
        self.records = manifest[manifest["split"] == split].reset_index(drop=True)
        # Pre-materialise rows as plain dicts to avoid pandas Series construction
        # in __getitem__, which dominates the DataLoader CPU path otherwise.
        self._rows = self.records.to_dict("records")
        self._transform = JointTransform(augment=augment_train and split == "train")
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
        if self._noise is not None:
            image = self._noise(image)
        image = normalise_tensor(
            image,
            self.normalisation,
            full_image_mean=row.get("image_mean"),
            full_image_std=row.get("image_std"),
        )

        return {"image": image, "mask": mask}
