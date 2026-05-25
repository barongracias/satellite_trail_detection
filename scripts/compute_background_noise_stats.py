#!/usr/bin/env python
"""Calibrate signal-dependent noise from empty train patches."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/patches/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/classical/background_noise_calibration.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    required = {"split", "patch_path", "positive_pixel_fraction"}
    missing = required.difference(manifest.columns)
    if missing:
        raise SystemExit(f"manifest missing required columns: {sorted(missing)}")

    rows = manifest[
        (manifest["split"] == "train")
        & (manifest["positive_pixel_fraction"].astype(float) == 0.0)
    ]
    if rows.empty:
        raise SystemExit("no empty train patches found for noise calibration")

    pixel_count = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for patch_path in rows["patch_path"]:
        with Image.open(patch_path) as image:
            arr = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        pixel_count += int(arr.size)
        sum_x += float(arr.sum())
        sum_x2 += float(np.square(arr).sum())

    if pixel_count == 0:
        raise SystemExit("empty train patches contained zero pixels")

    mu_bg = sum_x / pixel_count
    variance = max(0.0, (sum_x2 / pixel_count) - mu_bg ** 2)
    sigma_bg = math.sqrt(variance)
    if mu_bg <= 0.0:
        raise SystemExit(f"background mean must be positive, got {mu_bg}")

    alpha = sigma_bg ** 2 / (2.0 * mu_bg)
    beta = sigma_bg / math.sqrt(2.0)
    payload = {
        "manifest": str(args.manifest),
        "split": "train",
        "selection": "positive_pixel_fraction == 0",
        "empty_patch_count": int(len(rows)),
        "pixel_count": int(pixel_count),
        "mu_bg": mu_bg,
        "sigma_bg": sigma_bg,
        "alpha": alpha,
        "beta": beta,
        "formula": "noise_std(p) = multiplier * sqrt(alpha * p + beta**2)",
        "verification": "at p = mu_bg, noise_std = multiplier * sigma_bg",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
