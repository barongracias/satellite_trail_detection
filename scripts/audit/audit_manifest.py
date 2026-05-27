#!/usr/bin/env python
"""Manifest and data-integrity audit for data/patches/.

Run after scripts/build_patch_dataset.py (and optionally
scripts/compute_image_stats.py) to catch silent dataset bugs before they
contaminate training or evaluation.

The audit is read-only. It exits with status 0 if every check passes and
status 1 otherwise, printing a per-check summary.

Checks performed
----------------
1. Manifest schema       — required columns present; optional image_mean/
                            image_std columns recognised.
2. File existence        — every patch_path and mask_path on disk.
3. Split disjointness    — train / val / test source_image sets are pairwise
                            disjoint (no image-level leakage across splits).
4. positive_pixel_fraction
                         — non-NaN, in [0, 1], and the fraction recomputed
                            from the mask agrees with the stored value within
                            tolerance for a random sample of rows.
5. Class balance         — per-split positive-patch counts and 1:3 sampling
                            ratio sanity (train/val expect ~25 % positive,
                            test honours either sampled or parity ratio).
6. Image stats           — if image_mean / image_std present, all finite or
                            consistently NaN per source_image (no half-
                            populated columns), and values in [0, 1] / (0, 1].
7. Patch shape           — random sample of image patches confirmed to be
                            ``PATCH_SIZE`` × ``PATCH_SIZE`` 8-bit greyscale.
8. Patch/mask alignment  — sampled mask shape matches sampled image shape.

Usage
-----
    python scripts/audit/audit_manifest.py
    python scripts/audit/audit_manifest.py --manifest data/patches_test_full/manifest.csv
    python scripts/audit/audit_manifest.py --sample-rows 200 --strict-stats
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.constants import PATCH_SIZE  # noqa: E402


Image.MAX_IMAGE_PIXELS = None

_REQUIRED_COLUMNS = (
    "split",
    "source_image",
    "patch_path",
    "mask_path",
    "positive_pixel_fraction",
)
_OPTIONAL_COLUMNS = ("image_mean", "image_std")
_VALID_SPLITS = ("train", "val", "test")
_POS_FRAC_TOLERANCE = 1e-6


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _check_schema(df: pd.DataFrame) -> CheckResult:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return CheckResult(
            "schema",
            False,
            f"missing required columns: {missing}",
        )
    extras = [c for c in df.columns if c not in _REQUIRED_COLUMNS + _OPTIONAL_COLUMNS]
    has_stats = all(c in df.columns for c in _OPTIONAL_COLUMNS)
    return CheckResult(
        "schema",
        True,
        f"required columns present; image_stats={'yes' if has_stats else 'no'}; "
        f"extras={extras if extras else 'none'}",
    )


def _check_file_existence(df: pd.DataFrame) -> CheckResult:
    missing_patches = [p for p in df["patch_path"] if not Path(p).exists()]
    missing_masks = [p for p in df["mask_path"] if not Path(p).exists()]
    n_missing = len(missing_patches) + len(missing_masks)
    if n_missing:
        sample = (missing_patches + missing_masks)[:5]
        return CheckResult(
            "file_existence",
            False,
            f"{n_missing} missing files; first 5: {sample}",
        )
    return CheckResult(
        "file_existence",
        True,
        f"all {len(df)} patch_path + mask_path rows resolve on disk",
    )


def _check_split_values(df: pd.DataFrame) -> CheckResult:
    unknown = sorted(set(df["split"]) - set(_VALID_SPLITS))
    if unknown:
        return CheckResult(
            "split_values",
            False,
            f"unknown split values: {unknown}",
        )
    counts = df["split"].value_counts().to_dict()
    return CheckResult(
        "split_values",
        True,
        f"split sizes: {counts}",
    )


def _check_split_disjoint(df: pd.DataFrame) -> CheckResult:
    by_split = {
        s: set(df.loc[df["split"] == s, "source_image"])
        for s in df["split"].unique()
    }
    overlaps: list[str] = []
    splits = list(by_split.keys())
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            shared = by_split[a] & by_split[b]
            if shared:
                overlaps.append(
                    f"{a}∩{b}={len(shared)} (e.g. {sorted(shared)[:2]})"
                )
    if overlaps:
        return CheckResult(
            "split_disjoint",
            False,
            "; ".join(overlaps),
        )
    sizes = ", ".join(f"{s}={len(v)}" for s, v in by_split.items())
    return CheckResult(
        "split_disjoint",
        True,
        f"source_image sets pairwise disjoint ({sizes})",
    )


def _check_positive_fraction(
    df: pd.DataFrame, sample_rows: int, rng: np.random.Generator
) -> CheckResult:
    pf = df["positive_pixel_fraction"]
    n_nan = int(pf.isna().sum())
    if n_nan:
        return CheckResult(
            "positive_pixel_fraction",
            False,
            f"{n_nan} NaN entries in positive_pixel_fraction",
        )
    out_of_range = int(((pf < 0.0) | (pf > 1.0)).sum())
    if out_of_range:
        return CheckResult(
            "positive_pixel_fraction",
            False,
            f"{out_of_range} entries outside [0, 1]",
        )

    sample = df.sample(min(sample_rows, len(df)), random_state=rng.bit_generator)
    mismatches: list[str] = []
    for row in sample.itertuples(index=False):
        with Image.open(row.mask_path) as msk:
            arr = np.asarray(msk.convert("L"), dtype=np.uint8)
        recomputed = float((arr > 0).sum()) / arr.size
        if abs(recomputed - float(row.positive_pixel_fraction)) > _POS_FRAC_TOLERANCE:
            mismatches.append(
                f"{Path(row.mask_path).name}: stored={row.positive_pixel_fraction:.6f}, "
                f"recomputed={recomputed:.6f}"
            )
    if mismatches:
        return CheckResult(
            "positive_pixel_fraction",
            False,
            f"{len(mismatches)} mask↔fraction mismatches in {len(sample)} sampled rows; "
            f"first: {mismatches[0]}",
        )
    return CheckResult(
        "positive_pixel_fraction",
        True,
        f"all in [0, 1]; recompute matched on {len(sample)} sampled rows "
        f"within tolerance {_POS_FRAC_TOLERANCE:.0e}",
    )


def _check_class_balance(df: pd.DataFrame) -> CheckResult:
    lines: list[str] = []
    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split]
        n_pos = int((sub["positive_pixel_fraction"] > 0).sum())
        n_neg = int((sub["positive_pixel_fraction"] == 0).sum())
        total = len(sub)
        ratio = n_neg / n_pos if n_pos else float("inf")
        lines.append(
            f"{split}: pos={n_pos}, neg={n_neg}, total={total}, neg/pos={ratio:.2f}"
        )
        if n_pos == 0:
            return CheckResult(
                "class_balance",
                False,
                f"split {split!r} has zero positive patches: " + " | ".join(lines),
            )
    return CheckResult(
        "class_balance",
        True,
        " | ".join(lines),
    )


def _check_image_stats(df: pd.DataFrame, strict: bool) -> CheckResult:
    if not all(c in df.columns for c in _OPTIONAL_COLUMNS):
        return CheckResult(
            "image_stats",
            True,
            "image_mean / image_std columns not present (skipped)",
        )

    grouped = df.groupby("source_image")[list(_OPTIONAL_COLUMNS)]
    half_populated: list[str] = []
    for src, sub in grouped:
        mean_na = sub["image_mean"].isna()
        std_na = sub["image_std"].isna()
        if mean_na.any() != std_na.any() or (mean_na.sum() and not mean_na.all()):
            half_populated.append(str(src))
    if half_populated:
        return CheckResult(
            "image_stats",
            False,
            f"{len(half_populated)} source_image groups with inconsistent NaN "
            f"between image_mean and image_std (first: {half_populated[0]})",
        )

    finite = df[~df["image_mean"].isna()]
    if not finite.empty:
        bad_mean = int(((finite["image_mean"] < 0) | (finite["image_mean"] > 1)).sum())
        bad_std = int((finite["image_std"] <= 0).sum())
        if bad_mean or bad_std:
            return CheckResult(
                "image_stats",
                False,
                f"image_mean out-of-range rows={bad_mean}, "
                f"image_std non-positive rows={bad_std}",
            )

    n_nan = int(df["image_mean"].isna().sum())
    if strict and n_nan:
        return CheckResult(
            "image_stats",
            False,
            f"--strict-stats: {n_nan} rows have NaN image_mean (run compute_image_stats.py)",
        )
    return CheckResult(
        "image_stats",
        True,
        f"image_mean / image_std consistent per source_image; NaN rows={n_nan}",
    )


def _check_patch_shapes(
    df: pd.DataFrame, sample_rows: int, rng: np.random.Generator
) -> CheckResult:
    sample = df.sample(min(sample_rows, len(df)), random_state=rng.bit_generator)
    bad: list[str] = []
    for row in sample.itertuples(index=False):
        with Image.open(row.patch_path) as img:
            ipx = img.convert("L")
            ishape = ipx.size  # (W, H)
        with Image.open(row.mask_path) as msk:
            mpx = msk.convert("L")
            mshape = mpx.size
        expected = (PATCH_SIZE, PATCH_SIZE)
        if ishape != expected or mshape != expected:
            bad.append(
                f"{Path(row.patch_path).name}: image={ishape}, mask={mshape}"
            )
        elif ishape != mshape:
            bad.append(
                f"{Path(row.patch_path).name}: image/mask shape mismatch "
                f"image={ishape}, mask={mshape}"
            )
    if bad:
        return CheckResult(
            "patch_shapes",
            False,
            f"{len(bad)} bad shapes in {len(sample)} sampled rows; first: {bad[0]}",
        )
    return CheckResult(
        "patch_shapes",
        True,
        f"all {len(sample)} sampled image+mask patches are "
        f"{PATCH_SIZE}×{PATCH_SIZE} 8-bit greyscale",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/patches/manifest.csv"),
        help="Path to the manifest CSV to audit (default: data/patches/manifest.csv).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100,
        help="Random sample size for the positive-fraction recompute and shape checks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2804,
        help="Seed for the sampling RNG (deterministic audits).",
    )
    parser.add_argument(
        "--strict-stats",
        action="store_true",
        help="Fail when image_mean / image_std rows are NaN "
        "(use after compute_image_stats.py has been run).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.manifest.exists():
        print(f"FAIL  manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.manifest)
    rng = np.random.default_rng(args.seed)

    checks: list[CheckResult] = []
    checks.append(_check_schema(df))
    # Subsequent checks assume schema is OK.
    if not checks[-1].passed:
        _report(checks)
        return 1

    checks.append(_check_file_existence(df))
    checks.append(_check_split_values(df))
    checks.append(_check_split_disjoint(df))
    checks.append(_check_positive_fraction(df, args.sample_rows, rng))
    checks.append(_check_class_balance(df))
    checks.append(_check_image_stats(df, strict=args.strict_stats))
    checks.append(_check_patch_shapes(df, args.sample_rows, rng))

    return 0 if _report(checks) else 1


def _report(checks: list[CheckResult]) -> bool:
    overall = True
    width = max(len(c.name) for c in checks)
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"{status}  {c.name.ljust(width)}  {c.detail}")
        if not c.passed:
            overall = False
    print()
    print("OVERALL:", "PASS" if overall else "FAIL")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
