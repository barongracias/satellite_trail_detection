#!/usr/bin/env python
"""Probability-stratified Hough post-processing (soft-probability approximation).

`cv2.HoughLinesP` votes on a BINARY canvas, so it cannot weight pixels by the
U-Net probability directly. This script approximates probability weighting by
running the Hough transform at several probability strata and unioning the
detected line canvases: faint-but-real pixels still vote at low strata, while
high-confidence pixels vote at strict strata without background noise. This is
an APPROXIMATION of weighted/Radon voting, not a true weighted accumulator —
named accordingly in the output metadata.

It compares, on the same reconstructed test canvases and at the same operating
threshold, three regimes for pixel recall and patch-level FNR:
  - pre-Hough         : thresholded U-Net only
  - binary Hough      : single low-threshold stratum (the locked behaviour)
  - stratified Hough  : union over strata (this approximation)

The locked Hough path (src/classical/hough_runner.py, scripts/evaluation/hough_postprocess.py)
is NOT modified; this script reuses the locked canvas-reconstruction helpers and
the shared Hough-drawing helper, then writes a separate JSON artifact.

Usage (CSD3):
    CHECKPOINT=results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth \\
    THRESHOLD=0.45 \\
    TAG=t44_s2804 \\
      sbatch slurm/hough_prob_stratified.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # repo root (src.*, scripts.*)

import numpy as np
import pandas as pd
import torch
from PIL import Image

from scripts.evaluation.hough_postprocess import (
    _HOUGH_MAX_BATCH,
    _PATCH_SIZE,
    _chunks,
    _infer_batch,
    _load_normalised_patch,
    _parse_yx,
    load_model,
)
from src.classical.hough_runner import _apply_hough
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

# Preregistered M5.6 Hough parameters (match hough_postprocess.py defaults).
_HOUGH_KWARGS = dict(hough_threshold=50, min_line_length=100, max_line_gap=250, line_thickness=3)
_DEFAULT_STRATA = (0.1, 0.3, 0.5, 0.7, 0.9)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--threshold", type=float, required=True,
                   help="U-Net operating threshold (validation-optimal for the winner).")
    p.add_argument("--strata", default=",".join(str(s) for s in _DEFAULT_STRATA),
                   help="Comma-separated probability strata for the stratified union.")
    p.add_argument("--binary_input_threshold", type=float, default=0.1,
                   help="Single-stratum input threshold for the binary Hough baseline.")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _hough_union(prob_canvas: np.ndarray, strata: list[float]) -> np.ndarray:
    """Union of Hough line canvases across probability strata."""
    union = np.zeros(prob_canvas.shape, dtype=np.uint8)
    for s in strata:
        binary = (prob_canvas >= s).astype(np.uint8) * 255
        union = np.maximum(union, _apply_hough(binary, **_HOUGH_KWARGS))
    return union


def _resolve_out_path(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    if args.tag:
        return Path("results/classical") / f"hough_prob_stratified_{args.tag}.json"
    stem = Path(args.checkpoint).stem
    tag = stem[:-5] if stem.endswith("_best") else stem
    return Path("results/classical") / f"hough_prob_stratified_{tag}.json"


def main() -> None:
    args = parse_args()
    strata = [float(s) for s in args.strata.split(",") if s.strip() != ""]
    logger = get_logger("hough_prob_stratified")
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalisation = load_model(args.checkpoint, device)
    logger.info("Loaded %s (normalisation=%s, threshold=%.3f, strata=%s)",
                args.checkpoint, normalisation, args.threshold, strata)

    manifest = pd.read_csv(Path(args.patch_dir) / "manifest.csv")
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    has_stats = {"image_mean", "image_std"}.issubset(test_df.columns)

    # Aggregates over positive test images.
    n_pos_images = 0
    gt_total = 0
    cov_pre = cov_binary = cov_strat = 0
    pos_patches = fn_pre = fn_binary = fn_strat = 0
    hough_fp_binary = hough_fp_strat = 0
    rows = []

    for source_image, group in test_df.groupby("source_image"):
        if not bool((group["positive_pixel_fraction"] > 0).any()):
            continue

        yx = [_parse_yx(p) for p in group["patch_path"]]
        h = max(y for y, _ in yx) + _PATCH_SIZE
        w = max(x for _, x in yx) + _PATCH_SIZE
        prob_canvas = np.zeros((h, w), dtype=np.float32)
        target_canvas = np.zeros((h, w), dtype=np.uint8)

        group_rows = list(group.itertuples(index=False))
        for chunk in _chunks(group_rows, _HOUGH_MAX_BATCH):
            patches, coords = [], []
            for row in chunk:
                mean = getattr(row, "image_mean", None) if has_stats else None
                std = getattr(row, "image_std", None) if has_stats else None
                patches.append(_load_normalised_patch(row.patch_path, normalisation, mean, std))
                coords.append(_parse_yx(row.patch_path))
            probs = _infer_batch(patches, model, device)
            for (y, x), prob, row in zip(coords, probs, chunk):
                sl_y, sl_x = slice(y, y + _PATCH_SIZE), slice(x, x + _PATCH_SIZE)
                prob_canvas[sl_y, sl_x] = np.maximum(prob_canvas[sl_y, sl_x], prob)
                with Image.open(row.mask_path) as msk:
                    mask_patch = np.asarray(msk.convert("L"), dtype=np.uint8)
                target_canvas[sl_y, sl_x] = np.maximum(target_canvas[sl_y, sl_x], mask_patch)

        target_bool = target_canvas > 0
        binary_bool = prob_canvas >= args.threshold
        hough_binary = _hough_union(prob_canvas, [args.binary_input_threshold]) > 0
        hough_strat = _hough_union(prob_canvas, strata) > 0
        combined_binary = binary_bool | hough_binary
        combined_strat = binary_bool | hough_strat

        gt_pixels = int(target_bool.sum())
        c_pre = int((binary_bool & target_bool).sum())
        c_bin = int((combined_binary & target_bool).sum())
        c_str = int((combined_strat & target_bool).sum())
        fp_bin = int((hough_binary & ~target_bool).sum())
        fp_str = int((hough_strat & ~target_bool).sum())

        n_pp = n_fn_pre = n_fn_bin = n_fn_str = 0
        for row in group_rows:
            if row.positive_pixel_fraction <= 0:
                continue
            y, x = _parse_yx(row.patch_path)
            sl_y, sl_x = slice(y, y + _PATCH_SIZE), slice(x, x + _PATCH_SIZE)
            tgt = target_bool[sl_y, sl_x]
            if not tgt.any():
                continue
            n_pp += 1
            n_fn_pre += int(not (binary_bool[sl_y, sl_x] & tgt).any())
            n_fn_bin += int(not (combined_binary[sl_y, sl_x] & tgt).any())
            n_fn_str += int(not (combined_strat[sl_y, sl_x] & tgt).any())

        n_pos_images += 1
        gt_total += gt_pixels
        cov_pre += c_pre; cov_binary += c_bin; cov_strat += c_str
        pos_patches += n_pp; fn_pre += n_fn_pre; fn_binary += n_fn_bin; fn_strat += n_fn_str
        hough_fp_binary += fp_bin; hough_fp_strat += fp_str
        rows.append({
            "source_image": str(source_image), "gt_pixels": gt_pixels,
            "pixel_recall_pre": round(c_pre / gt_pixels, 6) if gt_pixels else None,
            "pixel_recall_binary": round(c_bin / gt_pixels, 6) if gt_pixels else None,
            "pixel_recall_stratified": round(c_str / gt_pixels, 6) if gt_pixels else None,
        })
        logger.info("%s | pixel recall pre/bin/strat=%.4f/%.4f/%.4f",
                    Path(source_image).name,
                    c_pre / gt_pixels if gt_pixels else 0.0,
                    c_bin / gt_pixels if gt_pixels else 0.0,
                    c_str / gt_pixels if gt_pixels else 0.0)

    rec = lambda c: round(c / gt_total, 6) if gt_total else None
    fnr = lambda c: round(c / pos_patches, 6) if pos_patches else None
    out = {
        "method": "probability_stratified_hough_approximation",
        "method_note": "union of Hough over probability strata; NOT a true weighted "
                       "accumulator (cv2.HoughLinesP votes on a binary canvas)",
        "checkpoint": str(args.checkpoint),
        "split": "test",
        "normalisation": normalisation,
        "threshold": args.threshold,
        "strata": strata,
        "binary_input_threshold": args.binary_input_threshold,
        "hough_kwargs": _HOUGH_KWARGS,
        "n_positive_images": n_pos_images,
        "n_positive_patches": pos_patches,
        "total_gt_pixels": gt_total,
        "pixel_recall_pre": rec(cov_pre),
        "pixel_recall_binary_hough": rec(cov_binary),
        "pixel_recall_stratified_hough": rec(cov_strat),
        "patch_fnr_pre": fnr(fn_pre),
        "patch_fnr_binary_hough": fnr(fn_binary),
        "patch_fnr_stratified_hough": fnr(fn_strat),
        "hough_fp_pixels_binary": hough_fp_binary,
        "hough_fp_pixels_stratified": hough_fp_strat,
        "per_image": rows,
        "generated": str(date.today()),
    }
    out_path = _resolve_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False))
    logger.info("pixel recall pre/binary/stratified: %s / %s / %s",
                out["pixel_recall_pre"], out["pixel_recall_binary_hough"],
                out["pixel_recall_stratified_hough"])
    logger.info("Saved to %s", out_path)
    print(f"Probability-stratified Hough -> {out_path}")
    print(f"  pixel recall pre/binary/stratified: {out['pixel_recall_pre']} / "
          f"{out['pixel_recall_binary_hough']} / {out['pixel_recall_stratified_hough']}")
    print(f"  Hough FP pixels binary/stratified: {hough_fp_binary} / {hough_fp_strat}")


if __name__ == "__main__":
    main()
