#!/usr/bin/env python
"""Geometry-aware test evaluation for a locked U-Net checkpoint.

One inference pass over the test split that reports, alongside the exact
pixel metrics, two thin-structure-aware diagnostics:

  - centerline Dice (clDice): topology overlap of the skeletonised masks,
    averaged over target-positive patches.
  - false-positive distance-to-mask: for every FP pixel, the Euclidean
    distance to the nearest ground-truth positive, aggregated into a
    histogram. FP pixels in patches with no trail (distance undefined) are
    reported separately as whole-patch FP.

These ground the "are the false positives boundary-adjacent or genuine
background hallucinations?" question without any new training. Validation-only
selection must already be locked; this re-scores the locked checkpoint's test
predictions at its validation-optimal threshold.

Usage (CSD3):
    CHECKPOINT=results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth \\
    THRESHOLD=0.45 \\
    TAG=t44_s2804 \\
      sbatch slurm/geometry_eval.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PatchDirectoryDataset
from src.evaluation.segmentation import (
    SegmentationCounts,
    centerline_dice,
    combine_counts,
    compute_metrics_from_counts,
    compute_segmentation_counts,
    false_positive_distances,
)
from src.models.loading import load_segmentation_model
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

# FP distance-to-mask histogram: 1px bins over [0, 50), with an explicit
# overflow bucket for the rare far-background FP beyond 50px.
_HIST_EDGES = np.arange(0.0, 51.0, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--tag", default=None, help="Short tag for the output filename.")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def _resolve_out_path(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    if args.tag:
        return Path("results/classical") / f"geometry_eval_{args.tag}.json"
    stem = Path(args.checkpoint).stem
    tag = stem[:-5] if stem.endswith("_best") else stem
    return Path("results/classical") / f"geometry_eval_{tag}.json"


def _approx_median(counts: np.ndarray, edges: np.ndarray) -> float | None:
    """Interpolated median from a 1px-bin histogram (None if no samples)."""
    total = int(counts.sum())
    if total == 0:
        return None
    cum = np.cumsum(counts)
    half = total / 2.0
    idx = int(np.searchsorted(cum, half))
    idx = min(idx, len(counts) - 1)
    lo = cum[idx - 1] if idx > 0 else 0
    within = counts[idx]
    frac = 0.0 if within == 0 else (half - lo) / within
    return float(edges[idx] + frac * (edges[idx + 1] - edges[idx]))


def main() -> None:
    args = parse_args()
    logger = get_logger("geometry_eval")
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalisation = load_segmentation_model(args.checkpoint, device)
    logger.info("Loaded %s (model=%s, normalisation=%s, threshold=%.3f)",
                args.checkpoint, type(model).__name__, normalisation, args.threshold)

    dataset = PatchDirectoryDataset(Path(args.patch_dir) / "test", normalisation=normalisation)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    counts = SegmentationCounts(0, 0, 0, 0)
    cldice_sum = 0.0
    cldice_n = 0
    hist_counts = np.zeros(len(_HIST_EDGES) - 1, dtype=np.int64)
    overflow = 0
    n_fp_with_gt = 0
    within_1 = within_2 = within_3 = 0
    whole_patch_fp = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            preds = (torch.sigmoid(model(images)).cpu().numpy() >= args.threshold)
            gts = batch["mask"].numpy() > 0.5
            for pred_i, gt_i in zip(preds, gts):
                p2d = pred_i.squeeze()
                g2d = gt_i.squeeze()
                counts = combine_counts([counts, compute_segmentation_counts(p2d, g2d)])
                if g2d.any():
                    cldice_sum += centerline_dice(p2d, g2d)
                    cldice_n += 1
                dists, whole = false_positive_distances(p2d, g2d)
                whole_patch_fp += whole
                if dists.size:
                    n_fp_with_gt += int(dists.size)
                    within_1 += int((dists <= 1.0).sum())
                    within_2 += int((dists <= 2.0).sum())
                    within_3 += int((dists <= 3.0).sum())
                    hist_counts += np.histogram(dists, bins=_HIST_EDGES)[0].astype(np.int64)
                    overflow += int((dists > _HIST_EDGES[-1]).sum())

    metrics = compute_metrics_from_counts(counts)
    frac = lambda n: (round(n / n_fp_with_gt, 6) if n_fp_with_gt else None)
    out = {
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "split": "test",
        "normalisation": normalisation,
        "exact_metrics": {
            "precision": round(metrics.precision, 6),
            "recall": round(metrics.recall, 6),
            "f1": round(metrics.dice, 6),       # pixel F1 == Dice
            "dice": round(metrics.dice, 6),
            "iou": round(metrics.iou, 6),
        },
        "centerline_dice": {
            "mean": round(cldice_sum / cldice_n, 6) if cldice_n else None,
            "n_target_positive_patches": cldice_n,
            "note": "per-patch clDice averaged over target-positive patches",
        },
        "fp_distance": {
            "n_fp_pixels_with_gt": n_fp_with_gt,
            "whole_patch_fp_pixels": whole_patch_fp,
            "fraction_within_1px": frac(within_1),
            "fraction_within_2px": frac(within_2),
            "fraction_within_3px": frac(within_3),
            "median_px_approx": _approx_median(hist_counts, _HIST_EDGES),
            "histogram_bin_edges": [float(x) for x in _HIST_EDGES],
            "histogram_counts": [int(x) for x in hist_counts],
            "overflow_gt_50px": overflow,
            "note": "distance from each FP pixel to nearest GT positive; whole-patch "
                    "FP (no GT in patch) counted separately, not in the histogram",
        },
        "generated": str(date.today()),
    }
    out_path = _resolve_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False))
    logger.info("exact P=%.4f R=%.4f Dice=%.4f | clDice=%s | FP<=1px=%s",
                metrics.precision, metrics.recall, metrics.dice,
                out["centerline_dice"]["mean"], out["fp_distance"]["fraction_within_1px"])
    logger.info("Saved to %s", out_path)
    print(f"Geometry eval -> {out_path}")
    print(f"  exact: P={metrics.precision:.4f} R={metrics.recall:.4f} Dice={metrics.dice:.4f}")
    print(f"  clDice(mean over {cldice_n} pos patches): {out['centerline_dice']['mean']}")
    print(f"  FP within 1/2/3px: {frac(within_1)}/{frac(within_2)}/{frac(within_3)}  "
          f"(whole-patch FP pixels: {whole_patch_fp})")


if __name__ == "__main__":
    main()
