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
    CHECKPOINT=results/checkpoints/model-best.pth \\
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PatchDirectoryDataset
from src.evaluation.segmentation import (
    SegmentationCounts,
    SurfaceDistanceCounts,
    centerline_dice,
    combine_counts,
    compute_metrics_from_counts,
    compute_segmentation_counts,
    false_positive_distances,
    surface_distance_counts_multi,
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
    p.add_argument("--threshold-source", default=None,
                   help="Path/description of the validation threshold-sweep that "
                        "selected --threshold (recorded as provenance in the output).")
    p.add_argument("--tag", default=None, help="Short tag for the output filename.")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--surface_nsd",
        action="store_true",
        help="Compute NSD / surface Dice at tau in {1,2} px on per-source-image "
             "canvases (patch borders create artificial boundary pixels, so "
             "surface metrics are computed on reconstructed canvases, not "
             "patches). Skips the patch-level metrics and writes "
             "geometry_eval_<tag>_nsd.json.",
    )
    return p.parse_args()


def _resolve_out_path(args: argparse.Namespace) -> Path:
    suffix = "_nsd" if args.surface_nsd else ""
    if args.out:
        return Path(args.out)
    if args.tag:
        return Path("results/classical") / f"geometry_eval_{args.tag}{suffix}.json"
    stem = Path(args.checkpoint).stem
    tag = stem[:-5] if stem.endswith("_best") else stem
    return Path("results/classical") / f"geometry_eval_{tag}{suffix}.json"


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


_NSD_TAUS = (1.0, 2.0)


def _run_surface_nsd(
    args: argparse.Namespace,
    model: torch.nn.Module,
    normalisation: str,
    device: torch.device,
    logger,
) -> None:
    """Canvas-level NSD / surface Dice at τ∈{1,2} px (M9.3 post-hoc diagnostic).

    Mirrors the per-source-image canvas reconstruction in
    scripts/evaluation/hough_postprocess.py (max-composited probability and
    mask canvases over the test patches of each source image) so that patch
    borders never create artificial boundary pixels. No Hough involvement:
    this scores the thresholded segmentation canvases only.
    """
    import pandas as pd
    from PIL import Image

    from scripts.evaluation.hough_postprocess import (
        _HOUGH_MAX_BATCH,
        _PATCH_SIZE,
        _chunks,
        _infer_batch,
        _load_normalised_patch,
        _parse_yx,
    )

    Image.MAX_IMAGE_PIXELS = None

    manifest = pd.read_csv(Path(args.patch_dir) / "manifest.csv")
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    has_full_image_stats = {"image_mean", "image_std"}.issubset(test_df.columns)
    groups = test_df.groupby("source_image")
    logger.info("Surface NSD: %d test patches over %d source images",
                len(test_df), len(groups))

    totals_all = {tau: SurfaceDistanceCounts(0, 0, 0, 0) for tau in _NSD_TAUS}
    totals_gt_pos = {tau: SurfaceDistanceCounts(0, 0, 0, 0) for tau in _NSD_TAUS}
    per_image: list[dict] = []
    n_scored = 0
    n_excluded_empty_both = 0
    n_empty_gt_nonempty_pred = 0
    n_empty_pred_nonempty_gt = 0

    for source_image, group in groups:
        yx_list = [_parse_yx(p) for p in group["patch_path"]]
        canvas_h = max(y for y, _ in yx_list) + _PATCH_SIZE
        canvas_w = max(x for _, x in yx_list) + _PATCH_SIZE
        prob_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        target_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        group_rows = list(group.itertuples(index=False))
        for chunk in _chunks(group_rows, _HOUGH_MAX_BATCH):
            patches = []
            coords = []
            for row in chunk:
                mean = getattr(row, "image_mean", None) if has_full_image_stats else None
                std = getattr(row, "image_std", None) if has_full_image_stats else None
                patches.append(
                    _load_normalised_patch(row.patch_path, normalisation, mean, std)
                )
                coords.append(_parse_yx(row.patch_path))
            probs = _infer_batch(patches, model, device)
            for (y, x), prob, row in zip(coords, probs, chunk):
                prob_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE] = np.maximum(
                    prob_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE], prob
                )
                with Image.open(row.mask_path) as msk:
                    mask_patch = np.asarray(msk.convert("L"), dtype=np.uint8)
                target_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE] = np.maximum(
                    target_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE], mask_patch
                )

        pred_bool = prob_canvas >= args.threshold
        gt_bool = target_canvas > 0
        del prob_canvas, target_canvas

        if not gt_bool.any() and not pred_bool.any():
            # Trivially correct: excluded from both aggregates, counted.
            n_excluded_empty_both += 1
            per_image.append({
                "source_image": str(source_image),
                "category": "empty_both_excluded",
            })
            logger.info("%s | empty GT and empty prediction — excluded",
                        Path(str(source_image)).name)
            continue

        if not gt_bool.any():
            category = "empty_gt_nonempty_pred"   # whole-image FP: scores 0
            n_empty_gt_nonempty_pred += 1
        elif not pred_bool.any():
            category = "empty_pred_nonempty_gt"   # whole-image FN: scores 0
            n_empty_pred_nonempty_gt += 1
        else:
            category = "scored"
            n_scored += 1

        counts = surface_distance_counts_multi(pred_bool, gt_bool, _NSD_TAUS)
        row_out = {"source_image": str(source_image), "category": category}
        for tau in _NSD_TAUS:
            c = counts[tau]
            totals_all[tau] = totals_all[tau] + c
            if gt_bool.any():
                totals_gt_pos[tau] = totals_gt_pos[tau] + c
            row_out[f"nsd_tau_{int(tau)}"] = (
                None if c.nsd is None else round(c.nsd, 6)
            )
            row_out[f"counts_tau_{int(tau)}"] = {
                "pred_boundary_total": c.pred_boundary_total,
                "gt_boundary_total": c.gt_boundary_total,
                "pred_boundary_within_tau": c.pred_boundary_within_tau,
                "gt_boundary_within_tau": c.gt_boundary_within_tau,
            }
        per_image.append(row_out)
        logger.info(
            "%s | %s | NSD@1=%s NSD@2=%s",
            Path(str(source_image)).name, category,
            row_out["nsd_tau_1"], row_out["nsd_tau_2"],
        )

    def _micro(totals: dict[float, SurfaceDistanceCounts]) -> dict:
        return {
            f"nsd_tau_{int(tau)}": (
                None if totals[tau].nsd is None else round(totals[tau].nsd, 6)
            )
            for tau in _NSD_TAUS
        }

    out = {
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "threshold_source": args.threshold_source,
        "split": "test",
        "normalisation": normalisation,
        "patch_dir": str(args.patch_dir),
        "pixel_spacing": 1.0,
        "taus_px": [float(t) for t in _NSD_TAUS],
        "aggregation": "canvas-level (per source image), micro over boundary counts",
        "micro_all_images": _micro(totals_all),
        "micro_gt_positive_images": _micro(totals_gt_pos),
        "n_images_scored_both_nonempty": n_scored,
        "n_images_excluded_empty_both": n_excluded_empty_both,
        "n_images_empty_gt_nonempty_pred": n_empty_gt_nonempty_pred,
        "n_images_empty_pred_nonempty_gt": n_empty_pred_nonempty_gt,
        "metric_note": (
            "NSD computed on per-source-image canvases (patch borders would "
            "create artificial boundary pixels). Pixel spacing is unity. With "
            "unit spacing and integer-pixel boundaries, NSD and surface Dice "
            "at tolerance tau (Nikolov et al. 2021) are the same quantity; "
            "both vocabularies map to these numbers. Empty-GT+empty-pred "
            "images are excluded; empty-GT+non-empty-pred images score 0 and "
            "are included in micro_all_images but not micro_gt_positive_images "
            "(counts above). On 1-3 px trails the mask boundary is essentially "
            "the mask, so NSD@1 is expected to track the existing +/-1 px "
            "boundary-tolerant F1 — equivalence, not independent confirmation."
        ),
        "per_image": per_image,
        "generated": str(date.today()),
    }
    out_path = _resolve_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False))
    logger.info("micro (all images): %s | micro (GT-positive): %s",
                out["micro_all_images"], out["micro_gt_positive_images"])
    logger.info("Saved to %s", out_path)
    print(f"Surface NSD eval -> {out_path}")
    print(f"  micro all images:    {out['micro_all_images']}")
    print(f"  micro GT-positive:   {out['micro_gt_positive_images']}")
    print(f"  scored/excluded/whole-FP/whole-FN: {n_scored}/{n_excluded_empty_both}/"
          f"{n_empty_gt_nonempty_pred}/{n_empty_pred_nonempty_gt}")


def main() -> None:
    args = parse_args()
    logger = get_logger("geometry_eval")
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalisation = load_segmentation_model(args.checkpoint, device)
    logger.info("Loaded %s (model=%s, normalisation=%s, threshold=%.3f)",
                args.checkpoint, type(model).__name__, normalisation, args.threshold)

    if args.surface_nsd:
        _run_surface_nsd(args, model, normalisation, device, logger)
        return

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
    n_fp_total = n_fp_with_gt + whole_patch_fp
    # Two denominators: distance-defined FP (patches with GT) vs ALL FP (whole-patch
    # FP have no defined distance and are reported as non-boundary-adjacent).
    frac = lambda n: (round(n / n_fp_with_gt, 6) if n_fp_with_gt else None)
    frac_all = lambda n: (round(n / n_fp_total, 6) if n_fp_total else None)
    out = {
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "threshold_source": args.threshold_source,
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
            "n_fp_pixels_total": n_fp_total,
            "fraction_within_1px": frac(within_1),
            "fraction_within_2px": frac(within_2),
            "fraction_within_3px": frac(within_3),
            "fraction_within_1px_all_fp": frac_all(within_1),
            "fraction_within_2px_all_fp": frac_all(within_2),
            "fraction_within_3px_all_fp": frac_all(within_3),
            "median_px_approx": _approx_median(hist_counts, _HIST_EDGES),
            "histogram_bin_edges": [float(x) for x in _HIST_EDGES],
            "histogram_counts": [int(x) for x in hist_counts],
            "overflow_gt_50px": overflow,
            "note": "fraction_within_*px is over distance-defined FP (patches with GT); "
                    "*_all_fp uses all FP (incl. whole-patch FP, which have no defined "
                    "distance and are treated as non-boundary-adjacent). Histogram is "
                    "over distance-defined FP only.",
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
    print(f"  FP within 1/2/3px (distance-defined): {frac(within_1)}/{frac(within_2)}/{frac(within_3)}")
    print(f"  FP within 1/2/3px (all FP incl. whole-patch): {frac_all(within_1)}/{frac_all(within_2)}/{frac_all(within_3)}"
          f"  (whole-patch FP pixels: {whole_patch_fp} of {n_fp_total})")


if __name__ == "__main__":
    main()
