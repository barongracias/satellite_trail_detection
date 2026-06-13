#!/usr/bin/env python
"""Hough post-processing on U-Net predicted masks.

For each test image, reconstructs a probability canvas from sampled test patches,
then applies probabilistic Hough transform to detect line-like structures that the
U-Net partially detected but fell below the main threshold.  Reports three
levels of FNR/recall before and after post-processing: image-level,
patch-level, and pixel-level.

FNR logic (image-level, requires overlap with ground truth):
- pre-Hough:  detected if the thresholded prediction canvas overlaps the
  reconstructed ground-truth mask canvas in at least one pixel.
- post-Hough: detected if (thresholded prediction OR Hough line canvas)
  overlaps the ground-truth canvas. Hough runs on a lower-threshold canvas
  (default 0.1) so it can recover faint detections below the main threshold;
  if it ran on the already-thresholded binary canvas it could never reduce
  FNR beyond what U-Net achieves alone.

Without the overlap requirement, any false-positive pixel would count as a
detection, trivially driving FNR to 0; the target-canvas mask grounds the
metric in the actual trail location.

Patch-level FNR uses the same overlap rule but on a per-patch denominator
(any test patch with positive_pixel_fraction > 0 in the manifest). The
denominator is ~28x the image-level one (864 vs 31 on our test set) and
gives Hough headroom to recover trail fragments the U-Net underpaints.

Pixel-level recall is GT-pixel coverage on positive images aggregated
across the test set: (binary & target).sum() / target.sum() and the
union with the Hough line canvas. Reports the absolute fraction of true
trail pixels covered, so a Hough that completes fragmented detections
shows up here even if every image is already detected at image-level.

Deviations from Stoppa et al. 2024:
- Hough is applied to the reconstructed patch-subset canvas rather than a
  dense full-image prediction, because the pre-built test patches use 1:3
  pos:neg sampling (all positive patches are retained).
- Image-level FNR: a test image is trail-positive if any of its test patches
  have positive_pixel_fraction > 0 in the manifest.

Usage (CSD3):
    CHECKPOINT=results/checkpoints/model-best.pth \
    THRESHOLD=0.45 \
      sbatch slurm/hough_postprocess.sbatch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.classical.hough_runner import run_hough_on_canvas
from src.config.constants import PAPER_FNR_POST_HOUGH, PAPER_FNR_PRE_HOUGH, PATCH_SIZE
from src.data.transforms import normalise_tensor
from src.models.loading import load_segmentation_model
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

Image.MAX_IMAGE_PIXELS = None

_PATCH_SIZE = PATCH_SIZE
_YX_RE = re.compile(r"_(\d+)_(\d+)_image$")
_HOUGH_MAX_BATCH = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hough_input_threshold", type=float, default=0.1,
                   help="Lower probability threshold for Hough input canvas — allows "
                        "Hough to recover faint detections below the main threshold.")
    p.add_argument("--hough_threshold", type=int, default=50)
    p.add_argument("--min_line_length", type=int, default=100)
    p.add_argument("--max_line_gap", type=int, default=250)
    p.add_argument("--line_thickness", type=int, default=3)
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: results/classical/hough_postprocess_<tag>.json "
             "where <tag> is derived from the checkpoint stem (trailing '_best' stripped).",
    )
    return p.parse_args()


def _resolve_out_path(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    stem = Path(args.checkpoint).stem
    tag = stem[:-5] if stem.endswith("_best") else stem
    return Path("results/classical") / f"hough_postprocess_{tag}.json"


def load_model(checkpoint_path: str, device: torch.device) -> tuple[torch.nn.Module, str]:
    return load_segmentation_model(checkpoint_path, device)


def _parse_yx(patch_path: str) -> tuple[int, int]:
    m = _YX_RE.search(Path(patch_path).stem)
    if m is None:
        raise ValueError(f"Cannot parse y,x from patch filename: {patch_path}")
    return int(m.group(1)), int(m.group(2))


def _load_normalised_patch(
    patch_path: str,
    normalisation: str,
    full_image_mean: float | None = None,
    full_image_std: float | None = None,
) -> torch.Tensor:
    """Load + normalise one patch as a (1, H, W) float tensor (still on CPU)."""
    with Image.open(patch_path) as img:
        t = TF.pil_to_tensor(img.convert("L")).float() / 255.0
    return normalise_tensor(t, normalisation, full_image_mean, full_image_std)


def _infer_batch(
    patches: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Forward a stacked batch of patches; return (N, H, W) sigmoid probs."""
    batch = torch.stack(patches).to(device, non_blocking=True)
    with torch.inference_mode():
        probs = torch.sigmoid(model(batch))
    return probs.squeeze(1).cpu().numpy()


def _chunks(seq: list, size: int):
    """Yield successive size-bounded slices of seq."""
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def main() -> None:
    args = parse_args()
    if not (0.0 < args.hough_input_threshold <= args.threshold <= 1.0):
        raise ValueError(
            f"Require 0 < hough_input_threshold ({args.hough_input_threshold}) "
            f"<= threshold ({args.threshold}) <= 1"
        )
    logger = get_logger("hough_postprocess")
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, normalisation = load_model(args.checkpoint, device)
    logger.info(
        "Loaded %s (normalisation=%s, threshold=%.3f, hough_input_threshold=%.3f)",
        args.checkpoint, normalisation, args.threshold, args.hough_input_threshold,
    )

    manifest = pd.read_csv(Path(args.patch_dir) / "manifest.csv")
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    logger.info("Test patches in manifest: %d", len(test_df))

    has_full_image_stats = {"image_mean", "image_std"}.issubset(test_df.columns)
    if normalisation == "full_image":
        if not has_full_image_stats:
            logger.warning(
                "Checkpoint trained with normalisation=full_image but manifest lacks "
                "image_mean/image_std; per-patch z-score fallback will be used."
            )
        else:
            n_nan = int(test_df[["image_mean", "image_std"]].isna().any(axis=1).sum())
            if n_nan:
                logger.warning(
                    "normalisation=full_image: %d/%d test rows have NaN image_mean/"
                    "image_std; per-patch z-score fallback will be used for those rows.",
                    n_nan, len(test_df),
                )

    groups = test_df.groupby("source_image")
    logger.info("Unique test images: %d", len(groups))

    rows = []
    n_positive = 0
    n_fn_pre = 0
    n_fn_post = 0
    total_pos_patches = 0
    total_fn_pre_patch = 0
    total_fn_post_patch = 0
    total_gt_pixels = 0
    total_pixels_pre = 0
    total_pixels_post = 0
    # FP accounting (schema_version 2) over ALL test images — negative images
    # contribute false positives with no true positives, so precision must not
    # be restricted to positive images the way the recall aggregates are.
    total_pred_pixels_pre = 0
    total_pred_pixels_post = 0
    total_tp_pixels_pre = 0
    total_tp_pixels_post = 0

    for source_image, group in groups:
        is_positive = bool((group["positive_pixel_fraction"] > 0).any())

        yx_list = [_parse_yx(p) for p in group["patch_path"]]
        canvas_h = max(y for y, _ in yx_list) + _PATCH_SIZE
        canvas_w = max(x for _, x in yx_list) + _PATCH_SIZE
        prob_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        target_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        # Per-source-image inference in chunks: cap load + stack + forward at
        # _HOUGH_MAX_BATCH to bound the per-step working set. Peak memory per
        # image is dominated by the two full-image canvases declared above
        # (prob_canvas float32 ≈ 446 MB at 10560×10560, target_canvas uint8
        # ≈ 112 MB) plus the current chunk's stacked-patch tensors
        # (chunk × 1 × 528 × 528 × 4 bytes ≈ 71 MB at MAX_BATCH=64). The
        # canvases are inherent to image-level FNR reconstruction; chunking
        # only bounds the *inference* tensor footprint, not the canvas one.
        # Batched forward is numerically equivalent (not bitwise-identical) to
        # single-patch inference: model.eval() disables dropout and there are
        # no BatchNorm layers, but cuDNN convolution kernel selection is
        # batch-shape-dependent so floating-point output may differ at LSB
        # precision. Predictions exactly at the threshold could in principle
        # flip; for thresholded FNR the difference is negligible.
        group_rows = list(group.itertuples(index=False))
        for chunk in _chunks(group_rows, _HOUGH_MAX_BATCH):
            patches: list[torch.Tensor] = []
            coords: list[tuple[int, int]] = []
            for row in chunk:
                mean = getattr(row, "image_mean", None) if has_full_image_stats else None
                std = getattr(row, "image_std", None) if has_full_image_stats else None
                patches.append(
                    _load_normalised_patch(row.patch_path, normalisation, mean, std)
                )
                coords.append(_parse_yx(row.patch_path))

            probs = _infer_batch(patches, model, device)   # (N, H, W) numpy

            for (y, x), prob, row in zip(coords, probs, chunk):
                prob_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE] = np.maximum(
                    prob_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE], prob
                )
                with Image.open(row.mask_path) as msk:
                    mask_patch = np.asarray(msk.convert("L"), dtype=np.uint8)
                target_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE] = np.maximum(
                    target_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE], mask_patch
                )

        hough_result = run_hough_on_canvas(
            prob_canvas,
            target_canvas,
            threshold=args.threshold,
            hough_input_threshold=args.hough_input_threshold,
            hough_threshold=args.hough_threshold,
            min_line_length=args.min_line_length,
            max_line_gap=args.max_line_gap,
            line_thickness=args.line_thickness,
        )

        detected_pre = hough_result.detected_pre
        detected_post = hough_result.detected_post
        binary_bool = hough_result.binary_canvas > 0
        target_bool = target_canvas > 0
        combined_bool = hough_result.combined_canvas > 0

        n_patches_pos = 0
        n_fn_pre_patch = 0
        n_fn_post_patch = 0
        for row in group_rows:
            if row.positive_pixel_fraction <= 0:
                continue
            y, x = _parse_yx(row.patch_path)
            sl_y = slice(y, y + _PATCH_SIZE)
            sl_x = slice(x, x + _PATCH_SIZE)
            tgt_patch = target_bool[sl_y, sl_x]
            if not tgt_patch.any():
                continue
            n_patches_pos += 1
            if not (binary_bool[sl_y, sl_x] & tgt_patch).any():
                n_fn_pre_patch += 1
            if not (combined_bool[sl_y, sl_x] & tgt_patch).any():
                n_fn_post_patch += 1

        gt_pixels = hough_result.gt_pixels
        pixels_pre = hough_result.pixels_pre
        pixels_post = hough_result.pixels_post

        # Per-image FP accounting on the canvases already in memory: predicted
        # pixel totals minus GT-overlapping (TP) pixels. combined ⊇ binary, so
        # fp_post − fp_pre is exactly the Hough-added FP mass.
        pred_pixels_pre = int(binary_bool.sum())
        pred_pixels_post = int(combined_bool.sum())
        fp_pixels_pre = pred_pixels_pre - pixels_pre
        fp_pixels_post = pred_pixels_post - pixels_post
        total_pred_pixels_pre += pred_pixels_pre
        total_pred_pixels_post += pred_pixels_post
        total_tp_pixels_pre += pixels_pre
        total_tp_pixels_post += pixels_post

        if is_positive:
            n_positive += 1
            if not detected_pre:
                n_fn_pre += 1
            if not detected_post:
                n_fn_post += 1
            total_pos_patches += n_patches_pos
            total_fn_pre_patch += n_fn_pre_patch
            total_fn_post_patch += n_fn_post_patch
            total_gt_pixels += gt_pixels
            total_pixels_pre += pixels_pre
            total_pixels_post += pixels_post

        logger.info(
            "%s | positive=%s | pre=%s | post=%s | patch FN pre/post=%d/%d | "
            "pixel recall pre/post=%.4f/%.4f",
            Path(source_image).name, is_positive, detected_pre, detected_post,
            n_fn_pre_patch, n_fn_post_patch,
            pixels_pre / gt_pixels if gt_pixels else 0.0,
            pixels_post / gt_pixels if gt_pixels else 0.0,
        )
        rows.append({
            "source_image": str(source_image),
            "is_positive": is_positive,
            "detected_pre_hough": detected_pre,
            "detected_post_hough": detected_post,
            "n_positive_patches": n_patches_pos,
            "n_fn_pre_patch": n_fn_pre_patch,
            "n_fn_post_patch": n_fn_post_patch,
            "gt_pixels": gt_pixels,
            "pixels_covered_pre": pixels_pre,
            "pixels_covered_post": pixels_post,
            "pred_pixels_pre": pred_pixels_pre,
            "pred_pixels_post": pred_pixels_post,
            "fp_pixels_pre": fp_pixels_pre,
            "fp_pixels_post": fp_pixels_post,
            "hough_fp_added_pixels": fp_pixels_post - fp_pixels_pre,
            "pixel_precision_pre": (
                None if pred_pixels_pre == 0
                else round(pixels_pre / pred_pixels_pre, 6)
            ),
            "pixel_precision_post": (
                None if pred_pixels_post == 0
                else round(pixels_post / pred_pixels_post, 6)
            ),
        })

    if n_positive > 0:
        fnr_pre = n_fn_pre / n_positive
        fnr_post = n_fn_post / n_positive
    else:
        fnr_pre = None
        fnr_post = None

    if total_pos_patches > 0:
        fnr_pre_patch = total_fn_pre_patch / total_pos_patches
        fnr_post_patch = total_fn_post_patch / total_pos_patches
    else:
        fnr_pre_patch = None
        fnr_post_patch = None

    if total_gt_pixels > 0:
        pixel_recall_pre = total_pixels_pre / total_gt_pixels
        pixel_recall_post = total_pixels_post / total_gt_pixels
    else:
        pixel_recall_pre = None
        pixel_recall_post = None

    total_fp_pixels_pre = total_pred_pixels_pre - total_tp_pixels_pre
    total_fp_pixels_post = total_pred_pixels_post - total_tp_pixels_post
    pixel_precision_pre = (
        total_tp_pixels_pre / total_pred_pixels_pre if total_pred_pixels_pre else None
    )
    pixel_precision_post = (
        total_tp_pixels_post / total_pred_pixels_post if total_pred_pixels_post else None
    )

    result = {
        "schema_version": 2,
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "hough_input_threshold": args.hough_input_threshold,
        "hough_threshold": args.hough_threshold,
        "min_line_length": args.min_line_length,
        "max_line_gap": args.max_line_gap,
        "line_thickness": args.line_thickness,
        "n_test_images": len(rows),
        "n_positive_images": n_positive,
        "n_positive_patches": total_pos_patches,
        "fnr_pre_hough": None if fnr_pre is None else round(fnr_pre, 6),
        "fnr_post_hough": None if fnr_post is None else round(fnr_post, 6),
        "fnr_pre_patch": None if fnr_pre_patch is None else round(fnr_pre_patch, 6),
        "fnr_post_patch": None if fnr_post_patch is None else round(fnr_post_patch, 6),
        "pixel_recall_pre": None if pixel_recall_pre is None else round(pixel_recall_pre, 6),
        "pixel_recall_post": None if pixel_recall_post is None else round(pixel_recall_post, 6),
        "pixel_fnr_pre": None if pixel_recall_pre is None else round(1.0 - pixel_recall_pre, 6),
        "pixel_fnr_post": None if pixel_recall_post is None else round(1.0 - pixel_recall_post, 6),
        "total_gt_pixels": total_gt_pixels,
        "total_pixels_covered_pre": total_pixels_pre,
        "total_pixels_covered_post": total_pixels_post,
        "total_pred_pixels_pre": total_pred_pixels_pre,
        "total_pred_pixels_post": total_pred_pixels_post,
        "total_fp_pixels_pre": total_fp_pixels_pre,
        "total_fp_pixels_post": total_fp_pixels_post,
        "hough_fp_added_pixels": total_fp_pixels_post - total_fp_pixels_pre,
        "pixel_precision_pre": (
            None if pixel_precision_pre is None else round(pixel_precision_pre, 6)
        ),
        "pixel_precision_post": (
            None if pixel_precision_post is None else round(pixel_precision_post, 6)
        ),
        "precision_scope_note": (
            "pixel_precision_* and FP totals are micro-aggregated over ALL test "
            "images (negative images contribute FP with no TP); the recall/"
            "coverage aggregates above keep their original positive-image-only "
            "scope for backwards compatibility. Post-Hough strict precision is "
            "depressed partly by the 3 px line-drawing convention rasterised "
            "onto 1-3 px masks; report alongside tolerant metrics."
        ),
        "opencv_rng_note": (
            "cv2.HoughLinesP is the probabilistic Hough variant and the global "
            "seeding harness does not call cv2.setRNGSeed, so reruns are near- "
            "but not guaranteed bit-identical. The locked runner is left "
            "unchanged; this field records the caveat."
        ),
        "paper_fnr_pre_hough": PAPER_FNR_PRE_HOUGH,
        "paper_fnr_post_hough": PAPER_FNR_POST_HOUGH,
        "fnr_metric_note": (
            "fnr_*_hough is IMAGE-level detection FNR (fraction of positive test images "
            "with no detected-pixel/GT overlap); it saturates at 0 once every positive "
            "image is hit by >=1 pixel, so Hough cannot change it and it is NOT comparable "
            "to the paper. fnr_*_patch is the analogous patch-level rate. pixel_fnr_* = "
            "1 - pixel_recall is full-field trail-pixel completeness. The paper's reported "
            "FNR (paper_fnr_*) is pixel completeness and is comparable ONLY to pixel_fnr_*."
        ),
        "per_image": rows,
    }

    out_path = _resolve_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    if fnr_pre is None:
        logger.warning(
            "No positive test images; FNR pre/post undefined (n_positive=0)."
        )
    else:
        logger.info(
            "FNR image-level  pre: %.4f  post: %.4f  (paper: %.4f → %.4f)",
            fnr_pre, fnr_post, PAPER_FNR_PRE_HOUGH, PAPER_FNR_POST_HOUGH,
        )
        if fnr_pre_patch is not None:
            logger.info(
                "FNR patch-level  pre: %.4f  post: %.4f  (n_pos_patches=%d)",
                fnr_pre_patch, fnr_post_patch, total_pos_patches,
            )
        if pixel_recall_pre is not None:
            logger.info(
                "Pixel recall     pre: %.4f  post: %.4f  (gt_pixels=%d)",
                pixel_recall_pre, pixel_recall_post, total_gt_pixels,
            )
    if pixel_precision_pre is not None and pixel_precision_post is not None:
        logger.info(
            "Pixel precision  pre: %.4f  post: %.4f  (hough_fp_added=%d)",
            pixel_precision_pre, pixel_precision_post,
            total_fp_pixels_post - total_fp_pixels_pre,
        )
    logger.info("Saved to %s", out_path)
    if fnr_pre is None:
        print("\nFNR pre/post: undefined (no positive test images)")
    else:
        print(
            f"\nFNR image-level  pre: {fnr_pre:.4f}  post: {fnr_post:.4f}  "
            f"(paper: {PAPER_FNR_PRE_HOUGH} → {PAPER_FNR_POST_HOUGH})"
        )
        if fnr_pre_patch is not None:
            print(
                f"FNR patch-level  pre: {fnr_pre_patch:.4f}  post: {fnr_post_patch:.4f}  "
                f"(n_pos_patches={total_pos_patches})"
            )
        if pixel_recall_pre is not None:
            print(
                f"Pixel recall     pre: {pixel_recall_pre:.4f}  post: {pixel_recall_post:.4f}  "
                f"(gt_pixels={total_gt_pixels})"
            )
        if pixel_precision_pre is not None and pixel_precision_post is not None:
            print(
                f"Pixel precision  pre: {pixel_precision_pre:.4f}  post: {pixel_precision_post:.4f}  "
                f"(hough_fp_added={total_fp_pixels_post - total_fp_pixels_pre})"
            )


if __name__ == "__main__":
    main()
