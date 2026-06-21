#!/usr/bin/env python
"""Dense full-frame Hough evaluation of the locked winner (retires the patch-canvas caveat).

Reproduces the committed patch-canvas parity Hough result on a DENSE FULL-FRAME
prediction instead of the 1:3 sampled-patch canvas. The ONLY thing that changes
vs results/classical/hough_postprocess_winner_t44_s2804_parity.json is full-frame
inference — reflect-padded, non-overlapping 528 px tiling over the whole image so
every pixel is predicted exactly once (no sampling gaps, no patch seams) — instead
of the sampled patch canvas. Every other knob is held fixed and READ from the
committed parity JSON: the detection threshold (0.45), the lower Hough-input
threshold (0.1), and hough_threshold / min_line_length / max_line_gap /
line_thickness. Ground truth is the original full-frame mask.

Inference-only on the already-locked checkpoint: NO retraining, re-tuning, or
reselection. Reuses the dense-inference machinery from
scripts/figures/decam_cold_inference.py (infer_probability_canvas) and the
two-threshold Hough scheme + scoring from src/classical/hough_runner.py
(run_hough_on_canvas), so the only methodological difference from the parity run
is the prediction canvas.

Usage (CSD3):
    CHECKPOINT=results/checkpoints/model-best.pth sbatch slurm/hough_fullframe.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

from scripts.figures.decam_cold_inference import git_commit, infer_probability_canvas
from src.classical.hough_runner import run_hough_on_canvas
from src.models.loading import load_segmentation_model
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

Image.MAX_IMAGE_PIXELS = None

_PARITY_JSON = "results/classical/hough_postprocess_winner_t44_s2804_parity.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="results/checkpoints/model-best.pth")
    p.add_argument(
        "--parity_json",
        default=_PARITY_JSON,
        help="Committed patch-canvas parity run; source of the 32 test images, the "
             "frozen Hough params/thresholds, and the patch-canvas comparison baseline.",
    )
    p.add_argument("--batch_size", type=int, default=16,
                   help="Tiles per forward pass (memory/throughput knob; no effect on metrics).")
    p.add_argument("--out", default="results/classical/hough_fullframe_winner_t44_s2804.json")
    return p.parse_args()


def _mask_path_for(image_path: str) -> str:
    return image_path.replace("_red.fits_full.png", "_red_mask.png")


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def main() -> None:
    args = parse_args()
    logger = get_logger("hough_fullframe")
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    parity = json.loads(Path(args.parity_json).read_text())
    threshold = parity["threshold"]
    hough_input_threshold = parity["hough_input_threshold"]
    hough_threshold = parity["hough_threshold"]
    min_line_length = parity["min_line_length"]
    max_line_gap = parity["max_line_gap"]
    # The committed parity JSON predates the line_thickness schema field; the run
    # used the default thickness 3 (the value Exp 6 confirmed == the 0.410 result).
    line_thickness = parity.get("line_thickness", 3)
    test_images = [r["source_image"] for r in parity["per_image"]]
    logger.info(
        "Frozen params from %s: thr=%.2f hough_in=%.2f hough_thr=%d min_len=%d max_gap=%d thick=%d | %d test images",
        args.parity_json, threshold, hough_input_threshold, hough_threshold,
        min_line_length, max_line_gap, line_thickness, len(test_images),
    )

    model, normalisation = load_segmentation_model(args.checkpoint, device)
    if normalisation != "full_image":
        raise ValueError(f"expected full_image normalisation, got {normalisation!r}")
    logger.info("Loaded %s (normalisation=%s)", args.checkpoint, normalisation)

    rows = []
    tot_gt = tot_tp_pre = tot_tp_post = tot_pred_pre = tot_pred_post = 0
    n_positive = 0
    for i, img_path in enumerate(test_images, 1):
        mask_path = _mask_path_for(img_path)
        with Image.open(img_path) as im:
            image_u8 = np.asarray(im.convert("L"), dtype=np.uint8)
        with Image.open(mask_path) as mk:
            gt_canvas = np.asarray(mk.convert("L"), dtype=np.uint8)
        if gt_canvas.shape != image_u8.shape:
            raise ValueError(
                f"mask shape {gt_canvas.shape} != image shape {image_u8.shape} for {img_path}"
            )

        prob_canvas, inf_meta = infer_probability_canvas(
            image_u8, model, device, batch_size=args.batch_size
        )
        res = run_hough_on_canvas(
            prob_canvas, gt_canvas,
            threshold=threshold,
            hough_input_threshold=hough_input_threshold,
            hough_threshold=hough_threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
            line_thickness=line_thickness,
        )

        pred_pre = int((res.binary_canvas > 0).sum())
        pred_post = int((res.combined_canvas > 0).sum())
        gt = res.gt_pixels
        is_positive = gt > 0
        fp_pre = pred_pre - res.pixels_pre
        fp_post = pred_post - res.pixels_post

        tot_gt += gt
        tot_tp_pre += res.pixels_pre
        tot_tp_post += res.pixels_post
        tot_pred_pre += pred_pre
        tot_pred_post += pred_post
        if is_positive:
            n_positive += 1

        rows.append({
            "source_image": img_path,
            "is_positive": is_positive,
            "image_shape": [int(image_u8.shape[0]), int(image_u8.shape[1])],
            "n_tiles": inf_meta["n_patches"],
            "gt_pixels": gt,
            "pred_pixels_pre": pred_pre,
            "pred_pixels_post": pred_post,
            "tp_pixels_pre": res.pixels_pre,
            "tp_pixels_post": res.pixels_post,
            "fp_pixels_pre": fp_pre,
            "fp_pixels_post": fp_post,
            "hough_fp_added_pixels": fp_post - fp_pre,
            "pixel_precision_pre": _ratio(res.pixels_pre, pred_pre),
            "pixel_precision_post": _ratio(res.pixels_post, pred_post),
            "pixel_recall_pre": _ratio(res.pixels_pre, gt),
            "pixel_recall_post": _ratio(res.pixels_post, gt),
        })
        logger.info(
            "[%2d/%d] %s | gt=%d | precision pre/post=%.4f/%.4f | recall pre/post=%.4f/%.4f",
            i, len(test_images), Path(img_path).name, gt,
            (res.pixels_pre / pred_pre) if pred_pre else 0.0,
            (res.pixels_post / pred_post) if pred_post else 0.0,
            (res.pixels_pre / gt) if gt else 0.0,
            (res.pixels_post / gt) if gt else 0.0,
        )
        del prob_canvas, gt_canvas, image_u8, res

    tot_fp_pre = tot_pred_pre - tot_tp_pre
    tot_fp_post = tot_pred_post - tot_tp_post
    ff = {
        "pixel_precision_pre": _ratio(tot_tp_pre, tot_pred_pre),
        "pixel_precision_post": _ratio(tot_tp_post, tot_pred_post),
        "pixel_recall_pre": _ratio(tot_tp_pre, tot_gt),
        "pixel_recall_post": _ratio(tot_tp_post, tot_gt),
    }
    ff["pixel_fnr_pre"] = None if ff["pixel_recall_pre"] is None else round(1.0 - ff["pixel_recall_pre"], 6)
    ff["pixel_fnr_post"] = None if ff["pixel_recall_post"] is None else round(1.0 - ff["pixel_recall_post"], 6)

    patch = {
        "pixel_precision_pre": parity["pixel_precision_pre"],
        "pixel_precision_post": parity["pixel_precision_post"],
        "pixel_recall_pre": parity["pixel_recall_pre"],
        "pixel_recall_post": parity["pixel_recall_post"],
        "pixel_fnr_pre": parity["pixel_fnr_pre"],
        "pixel_fnr_post": parity["pixel_fnr_post"],
        "source": args.parity_json,
    }
    delta = {
        k: round(ff[k] - patch[k], 6)
        for k in ("pixel_precision_pre", "pixel_precision_post", "pixel_recall_pre", "pixel_recall_post")
        if ff[k] is not None and patch[k] is not None
    }
    bit_identical = (
        tot_gt == parity["total_gt_pixels"]
        and tot_pred_pre == parity["total_pred_pixels_pre"]
        and tot_pred_post == parity["total_pred_pixels_post"]
        and tot_tp_pre == parity["total_pixels_covered_pre"]
        and tot_tp_post == parity["total_pixels_covered_post"]
    )

    # Optional context: the genuinely different 1:3 negative-sampled canvas
    # (the non-parity run), so the JSON records all three side by side.
    sampled = None
    sampled_path = args.parity_json.replace("_parity.json", ".json")
    sp = Path(sampled_path)
    if sp != Path(args.parity_json) and sp.exists():
        s = json.loads(sp.read_text())
        sampled = {
            "pixel_precision_pre": s.get("pixel_precision_pre"),
            "pixel_precision_post": s.get("pixel_precision_post"),
            "pixel_recall_pre": s.get("pixel_recall_pre"),
            "pixel_recall_post": s.get("pixel_recall_post"),
            "source": sampled_path,
            "note": "1:3 pos:neg sampled patch canvas (data/patches); fewer negatives -> "
                    "fewer off-trail FP -> optimistic precision. Not paper-comparable.",
        }

    if bit_identical:
        finding = (
            "The dense full-frame prediction reproduces the committed parity run's pixel "
            "totals EXACTLY (bit-identical gt/pred/tp counts) via an independent code path: "
            "the full raw PNG is loaded, full-image normalisation is computed on the fly, the "
            "frame is tiled+inferred, and scoring uses the ORIGINAL full-frame mask. This "
            "confirms the patch-canvas parity run (data/patches_test_full = all patches, full "
            "20x20 tiling) was already a faithful dense full-frame prediction: the post-Hough "
            "precision fall (0.780 -> 0.410) is a genuine property of the dense prediction, NOT "
            "a patch-sampling or seam artifact. The caveat is retired."
        )
    else:
        finding = (
            "The dense full-frame prediction differs from the patch-canvas parity run; see the "
            "comparison block for the measured dense numbers that retire the caveat."
        )

    result = {
        "schema_version": 1,
        "mode": "dense_full_frame",
        "checkpoint": str(args.checkpoint),
        "normalisation": normalisation,
        "threshold": threshold,
        "hough_input_threshold": hough_input_threshold,
        "hough_threshold": hough_threshold,
        "min_line_length": min_line_length,
        "max_line_gap": max_line_gap,
        "line_thickness": line_thickness,
        "n_test_images": len(rows),
        "n_positive_images": n_positive,
        "tiling": {
            "scheme": "reflect-padded non-overlapping 528 px tiles (infer_probability_canvas)",
            "overlap": False,
            "stride_px": 528,
            "note": "every pixel predicted exactly once; dense full-frame coverage with no "
                    "1:3 negative sub-sampling and no inter-patch gaps.",
        },
        "total_gt_pixels": tot_gt,
        "total_pred_pixels_pre": tot_pred_pre,
        "total_pred_pixels_post": tot_pred_post,
        "total_tp_pixels_pre": tot_tp_pre,
        "total_tp_pixels_post": tot_tp_post,
        "total_fp_pixels_pre": tot_fp_pre,
        "total_fp_pixels_post": tot_fp_post,
        "hough_fp_added_pixels": tot_fp_post - tot_fp_pre,
        "pixel_precision_pre": ff["pixel_precision_pre"],
        "pixel_precision_post": ff["pixel_precision_post"],
        "pixel_recall_pre": ff["pixel_recall_pre"],
        "pixel_recall_post": ff["pixel_recall_post"],
        "pixel_fnr_pre": ff["pixel_fnr_pre"],
        "pixel_fnr_post": ff["pixel_fnr_post"],
        "bit_identical_to_patch_parity": bit_identical,
        "finding": finding,
        "comparison": {
            "dense_full_frame": ff,
            "patch_canvas_parity": patch,
            "patch_canvas_sampled_1to3": sampled,
            "delta_full_frame_minus_patch": delta,
        },
        "precision_scope_note": (
            "pixel_precision_* and FP totals are micro-aggregated over ALL test images "
            "(negative images contribute FP with no TP); recall/coverage aggregate over the "
            "trail pixels of positive images. Same scope as the patch-canvas parity run, so "
            "the only methodological difference is the dense full-frame canvas."
        ),
        "opencv_rng_note": parity.get("opencv_rng_note"),
        "deviations": [
            "Tiling is non-overlapping (stride == 528 px tile size) with reflect padding to a "
            "patch-size multiple, reusing infer_probability_canvas; no overlap-averaging is "
            "applied. Full frames are exact 528 px multiples (10560 = 20x528) so no padding is "
            "added in practice; every pixel is predicted exactly once.",
            "Full-frame normalisation uses full-image mean/std of the display PNG "
            "(full_image_stats), the locked full_image z-score, applied per tile.",
            "Ground truth is the original full-frame *_red_mask.png (all trail pixels), whereas "
            "the patch-canvas parity GT was reconstructed from the retained positive patches; "
            "this is the intended like-for-like-with-the-paper change.",
            "The entire display frame is scored as-is, including any zero/border regions; no "
            "separate invalid-pixel masking is applied.",
        ],
        "provenance": {
            "git_commit": git_commit(),
            "generated": str(date.today()),
            "parity_json": args.parity_json,
        },
        "per_image": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False))

    logger.info("=== DENSE FULL-FRAME (vs patch-canvas parity) ===")
    logger.info("precision pre : %.4f  (patch %.4f)", ff["pixel_precision_pre"], patch["pixel_precision_pre"])
    logger.info("precision post: %.4f  (patch %.4f)", ff["pixel_precision_post"], patch["pixel_precision_post"])
    logger.info("recall    pre : %.4f  (patch %.4f)", ff["pixel_recall_pre"], patch["pixel_recall_pre"])
    logger.info("recall    post: %.4f  (patch %.4f)", ff["pixel_recall_post"], patch["pixel_recall_post"])
    logger.info("Saved to %s", out_path)
    print(
        f"\nDENSE FULL-FRAME  precision pre/post: {ff['pixel_precision_pre']:.4f}/{ff['pixel_precision_post']:.4f}"
        f"  recall pre/post: {ff['pixel_recall_pre']:.4f}/{ff['pixel_recall_post']:.4f}"
        f"\nPATCH-CANVAS      precision pre/post: {patch['pixel_precision_pre']:.4f}/{patch['pixel_precision_post']:.4f}"
        f"  recall pre/post: {patch['pixel_recall_pre']:.4f}/{patch['pixel_recall_post']:.4f}"
    )


if __name__ == "__main__":
    main()
