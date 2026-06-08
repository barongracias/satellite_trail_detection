#!/usr/bin/env python
"""PS-0 — mask-dilation sensitivity check for the precision study.

Re-scores a locked checkpoint's test predictions against ground-truth masks
dilated/eroded by k pixels. The M5.6 FP-decomposition found that ~82.5% of false
positives are intra-patch (the U-Net painting trail pixels around the true trail).
If a material share of that is *label-induced* — i.e. the GT masks are
systematically thinner than the real trails — then dilating the GT by a few
pixels should sharply raise precision (those "FP" pixels were actually correct).
A sharp precision gain at small k is evidence the precision gap is partly a
labelling artefact, not model error, and must be reported as such before chasing
it with a precision-oriented loss.

Predictions are computed once at the supplied (validation-optimal) threshold; only
the GT mask is morphologically adjusted. No retraining.

Usage (CSD3 or login CPU):
    CHECKPOINT=results/checkpoints/model-best.pth \\
    THRESHOLD=0.45 \\
      python scripts/evaluation/mask_dilation_sweep.py --checkpoint ... --threshold ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PatchDirectoryDataset
from src.models.loading import load_segmentation_model
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument(
        "--kernels",
        default="-1,0,1,2,3",
        help="Comma-separated pixel radii. Negative = erode, 0 = baseline, positive = dilate.",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def _adjust(mask: np.ndarray, k: int) -> np.ndarray:
    """Dilate (k>0) or erode (k<0) a 0/1 mask by an elliptical kernel of radius |k|."""
    if k == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(k) + 1, 2 * abs(k) + 1))
    op = cv2.dilate if k > 0 else cv2.erode
    return op(mask.astype(np.uint8), kernel)


def main() -> None:
    args = parse_args()
    logger = get_logger("mask_dilation_sweep")
    seed_everything()
    kernels = [int(x) for x in args.kernels.split(",") if x.strip() != ""]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalisation = load_segmentation_model(args.checkpoint, device)
    logger.info("Loaded %s (model=%s, threshold=%.3f, kernels=%s)",
                args.checkpoint, type(model).__name__, args.threshold, kernels)

    dataset = PatchDirectoryDataset(Path(args.patch_dir) / "test", normalisation=normalisation)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    # tp/fp/fn accumulators per kernel
    tp = {k: 0 for k in kernels}
    fp = {k: 0 for k in kernels}
    fn = {k: 0 for k in kernels}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            preds = (torch.sigmoid(model(images)).cpu().numpy() >= args.threshold)
            gts = (batch["mask"].numpy() > 0.5)
            for pred_i, gt_i in zip(preds, gts):
                pred = pred_i.squeeze().astype(bool)
                gt0 = gt_i.squeeze().astype(np.uint8)
                for k in kernels:
                    gt = _adjust(gt0, k).astype(bool)
                    tp[k] += int(np.logical_and(pred, gt).sum())
                    fp[k] += int(np.logical_and(pred, ~gt).sum())
                    fn[k] += int(np.logical_and(~pred, gt).sum())

    per_kernel = {}
    for k in kernels:
        prec = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) else 0.0
        rec = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) else 0.0
        dice = 2 * tp[k] / (2 * tp[k] + fp[k] + fn[k]) if (2 * tp[k] + fp[k] + fn[k]) else 0.0
        per_kernel[str(k)] = {"precision": round(prec, 6), "recall": round(rec, 6),
                              "dice": round(dice, 6), "tp": tp[k], "fp": fp[k], "fn": fn[k]}
        logger.info("k=%+d  P=%.4f R=%.4f Dice=%.4f", k, prec, rec, dice)

    base_p = per_kernel.get("0", {}).get("precision")
    gain_k1 = (per_kernel.get("1", {}).get("precision", 0.0) - base_p) if base_p is not None else None
    interpretation = "inconclusive"
    if gain_k1 is not None:
        interpretation = "label-induced-FP-likely" if gain_k1 >= 0.05 else "model-FP-dominant"

    out = {
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "kernels": kernels,
        "per_kernel": per_kernel,
        "precision_gain_dilate_1px": None if gain_k1 is None else round(gain_k1, 6),
        "interpretation": interpretation,
        "generated": str(date.today()),
    }
    out_path = Path(args.out) if args.out else Path("results/classical") / (
        f"mask_dilation_{Path(args.checkpoint).stem.removesuffix('_best')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False))
    logger.info("precision_gain_dilate_1px=%s interpretation=%s -> %s",
                out["precision_gain_dilate_1px"], interpretation, out_path)
    print(f"Mask-dilation: +1px precision gain = {out['precision_gain_dilate_1px']} "
          f"({interpretation}) -> {out_path}")


if __name__ == "__main__":
    main()
