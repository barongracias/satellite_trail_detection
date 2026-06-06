#!/usr/bin/env python
"""Step 6 — false-positive decomposition driver for the M5.6 winner.

Runs the winning segmentation model on the test split at its validation-optimal
threshold, then classifies every false-positive pixel as inter-patch (on empty
patches — recoverable by a classifier gate) or intra-patch (inside true-trail
patches — not recoverable). The result tells whether the two-stage gate can
materially help, so it precedes the Step 7 two-stage evaluation.

Validation-only selection must already be locked: this inspects test predictions
of the chosen checkpoint and must not be used to choose a model.

Usage (CSD3):
    CHECKPOINT=results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth \\
    THRESHOLD=0.45 \\
      sbatch slurm/fp_decompose.sbatch
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

from src.analysis.fp_decomposition import decompose_false_positives
from src.data.dataset import PatchDirectoryDataset
from src.models.loading import load_segmentation_model
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Validation-optimal threshold (t*) for the winning checkpoint.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON. Default: results/classical/fp_decomposition_<tag>.json",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def _resolve_out(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    stem = Path(args.checkpoint).stem
    tag = stem[:-5] if stem.endswith("_best") else stem
    return Path("results/classical") / f"fp_decomposition_{tag}.json"


def main() -> None:
    args = parse_args()
    logger = get_logger("fp_decompose")
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalisation = load_segmentation_model(args.checkpoint, device)
    logger.info(
        "Loaded %s (model=%s, normalisation=%s, threshold=%.3f)",
        args.checkpoint, type(model).__name__, normalisation, args.threshold,
    )

    dataset = PatchDirectoryDataset(Path(args.patch_dir) / "test", normalisation=normalisation)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    logger.info("Test patches: %d", len(dataset))

    samples: list[tuple[np.ndarray, np.ndarray, bool]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            preds = probs >= args.threshold
            gts = batch["mask"].numpy() > 0.5
            for pred_i, gt_i in zip(preds, gts):
                gt_sq = gt_i.squeeze()
                samples.append((pred_i.squeeze(), gt_sq, bool(gt_sq.any())))

    result = decompose_false_positives(samples)
    out = {
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "n_patches": len(samples),
        "inter_patch_fp_pixels": result.inter_patch_fp_pixels,
        "intra_patch_fp_pixels": result.intra_patch_fp_pixels,
        "inter_patch_fp_fraction": round(result.inter_patch_fp_fraction, 6),
        "empty_patch_fp_rate": round(result.empty_patch_fp_rate, 6),
        "interpretation": result.interpretation,
        "generated": str(date.today()),
    }
    out_path = _resolve_out(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False))
    logger.info(
        "inter_patch_fp_fraction=%.4f empty_patch_fp_rate=%.4f interpretation=%s",
        result.inter_patch_fp_fraction, result.empty_patch_fp_rate, result.interpretation,
    )
    logger.info("Saved to %s", out_path)
    print(
        f"FP decomposition: inter_fraction={result.inter_patch_fp_fraction:.4f} "
        f"interpretation={result.interpretation} -> {out_path}"
    )


if __name__ == "__main__":
    main()
