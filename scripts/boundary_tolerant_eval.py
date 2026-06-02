#!/usr/bin/env python
"""Step 14 — boundary-tolerant test evaluation.

Reports micro-averaged precision/recall/F1 under a sweep of boundary tolerances
(±tolerance px), alongside the exact (tolerance=0) metrics. Motivated by PS-0:
the mask-dilation check showed that small boundary tolerances materially change
precision, consistent with boundary/label-thickness effects. A tolerant metric
therefore tests sensitivity to small mask-placement differences and should be
reported explicitly when compared with the paper. Unlike the PS-0 dilation sweep,
the tolerance is applied symmetrically (GT-side for precision, prediction-side for
recall), so it does NOT trade recall for precision - both improve toward the
tolerant ceiling.

Validation-only selection must already be locked; this re-scores the locked
checkpoint's test predictions at its validation-optimal threshold.

Usage (CSD3):
    CHECKPOINT=results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth \\
    THRESHOLD=0.45 \\
      sbatch slurm/boundary_tolerant_eval.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PatchDirectoryDataset
from src.evaluation.segmentation import (
    BoundaryTolerantCounts,
    boundary_tolerant_counts,
    boundary_tolerant_metrics,
)
from src.models.loading import load_segmentation_model
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--tolerances", default="0,1,2,3", help="Comma-separated pixel tolerances.")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("boundary_tolerant_eval")
    seed_everything()
    tolerances = [int(x) for x in args.tolerances.split(",") if x.strip() != ""]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalisation = load_segmentation_model(args.checkpoint, device)
    logger.info("Loaded %s (model=%s, threshold=%.3f, tolerances=%s)",
                args.checkpoint, type(model).__name__, args.threshold, tolerances)

    dataset = PatchDirectoryDataset(Path(args.patch_dir) / "test", normalisation=normalisation)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    totals = {t: BoundaryTolerantCounts(0, 0, 0, 0) for t in tolerances}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            preds = (torch.sigmoid(model(images)).cpu().numpy() >= args.threshold)
            gts = batch["mask"].numpy() > 0.5
            for pred_i, gt_i in zip(preds, gts):
                p2d, g2d = pred_i.squeeze(), gt_i.squeeze()
                for t in tolerances:
                    totals[t] = totals[t] + boundary_tolerant_counts(p2d, g2d, t)

    per_tol = {}
    for t in tolerances:
        m = boundary_tolerant_metrics(totals[t])
        per_tol[str(t)] = {k: round(v, 6) for k, v in m.items()}
        logger.info("tolerance=%d  P=%.4f R=%.4f F1=%.4f", t, m["precision"], m["recall"], m["f1"])

    out = {
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "tolerances": tolerances,
        "tolerance_kernel": "cv2.MORPH_RECT",
        "tolerance_neighbourhood": "square/Chebyshev; 8-neighbour at tolerance=1",
        "tolerance_application": (
            "target-side dilation for precision and prediction-side dilation for recall"
        ),
        "per_tolerance": per_tol,
        "generated": str(date.today()),
    }
    out_path = Path(args.out) if args.out else Path("results/classical") / (
        f"boundary_tolerant_{Path(args.checkpoint).stem.removesuffix('_best')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False))
    logger.info("Saved to %s", out_path)
    print(f"Boundary-tolerant eval -> {out_path}")
    for t in tolerances:
        print(f"  ±{t}px: P={per_tol[str(t)]['precision']} R={per_tol[str(t)]['recall']} "
              f"F1={per_tol[str(t)]['f1']}")


if __name__ == "__main__":
    main()
