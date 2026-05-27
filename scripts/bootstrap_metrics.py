#!/usr/bin/env python
"""Bootstrapped confidence intervals on test segmentation metrics (M5.3).

Runs inference on the test split with the supplied checkpoint and threshold,
accumulates per-patch SegmentationCounts (TP/FP/FN/TN), then computes two CIs:

- Cluster bootstrap by source image (primary): resamples source images with
  replacement, treating within-image patches as a unit. Captures lighting,
  noise, and annotator correlation that an i.i.d. patch bootstrap ignores.
- Patch-level bootstrap (secondary): i.i.d. resample of per-patch counts.
  The gap between the two CI widths quantifies within-image correlation.

Threshold is required — pass the F1-optimal value from the matching
threshold_sweep_<tag>.json. Output JSON pairs with the existing
threshold_sweep_<tag>.json and hough_postprocess_<tag>.json artefacts.

Usage:
    CHECKPOINT=results/checkpoints/unet_<tag>_best.pth \\
    THRESHOLD=0.72 \\
      sbatch slurm/bootstrap_metrics.sbatch

or directly:
    python scripts/bootstrap_metrics.py \\
        --checkpoint results/checkpoints/unet_<tag>_best.pth \\
        --threshold 0.72
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config.constants import GLOBAL_SEED
from src.data.dataset import PatchDirectoryDataset
from src.evaluation.segmentation import (
    SegmentationCounts,
    bootstrap_metrics_cluster,
    bootstrap_metrics_patch,
    combine_counts,
    compute_metrics_from_counts,
)
from src.models.unet import UNet
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
        help="Threshold for binarising predictions. Pass the F1-optimal value "
             "from the matching threshold_sweep_<tag>.json.",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument(
        "--tag",
        default=None,
        help="Suffix for output filename. Defaults to checkpoint stem with "
             "trailing '_best' stripped: bootstrap_<tag>.json.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: results/classical/bootstrap_<tag>.json",
    )
    return p.parse_args()


def _resolve_tag(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    stem = Path(args.checkpoint).stem
    return stem[:-5] if stem.endswith("_best") else stem


def _resolve_out_path(args: argparse.Namespace, tag: str) -> Path:
    if args.out:
        return Path(args.out)
    return Path("results/classical") / f"bootstrap_{tag}.json"


def _counts_from_pair(pred_bool: torch.Tensor, target_bool: torch.Tensor) -> SegmentationCounts:
    not_pred = ~pred_bool
    not_target = ~target_bool
    tp = int(torch.logical_and(pred_bool, target_bool).sum())
    fp = int(torch.logical_and(pred_bool, not_target).sum())
    fn = int(torch.logical_and(not_pred, target_bool).sum())
    tn = int(torch.logical_and(not_pred, not_target).sum())
    return SegmentationCounts(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def main() -> None:
    args = parse_args()
    logger = get_logger("bootstrap_metrics")
    seed_everything(args.seed)

    tag = _resolve_tag(args)
    out_path = _resolve_out_path(args, tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | tag: %s | threshold: %.4f", device, tag, args.threshold)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = UNet(base_channels=cfg.get("base_channels", 8)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    normalisation = cfg.get("normalisation", "fixed")
    logger.info("Loaded checkpoint (normalisation=%s)", normalisation)

    dataset = PatchDirectoryDataset(
        Path(args.patch_dir) / "test", normalisation=normalisation
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    logger.info("Test patches: %d", len(dataset))

    # DataLoader runs shuffle=False, so dataset emits in dataset.records order.
    # source_image[i] is therefore the source for the i-th emitted patch.
    source_images = dataset.records["source_image"].tolist()
    if len(source_images) != len(dataset):
        raise RuntimeError(
            f"Manifest rows ({len(source_images)}) do not match dataset length "
            f"({len(dataset)}); refusing to proceed."
        )

    patch_counts: list[SegmentationCounts] = []
    per_image_counts: dict[str, list[SegmentationCounts]] = {}

    idx = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            masks = batch["mask"].to(device, dtype=torch.float32, non_blocking=True)
            probs = torch.sigmoid(model(images))
            preds = probs >= args.threshold
            targets = masks > 0.5
            for i in range(preds.shape[0]):
                counts = _counts_from_pair(preds[i], targets[i])
                patch_counts.append(counts)
                per_image_counts.setdefault(source_images[idx], []).append(counts)
                idx += 1

    if idx != len(dataset):
        raise RuntimeError(
            f"Processed {idx} patches but dataset reports {len(dataset)}; "
            f"refusing to proceed (likely DataLoader drop_last issue)."
        )

    logger.info(
        "Accumulated %d per-patch counts across %d source images",
        len(patch_counts), len(per_image_counts),
    )

    image_counts = {
        src: combine_counts(items) for src, items in per_image_counts.items()
    }

    point_counts = combine_counts(patch_counts)
    point_metrics = compute_metrics_from_counts(point_counts)

    cluster_ci = bootstrap_metrics_cluster(
        image_counts, n_resamples=args.n_resamples, seed=args.seed
    )
    patch_ci = bootstrap_metrics_patch(
        patch_counts, n_resamples=args.n_resamples, seed=args.seed
    )

    def _format(ci_dict: dict[str, tuple[float, float, float]]) -> dict[str, dict[str, float]]:
        return {
            name: {"point": pt, "lo": lo, "hi": hi}
            for name, (pt, lo, hi) in ci_dict.items()
        }

    result = {
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "threshold": args.threshold,
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "n_patches": len(patch_counts),
        "n_source_images": len(per_image_counts),
        "point_metrics": {
            "precision": point_metrics.precision,
            "recall": point_metrics.recall,
            "dice": point_metrics.dice,
            "iou": point_metrics.iou,
        },
        "cluster_bootstrap": _format(cluster_ci),
        "patch_bootstrap": _format(patch_ci),
    }

    out_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    logger.info("Saved %s", out_path)

    print(f"\nCluster bootstrap by source image ({len(per_image_counts)} images, "
          f"{args.n_resamples} resamples):")
    for name in ("precision", "recall", "dice", "iou"):
        pt, lo, hi = cluster_ci[name]
        print(f"  {name:>9s}: {pt:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")

    print(f"\nPatch bootstrap ({len(patch_counts)} patches, "
          f"{args.n_resamples} resamples):")
    for name in ("precision", "recall", "dice", "iou"):
        pt, lo, hi = patch_ci[name]
        print(f"  {name:>9s}: {pt:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
