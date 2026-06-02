#!/usr/bin/env python
"""Threshold sweep on the trained U-Net baseline.

Runs inference once on the val split, sweeps thresholds 0.05→0.95, picks the
F1-optimal threshold, evaluates on the test split, saves a PR curve figure and
a results JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PatchDirectoryDataset
from src.evaluation.segmentation import (
    SegmentationCounts,
    compute_metrics_from_counts,
    evaluate_model_on_dataloader,
)
from src.models.loading import load_segmentation_model
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth")
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--tag",
        default=None,
        help="Suffix for output filenames. Defaults to the checkpoint stem with "
             "any trailing '_best' stripped, so each ablation gets its own "
             "threshold_sweep_<tag>.json and threshold_sweep_<tag>_pr_curve.png.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, skip the val PR sweep and evaluate the test split at this "
             "threshold directly. Required when the manifest has no val split "
             "(e.g., parity test sets built with --splits test). Pass the "
             "F1-optimal threshold from a matching val-sweep JSON.",
    )
    p.add_argument(
        "--val-only",
        action="store_true",
        help="Run only the validation PR sweep and do not evaluate or write test "
             "metrics. Used for diagnostic sensitivity runs where test metrics "
             "are intentionally inadmissible.",
    )
    return p.parse_args()


def _resolve_tag(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    stem = Path(args.checkpoint).stem
    return stem[:-5] if stem.endswith("_best") else stem


def streaming_sweep_thresholds(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: np.ndarray,
) -> list[dict]:
    """Sweep `thresholds` over a loader without materialising the full prob array.

    Per-threshold TP/FP/FN/TN are accumulated as int64 tensors on `device` across
    batches; a single .tolist() syncs to Python at the end. Memory cost is
    O(num_thresholds × 4 × 8 bytes) regardless of dataset size.
    """
    num_t = len(thresholds)
    tp = torch.zeros(num_t, dtype=torch.int64, device=device)
    fp = torch.zeros(num_t, dtype=torch.int64, device=device)
    tn = torch.zeros(num_t, dtype=torch.int64, device=device)
    fn = torch.zeros(num_t, dtype=torch.int64, device=device)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            masks = batch["mask"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            probs = torch.sigmoid(model(images))
            target_bool = masks > 0.5
            not_target_bool = ~target_bool
            for ti, t in enumerate(thresholds):
                pred_bool = probs >= float(t)
                not_pred_bool = ~pred_bool
                tp[ti] += torch.logical_and(pred_bool, target_bool).sum(dtype=torch.int64)
                fp[ti] += torch.logical_and(pred_bool, not_target_bool).sum(dtype=torch.int64)
                fn[ti] += torch.logical_and(not_pred_bool, target_bool).sum(dtype=torch.int64)
                tn[ti] += torch.logical_and(not_pred_bool, not_target_bool).sum(dtype=torch.int64)

    tp_list = tp.tolist()
    fp_list = fp.tolist()
    fn_list = fn.tolist()
    tn_list = tn.tolist()

    rows: list[dict] = []
    for ti, t in enumerate(thresholds):
        counts = SegmentationCounts(
            true_positive=tp_list[ti],
            false_positive=fp_list[ti],
            true_negative=tn_list[ti],
            false_negative=fn_list[ti],
        )
        m = compute_metrics_from_counts(counts)
        f1 = (
            2.0 * m.precision * m.recall / (m.precision + m.recall)
            if (m.precision + m.recall) > 0
            else 0.0
        )
        rows.append({
            "threshold": float(t),
            "precision": m.precision,
            "recall": m.recall,
            "f1": f1,
        })
    return rows


def evaluate_at_threshold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    """Evaluate the model on `loader` at a single threshold (tensor-native)."""
    result = evaluate_model_on_dataloader(
        model=model,
        dataloader=loader,
        device=device,
        threshold=threshold,
    )
    return {
        "precision": result.metrics.precision,
        "recall": result.metrics.recall,
        "dice": result.metrics.dice,
        "iou": result.metrics.iou,
    }


def main() -> None:
    args = parse_args()
    if args.val_only and args.threshold is not None:
        raise SystemExit("--val-only cannot be combined with --threshold")

    tag = _resolve_tag(args)
    logger = get_logger("threshold_sweep")
    seed_everything()
    logger.info("Output tag: %s", tag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, normalisation = load_segmentation_model(args.checkpoint, device)
    logger.info("Loaded checkpoint: %s (normalisation=%s)", args.checkpoint, normalisation)

    def make_loader(split: str) -> DataLoader:
        ds = PatchDirectoryDataset(Path(args.patch_dir) / split, normalisation=normalisation)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    if args.threshold is not None:
        # Fixed-threshold mode: skip val sweep, evaluate test only. Used when
        # the manifest has no val split (e.g., a parity test set built by
        # passing --splits test to build_patch_dataset.py). PR curve and val
        # metrics are unavailable in this mode.
        optimal_threshold = float(args.threshold)
        sweep = []
        best = {"threshold": optimal_threshold, "f1": None,
                "precision": None, "recall": None}
        logger.info(
            "Skipping val sweep; evaluating test at supplied threshold %.4f",
            optimal_threshold,
        )
    else:
        val_loader = make_loader("val")
        logger.info("Val set: %d patches", len(val_loader.dataset))  # type: ignore[arg-type]

        thresholds = np.round(np.arange(0.05, 0.955, 0.01), 2)
        sweep = streaming_sweep_thresholds(model, val_loader, device, thresholds)

        best = max(sweep, key=lambda r: r["f1"])
        optimal_threshold = best["threshold"]
        logger.info(
            "Optimal threshold %.2f: val_F1=%.4f  P=%.4f  R=%.4f",
            optimal_threshold, best["f1"], best["precision"], best["recall"],
        )

    test_m: dict[str, float] | None = None
    if args.val_only:
        logger.info("Val-only mode: skipping test evaluation and test metric output.")
    else:
        test_loader = make_loader("test")
        logger.info("Test set: %d patches", len(test_loader.dataset))  # type: ignore[arg-type]
        test_m = evaluate_at_threshold(model, test_loader, device, optimal_threshold)
        logger.info(
            "Test @ t=%.2f: P=%.4f  R=%.4f  Dice=%.4f  IoU=%.4f",
            optimal_threshold, test_m["precision"], test_m["recall"], test_m["dice"], test_m["iou"],
        )

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not sweep:
        logger.info("No val sweep performed (--threshold supplied); skipping PR-curve figure.")
    else:
        precisions = [r["precision"] for r in sweep]
        recalls = [r["recall"] for r in sweep]
        f1s = [r["f1"] for r in sweep]
        thresh_vals = [r["threshold"] for r in sweep]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(recalls, precisions, linewidth=1.5)
        axes[0].scatter(
            [best["recall"]], [best["precision"]],
            color="red", zorder=5,
            label=f"F1-optimal (t={optimal_threshold:.2f})",
        )
        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].set_title("Precision–Recall curve (val)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(thresh_vals, f1s, label="F1", linewidth=1.5)
        axes[1].plot(thresh_vals, precisions, "--", label="Precision", linewidth=1.0)
        axes[1].plot(thresh_vals, recalls, "--", label="Recall", linewidth=1.0)
        axes[1].axvline(
            optimal_threshold, color="k", linestyle=":", alpha=0.7,
            label=f"t*={optimal_threshold:.2f}",
        )
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Metrics vs threshold (val)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = figures_dir / f"threshold_sweep_{tag}_pr_curve.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure to %s", fig_path)

    out: dict = {
        "tag": tag,
        "checkpoint": str(args.checkpoint),
        "optimal_threshold": optimal_threshold,
        "val_f1": best["f1"],
        "val_precision": best["precision"],
        "val_recall": best["recall"],
        "pr_curve": sweep,
    }
    if test_m is not None:
        out.update({
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_dice": test_m["dice"],
            "test_iou": test_m["iou"],
        })
    out_path = Path("results/classical") / f"threshold_sweep_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    logger.info("Saved results to %s", out_path)

    print(f"\nOptimal threshold: {optimal_threshold:.2f}")
    if args.threshold is None:
        print(
            f"Val   F1={best['f1']:.4f}  "
            f"P={best['precision']:.4f}  R={best['recall']:.4f}"
        )
    else:
        print(f"Val   <skipped — threshold supplied: {args.threshold:.4f}>")
    if test_m is None:
        print("Test  <skipped - val-only mode>")
    else:
        print(
            f"Test  P={test_m['precision']:.4f}  R={test_m['recall']:.4f}"
            f"  Dice={test_m['dice']:.4f}  IoU={test_m['iou']:.4f}"
        )


if __name__ == "__main__":
    main()
