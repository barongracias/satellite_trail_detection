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
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import PatchDirectoryDataset
from src.models.unet import UNet
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments/unet_baseline.yaml")
    p.add_argument("--checkpoint", default="results/checkpoints/unet_baseline_best.pth")
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def collect_probs_and_targets(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            masks = batch["mask"].to(device=device, dtype=torch.float32)
            probs = torch.sigmoid(model(images))
            all_probs.append(probs.cpu().numpy().reshape(-1))
            all_targets.append(masks.cpu().numpy().reshape(-1))
    return np.concatenate(all_probs), np.concatenate(all_targets)


def sweep_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict]:
    target_bool = targets >= 0.5
    rows = []
    for t in thresholds:
        pred_bool = probs >= t
        tp = int(np.logical_and(pred_bool, target_bool).sum())
        fp = int(np.logical_and(pred_bool, ~target_bool).sum())
        fn = int(np.logical_and(~pred_bool, target_bool).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "f1": f1})
    return rows


def eval_at_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    target_bool = targets >= 0.5
    pred_bool = probs >= threshold
    tp = int(np.logical_and(pred_bool, target_bool).sum())
    fp = int(np.logical_and(pred_bool, ~target_bool).sum())
    fn = int(np.logical_and(~pred_bool, target_bool).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "dice": dice, "iou": iou}


def main() -> None:
    args = parse_args()
    logger = get_logger("threshold_sweep")
    seed_everything()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = UNet(
        base_channels=cfg["base_channels"],
        dropout_rate=cfg["dropout_rate"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    def make_loader(split: str) -> DataLoader:
        ds = PatchDirectoryDataset(Path(args.patch_dir) / split)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    val_loader = make_loader("val")
    logger.info("Val set: %d patches", len(val_loader.dataset))  # type: ignore[arg-type]
    val_probs, val_targets = collect_probs_and_targets(model, val_loader, device)

    thresholds = np.round(np.arange(0.05, 0.955, 0.01), 2)
    sweep = sweep_thresholds(val_probs, val_targets, thresholds)

    best = max(sweep, key=lambda r: r["f1"])
    optimal_threshold = best["threshold"]
    logger.info(
        "Optimal threshold %.2f: val_F1=%.4f  P=%.4f  R=%.4f",
        optimal_threshold, best["f1"], best["precision"], best["recall"],
    )

    test_loader = make_loader("test")
    logger.info("Test set: %d patches", len(test_loader.dataset))  # type: ignore[arg-type]
    test_probs, test_targets = collect_probs_and_targets(model, test_loader, device)
    test_m = eval_at_threshold(test_probs, test_targets, optimal_threshold)
    logger.info(
        "Test @ t=%.2f: P=%.4f  R=%.4f  Dice=%.4f  IoU=%.4f",
        optimal_threshold, test_m["precision"], test_m["recall"], test_m["dice"], test_m["iou"],
    )

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

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
    fig_path = figures_dir / "threshold_sweep_pr_curve.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", fig_path)

    out: dict = {
        "optimal_threshold": optimal_threshold,
        "val_f1": best["f1"],
        "val_precision": best["precision"],
        "val_recall": best["recall"],
        "test_precision": test_m["precision"],
        "test_recall": test_m["recall"],
        "test_dice": test_m["dice"],
        "test_iou": test_m["iou"],
        "pr_curve": sweep,
    }
    out_path = Path("results/classical/threshold_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved results to %s", out_path)

    print(f"\nOptimal threshold: {optimal_threshold:.2f}")
    print(f"Val   F1={best['f1']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}")
    print(
        f"Test  P={test_m['precision']:.4f}  R={test_m['recall']:.4f}"
        f"  Dice={test_m['dice']:.4f}  IoU={test_m['iou']:.4f}"
    )


if __name__ == "__main__":
    main()
