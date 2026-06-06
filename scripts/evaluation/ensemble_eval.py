#!/usr/bin/env python
"""Ensemble + optional TTA evaluation (extension "improve-the-method" track).

Averages sigmoid probabilities across several trained checkpoints (e.g. the five
M5.6 winner seeds) and optionally over a 4-view self-inverse test-time
augmentation group {identity, hflip, vflip, rot180}. The ensemble threshold is
selected on the validation split (F1-optimal), then test metrics are reported at
that threshold — mirroring the single-model threshold-sweep methodology, so the
result is directly comparable to the locked baseline. No training; reuses
existing checkpoints, so it is a near-free precision/robustness lever.

Apples-to-apples note: this is a performance-track extension, NOT a paper
replication model. The baseline comparison remains the single-model M5.6 winner.

Usage (CSD3):
    CHECKPOINTS="results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth,...x5" \\
    python scripts/evaluation/ensemble_eval.py --checkpoints "$CHECKPOINTS" --tta \\
        --out results/classical/ensemble_t44.json
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
from src.models.loading import load_segmentation_model
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

_THRESHOLDS = np.round(np.arange(0.05, 0.955, 0.01), 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint paths.")
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--tta", action="store_true", help="Enable 4-view self-inverse TTA.")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold (skip val sweep; for a parity manifest with no val split).",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def _load(path: str, device: torch.device) -> tuple[torch.nn.Module, str]:
    return load_segmentation_model(path, device)


# 4-view self-inverse dihedral transforms (each is its own inverse), so applying
# the same op to the prediction maps it back to the original orientation.
def _tta_views(x: torch.Tensor) -> list[torch.Tensor]:
    return [x, torch.flip(x, dims=[-1]), torch.flip(x, dims=[-2]), torch.rot90(x, 2, dims=[-2, -1])]


def _untta(p: torch.Tensor, idx: int) -> torch.Tensor:
    if idx == 0:
        return p
    if idx == 1:
        return torch.flip(p, dims=[-1])
    if idx == 2:
        return torch.flip(p, dims=[-2])
    return torch.rot90(p, 2, dims=[-2, -1])


def _ensemble_probs(models: list[torch.nn.Module], images: torch.Tensor, tta: bool) -> torch.Tensor:
    acc = torch.zeros(images.shape[0], images.shape[2], images.shape[3], device=images.device)
    n = 0
    views = _tta_views(images) if tta else [images]
    for model in models:
        for idx, view in enumerate(views):
            prob = torch.sigmoid(model(view)).squeeze(1)
            acc += _untta(prob, idx if tta else 0)
            n += 1
    return acc / n


def _counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return tp, fp, fn


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"precision": p, "recall": r, "dice": dice, "iou": iou}


def _probs_and_gts(models, loader, device, tta):
    probs_list, gts_list = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            probs_list.append(_ensemble_probs(models, images, tta).cpu().numpy())
            gts_list.append(batch["mask"].numpy().squeeze(1) > 0.5)
    return np.concatenate(probs_list), np.concatenate(gts_list)


def main() -> None:
    args = parse_args()
    logger = get_logger("ensemble_eval")
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [s.strip() for s in args.checkpoints.split(",") if s.strip()]
    if not paths:
        raise SystemExit("--checkpoints must contain at least one checkpoint path")
    loaded = [_load(p, device) for p in paths]
    models = [model for model, _ in loaded]
    norms = [normalisation for _, normalisation in loaded]
    norm = norms[0]
    if any(item != norm for item in norms):
        raise SystemExit(f"ensemble checkpoints have mixed normalisation modes: {dict(zip(paths, norms))}")
    logger.info("Ensemble of %d checkpoints | tta=%s | normalisation=%s", len(models), args.tta, norm)

    def loader_for(split: str) -> DataLoader:
        ds = PatchDirectoryDataset(Path(args.patch_dir) / split, normalisation=norm)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")

    if args.threshold is not None:
        optimal_threshold = float(args.threshold)
        logger.info("Fixed threshold %.3f (no val sweep)", optimal_threshold)
    else:
        vprobs, vgts = _probs_and_gts(models, loader_for("val"), device, args.tta)
        best_f1, optimal_threshold = -1.0, 0.5
        for t in _THRESHOLDS:
            tp, fp, fn = _counts(vprobs >= t, vgts)
            m = _metrics(tp, fp, fn)
            f1 = (2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
                  if (m["precision"] + m["recall"]) else 0.0)
            if f1 > best_f1:
                best_f1, optimal_threshold = f1, float(t)
        logger.info("Val F1-optimal threshold=%.2f (val_F1=%.4f)", optimal_threshold, best_f1)

    tprobs, tgts = _probs_and_gts(models, loader_for("test"), device, args.tta)
    tp, fp, fn = _counts(tprobs >= optimal_threshold, tgts)
    tm = _metrics(tp, fp, fn)
    out = {
        "ensemble_checkpoints": paths,
        "tta": args.tta,
        "normalisation": norm,
        "checkpoint_normalisations": dict(zip(paths, norms)),
        "optimal_threshold": optimal_threshold,
        "test_precision": tm["precision"],
        "test_recall": tm["recall"],
        "test_dice": tm["dice"],
        "test_iou": tm["iou"],
        "generated": str(date.today()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, allow_nan=False))
    logger.info("TEST P=%.4f R=%.4f Dice=%.4f IoU=%.4f @t=%.2f",
                tm["precision"], tm["recall"], tm["dice"], tm["iou"], optimal_threshold)
    print(f"Ensemble({len(models)}) tta={args.tta}: test Dice={tm['dice']:.4f} "
          f"P={tm['precision']:.4f} R={tm['recall']:.4f} @t={optimal_threshold} -> {args.out}")


if __name__ == "__main__":
    main()
