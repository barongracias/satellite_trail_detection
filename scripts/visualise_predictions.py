#!/usr/bin/env python
"""Prediction-visualisation figures for the final reported checkpoint (M5.1).

Run against the final reported checkpoint — either the M5.2 re-Optuna retrain
if it beats the winning ablation on val_f1, or the winning ablation itself.
Decision rule lives in agents/PLAN.md M5.1; this script does not enforce it.

Produces four figure types under results/figures/predictions/:
1. Test-patch overlay grid: image / GT mask / predicted mask. N random plus
   N worst-Dice patches.
2. FP/FN confusion gallery: most-confident false positives, hardest false
   negatives, cleanest true positives. Top-K per category.
3. Per-image FNR bar chart pre- and post-Hough. Reads detection flags from
   the matching hough_postprocess_<tag>.json.
4. Full-image probability heatmap for one illustrative positive test image
   (reconstructs the prob canvas from sampled patches, same logic as
   hough_postprocess.py).

Usage:
    python scripts/visualise_predictions.py \\
        --checkpoint results/checkpoints/unet_<tag>_best.pth \\
        --threshold 0.72 \\
        --hough_json results/classical/hough_postprocess_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader

from src.config.constants import PATCH_SIZE
from src.data.dataset import PatchDirectoryDataset
from src.data.transforms import normalise_tensor
from src.evaluation.segmentation import SegmentationCounts, compute_metrics_from_counts
from src.models.unet import UNet
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

Image.MAX_IMAGE_PIXELS = None

_PATCH_SIZE = PATCH_SIZE
_YX_RE = re.compile(r"_(\d+)_(\d+)_image$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument(
        "--hough_json",
        default=None,
        help="hough_postprocess_<tag>.json. Required for the per-image FNR "
             "bar chart; omit to skip that figure.",
    )
    p.add_argument(
        "--out_dir",
        default="results/figures/predictions",
        help="Output directory. Default: results/figures/predictions/",
    )
    p.add_argument("--n_random", type=int, default=8, help="Random patches in overlay grid.")
    p.add_argument("--n_worst", type=int, default=8, help="Worst-Dice patches in overlay grid.")
    p.add_argument("--top_k", type=int, default=6, help="Top-K per FP/FN/TP gallery row.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--tag",
        default=None,
        help="Suffix for output filenames. Defaults to checkpoint stem with "
             "trailing '_best' stripped.",
    )
    return p.parse_args()


def _resolve_tag(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    stem = Path(args.checkpoint).stem
    return stem[:-5] if stem.endswith("_best") else stem


def _load_model(checkpoint: str, device: torch.device) -> tuple[UNet, str]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = UNet(base_channels=cfg.get("base_channels", 8)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg.get("normalisation", "fixed")


def _counts(pred: torch.Tensor, target: torch.Tensor) -> SegmentationCounts:
    not_p, not_t = ~pred, ~target
    return SegmentationCounts(
        true_positive=int(torch.logical_and(pred, target).sum()),
        false_positive=int(torch.logical_and(pred, not_t).sum()),
        true_negative=int(torch.logical_and(not_p, not_t).sum()),
        false_negative=int(torch.logical_and(not_p, target).sum()),
    )


def _collect_per_patch(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[list[dict], np.ndarray]:
    """Return one record per test patch plus a flat array of mean probabilities.

    Each record contains patch_path, mask_path, source_image, dice, mean_prob,
    is_positive (target has any positive pixel), pred_positive_fraction, and
    full_image stats (image_mean/image_std if the manifest has them, else
    None). The stats are required by overlay_grid / fp_fn_gallery so their
    on-the-fly inference matches the recorded predictions when
    normalisation=full_image (without them, normalise_tensor silently falls
    back to per-patch z-score).
    """
    records: list[dict] = []
    mean_probs: list[float] = []
    rows = loader.dataset.records.to_dict("records")  # type: ignore[attr-defined]
    has_full = {"image_mean", "image_std"}.issubset(loader.dataset.records.columns)  # type: ignore[attr-defined]
    idx = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            masks = batch["mask"].to(device, dtype=torch.float32, non_blocking=True)
            probs = torch.sigmoid(model(images))
            preds = probs >= threshold
            targets = masks > 0.5
            for i in range(preds.shape[0]):
                row = rows[idx]
                metrics = compute_metrics_from_counts(_counts(preds[i], targets[i]))
                records.append({
                    "patch_path": row["patch_path"],
                    "mask_path": row["mask_path"],
                    "source_image": row["source_image"],
                    "dice": metrics.dice,
                    "mean_prob": float(probs[i].mean()),
                    "is_positive": bool(targets[i].any()),
                    "pred_positive_fraction": float(preds[i].float().mean()),
                    "image_mean": row.get("image_mean") if has_full else None,
                    "image_std": row.get("image_std") if has_full else None,
                })
                mean_probs.append(float(probs[i].mean()))
                idx += 1
    return records, np.asarray(mean_probs)


def _load_patch_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def _overlay_panel(ax, img: np.ndarray, mask: np.ndarray | None, title: str) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    if mask is not None:
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.5)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _figure_overlay_grid(
    selected: list[dict],
    model: UNet,
    device: torch.device,
    normalisation: str,
    threshold: float,
    out_path: Path,
) -> None:
    """Image / GT mask / predicted mask, three columns per patch row."""
    n = len(selected)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for r, rec in enumerate(selected):
        img = _load_patch_image(rec["patch_path"])
        mask = _load_patch_image(rec["mask_path"])
        with torch.no_grad():
            t = TF.pil_to_tensor(Image.fromarray(img)).float().unsqueeze(0) / 255.0
            t = normalise_tensor(
                t.squeeze(0), normalisation,
                rec.get("image_mean"), rec.get("image_std"),
            ).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
        pred_mask = (prob >= threshold).astype(np.uint8) * 255
        _overlay_panel(axes[r, 0], img, None, Path(rec["patch_path"]).name)
        _overlay_panel(axes[r, 1], img, mask, f"GT (Dice={rec['dice']:.3f})")
        _overlay_panel(axes[r, 2], img, pred_mask, f"Pred (mean p={rec['mean_prob']:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _figure_fp_fn_gallery(
    records: list[dict],
    top_k: int,
    model: UNet,
    device: torch.device,
    normalisation: str,
    threshold: float,
    out_path: Path,
) -> None:
    """Most-confident FPs (high mean prob on empty patches), hardest FNs
    (low mean prob on positive patches), cleanest TPs (high Dice)."""
    pos = [r for r in records if r["is_positive"]]
    neg = [r for r in records if not r["is_positive"]]
    confident_fp = sorted(neg, key=lambda r: -r["mean_prob"])[:top_k]
    hardest_fn = sorted(pos, key=lambda r: r["dice"])[:top_k]
    cleanest_tp = sorted(pos, key=lambda r: -r["dice"])[:top_k]

    rows = [("Confident FPs", confident_fp),
            ("Hardest FNs", hardest_fn),
            ("Cleanest TPs", cleanest_tp)]
    fig, axes = plt.subplots(3, top_k, figsize=(2.5 * top_k, 8))
    if top_k == 1:
        axes = axes[:, np.newaxis]
    for r, (label, items) in enumerate(rows):
        for c in range(top_k):
            ax = axes[r, c]
            if c >= len(items):
                ax.axis("off")
                continue
            rec = items[c]
            img = _load_patch_image(rec["patch_path"])
            mask = _load_patch_image(rec["mask_path"])
            with torch.no_grad():
                t = TF.pil_to_tensor(Image.fromarray(img)).float().unsqueeze(0) / 255.0
                t = normalise_tensor(
                    t.squeeze(0), normalisation,
                    rec.get("image_mean"), rec.get("image_std"),
                ).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
            pred_mask = (prob >= threshold).astype(np.uint8) * 255
            _overlay_panel(ax, img, np.maximum(mask, pred_mask // 2), "")
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_title(f"D={rec['dice']:.2f} mp={rec['mean_prob']:.2f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_HOUGH_PRE_FIX_MAX_LINE_GAP = 50


def _figure_per_image_fnr(hough_json: Path, out_path: Path, logger) -> bool:
    """Bar chart of pre/post-Hough detection per positive test image.

    Returns True iff a figure was written. Surfaces the Hough max_line_gap
    used to produce the JSON in the title so dissertation figures can be
    visually distinguished from runs produced with the pre-2026-05-21
    default of 50 (paper-faithful default is now 250).
    """
    data = json.loads(Path(hough_json).read_text())
    rows = [r for r in data["per_image"] if r["is_positive"]]
    if not rows:
        logger.warning(
            "%s has no positive test images; skipping per-image FNR figure", hough_json,
        )
        return False
    max_line_gap = data.get("max_line_gap")
    if max_line_gap == _HOUGH_PRE_FIX_MAX_LINE_GAP:
        logger.warning(
            "%s was produced with max_line_gap=%d (pre-2026-05-21 default). "
            "Paper-faithful default is now 250. Re-run Hough postprocess "
            "before using this figure in the dissertation.",
            hough_json, max_line_gap,
        )
    labels = [Path(r["source_image"]).stem.replace("_red.fits_full", "") for r in rows]
    pre = np.asarray([1 if r["detected_pre_hough"] else 0 for r in rows])
    post = np.asarray([1 if r["detected_post_hough"] else 0 for r in rows])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.3), 4))
    ax.bar(x - 0.2, pre, width=0.4, label="pre-Hough")
    ax.bar(x + 0.2, post, width=0.4, label="post-Hough")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Detected (1) / Missed (0)")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        f"Per-image detection (pre vs post-Hough). "
        f"FNR pre={data.get('fnr_pre_hough')} post={data.get('fnr_post_hough')} "
        f"| max_line_gap={max_line_gap}",
        fontsize=10,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _figure_full_image_heatmap(
    records: list[dict],
    model: UNet,
    device: torch.device,
    normalisation: str,
    patch_dir: Path,
    out_path: Path,
) -> None:
    """Reconstruct prob canvas for one positive test image with the most
    positive patches; save as a heatmap. Mirrors hough_postprocess.py canvas
    reconstruction."""
    pos = [r for r in records if r["is_positive"]]
    if not pos:
        return
    # pick the source image with the most positive patches
    counts = pd.Series([r["source_image"] for r in pos]).value_counts()
    chosen_src = counts.index[0]
    manifest = pd.read_csv(patch_dir / "manifest.csv")
    group = manifest[(manifest["split"] == "test") & (manifest["source_image"] == chosen_src)]
    yx = [_YX_RE.search(Path(p).stem) for p in group["patch_path"]]
    coords = [(int(m.group(1)), int(m.group(2))) for m in yx if m is not None]
    if not coords:
        return
    canvas_h = max(y for y, _ in coords) + _PATCH_SIZE
    canvas_w = max(x for _, x in coords) + _PATCH_SIZE
    prob_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    has_full = {"image_mean", "image_std"}.issubset(group.columns)
    rows = list(group.itertuples(index=False))
    with torch.no_grad():
        for row in rows:
            img = _load_patch_image(row.patch_path)
            t = TF.pil_to_tensor(Image.fromarray(img)).float().unsqueeze(0) / 255.0
            mean = getattr(row, "image_mean", None) if has_full else None
            std = getattr(row, "image_std", None) if has_full else None
            t = normalise_tensor(t.squeeze(0), normalisation, mean, std).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
            m = _YX_RE.search(Path(row.patch_path).stem)
            if m is None:
                continue
            y, x = int(m.group(1)), int(m.group(2))
            prob_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE] = np.maximum(
                prob_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE], prob
            )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(prob_canvas, cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"Full-image probability heatmap: {Path(chosen_src).stem}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logger = get_logger("visualise_predictions")
    seed_everything()
    tag = _resolve_tag(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalisation = _load_model(args.checkpoint, device)
    logger.info("Loaded %s (normalisation=%s, threshold=%.3f, device=%s)",
                args.checkpoint, normalisation, args.threshold, device)

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

    records, _ = _collect_per_patch(model, loader, device, args.threshold)

    rng = np.random.default_rng(2804)
    pos = [r for r in records if r["is_positive"]]
    worst = sorted(pos, key=lambda r: r["dice"])[: args.n_worst]
    random_pick = (
        [records[int(i)] for i in rng.choice(len(records), size=args.n_random, replace=False)]
        if args.n_random > 0 else []
    )
    overlay_set = worst + random_pick
    _figure_overlay_grid(
        overlay_set, model, device, normalisation, args.threshold,
        out_dir / f"overlay_grid_{tag}.png",
    )
    logger.info("Wrote overlay grid (%d patches)", len(overlay_set))

    _figure_fp_fn_gallery(
        records, args.top_k, model, device, normalisation, args.threshold,
        out_dir / f"fp_fn_gallery_{tag}.png",
    )
    logger.info("Wrote FP/FN gallery (top_k=%d)", args.top_k)

    if args.hough_json:
        wrote = _figure_per_image_fnr(
            Path(args.hough_json), out_dir / f"per_image_fnr_{tag}.png", logger,
        )
        if wrote:
            logger.info("Wrote per-image FNR bar chart")
    else:
        logger.info("No --hough_json supplied; skipping per-image FNR figure")

    _figure_full_image_heatmap(
        records, model, device, normalisation,
        Path(args.patch_dir), out_dir / f"full_image_heatmap_{tag}.png",
    )
    logger.info("Wrote full-image probability heatmap")

    print(f"Figures written under {out_dir}/ with tag '{tag}'")


if __name__ == "__main__":
    main()
