#!/usr/bin/env python
"""Prediction-visualisation figures for the final reported checkpoint (M5.1).

Run against the final reported checkpoint — either the M5.2 re-Optuna retrain
if it beats the winning ablation on val_f1, or the winning ablation itself.
Decision rule lives in agents/PLAN.md M5.1; this script does not enforce it.

Produces four figure types under results/figures/predictions/:
1. Test-patch overlay grid: a compact deterministic subset with GT and prediction
   contours overlaid in one panel per patch.
2. FP/FN confusion gallery: most-confident false positives, hardest false
   negatives, cleanest true positives. Top-K per category.
3. Per-image FNR summary: skips the bar chart when every positive image is detected.
4. Full-image contour overview for one illustrative positive test image
   (raw image plus probability/binary/Hough contours).

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
from typing import Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import binary_erosion, label as ndi_label
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from matplotlib.patches import Rectangle
from PIL import Image
from torch.utils.data import DataLoader

from src.config.constants import PATCH_SIZE
from src.data.dataset import PatchDirectoryDataset
from src.data.transforms import normalise_tensor
from src.evaluation.segmentation import SegmentationCounts, compute_metrics_from_counts
from src.models.loading import load_segmentation_model
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
    p.add_argument("--n_random", type=int, default=6, help="Deprecated; overlay grid now uses a deterministic balanced positive subset.")
    p.add_argument("--n_worst", type=int, default=6, help="Deprecated; overlay grid now uses a deterministic balanced positive subset.")
    p.add_argument("--top_k", type=int, default=5, help="Top-K per FP/FN/TP gallery row.")
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


def _load_model(checkpoint: str, device: torch.device) -> tuple[torch.nn.Module, str]:
    return load_segmentation_model(checkpoint, device)


def _counts(pred: torch.Tensor, target: torch.Tensor) -> SegmentationCounts:
    not_p, not_t = ~pred, ~target
    return SegmentationCounts(
        true_positive=int(torch.logical_and(pred, target).sum()),
        false_positive=int(torch.logical_and(pred, not_t).sum()),
        true_negative=int(torch.logical_and(not_p, not_t).sum()),
        false_negative=int(torch.logical_and(not_p, target).sum()),
    )


def _collect_per_patch(
    model: torch.nn.Module,
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


def _mask_outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    outline = mask & ~eroded
    if thickness > 1:
        outline = cv2.dilate(outline.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=thickness - 1) > 0
    return outline


def _prediction_overlay(raw: np.ndarray, target: np.ndarray, prob: np.ndarray, threshold: float) -> np.ndarray:
    base = np.clip(raw.astype(np.float32) / 255.0, 0, 1)
    rgb = np.stack([base, base, base], axis=-1)
    pred = prob >= threshold
    # Thick GT band underneath, thinner prediction on top. Where they overlap the
    # magenta prediction sits inside a green halo, so a correct detection reads as
    # "magenta within green" instead of a single ambiguous colour; GT-only stays
    # green (missed) and prediction-only stays magenta (false positive).
    gt_band = _mask_outline(target.astype(bool), thickness=4)
    pred_line = _mask_outline(pred, thickness=2)
    rgb[gt_band] = (0.0, 0.85, 0.30)
    rgb[pred_line] = (1.0, 0.0, 0.82)
    return rgb


def _infer_patch_prob(
    img: np.ndarray,
    rec: dict,
    model: torch.nn.Module,
    device: torch.device,
    normalisation: str,
) -> np.ndarray:
    with torch.no_grad():
        t = TF.pil_to_tensor(Image.fromarray(img)).float().unsqueeze(0) / 255.0
        t = normalise_tensor(
            t.squeeze(0), normalisation,
            rec.get("image_mean"), rec.get("image_std"),
        ).unsqueeze(0).to(device)
        return torch.sigmoid(model(t)).squeeze().cpu().numpy()


# Fixed predeclared patch sets so both models (t44 and t7) render the SAME panels in the
# SAME order; each model still shows its own segmentation. Overlay set from the t44
# selection, gallery set from the t7 selection, including the author-picked
# diffraction-star examples (bright-star failures and stars correctly ignored).
OVERLAY_PATCHES = [   # 8 patches: first 4 low-Dice, last 4 high-Dice
    "ML1_20220727_185824_red.fits_full_2640_4752_image.png",
    "ML1_20220524_040528_red.fits_full_4224_5808_image.png",
    "ML1_20220629_171546_red.fits_full_5280_528_image.png",
    "ML1_20220629_171546_red.fits_full_2640_7392_image.png",
    "ML1_20220531_190959_red.fits_full_2112_2112_image.png",
    "ML1_20210430_173349_red.fits_full_5280_1584_image.png",
    "ML1_20210217_185657_red.fits_full_3168_4752_image.png",
    "ML1_20210217_185657_red.fits_full_3168_6336_image.png",
]
GALLERY_PATCHES = [   # 8 patches: first 4 false positives (empty GT), last 4 true positives
    "ML1_20221127_194620_red.fits_full_0_10032_image.png",
    "ML1_20220530_174559_red.fits_full_8976_8976_image.png",
    "ML1_20171023_002210_red.fits_full_6864_528_image.png",
    "ML1_20220629_171546_red.fits_full_10032_4224_image.png",
    "ML1_20221127_194620_red.fits_full_8448_1056_image.png",
    "ML1_20210217_185657_red.fits_full_3168_7920_image.png",
    "ML1_20210217_185657_red.fits_full_3168_6336_image.png",
    "ML1_20220525_200632_red.fits_full_5280_4224_image.png",
]


def _find_record(records: Sequence[dict], needle: str) -> dict | None:
    for rec in records:
        if needle in str(rec.get("patch_path", "")):
            return rec
    return None


def _figure_overlay_grid(
    records: list[dict],
    model: torch.nn.Module,
    device: torch.device,
    normalisation: str,
    threshold: float,
    out_path: Path,
) -> None:
    """Fixed 8-patch overlay grid; identical patches (and order) for both models."""
    selected = [r for r in (_find_record(records, needle) for needle in OVERLAY_PATCHES) if r is not None]
    n = len(selected)
    if n == 0:
        return
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 2.85 * nrows + 0.28))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, rec in zip(axes_arr, selected):
        img = _load_patch_image(rec["patch_path"])
        mask = _load_patch_image(rec["mask_path"])
        prob = _infer_patch_prob(img, rec, model, device, normalisation)
        ax.imshow(_prediction_overlay(img, mask, prob, threshold), interpolation="nearest")
        ax.set_title(f"Dice={rec['dice']:.2f}; mean p={rec['mean_prob']:.2f}", fontsize=7, pad=5)
        ax.axis("off")
    for ax in axes_arr[n:]:
        ax.axis("off")
    fig.suptitle(
        "Prediction overlay subset: green = GT, magenta = prediction (overlap = magenta inside green)\n"
        "Low-Dice patches (top row) and high-Dice patches (bottom row)",
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(left=0.012, right=0.995, bottom=0.015, top=0.86, hspace=0.26, wspace=0.035)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _figure_fp_fn_gallery(
    records: list[dict],
    top_k: int,
    model: torch.nn.Module,
    device: torch.device,
    normalisation: str,
    threshold: float,
    out_path: Path,
) -> None:
    """Fixed 8-patch gallery (identical patches and order for both models): FPs over TPs."""
    ncols = 4
    confident_fp = [r for r in (_find_record(records, needle) for needle in GALLERY_PATCHES[:ncols]) if r is not None]
    cleanest_tp = [r for r in (_find_record(records, needle) for needle in GALLERY_PATCHES[ncols:]) if r is not None]
    rows = [
        ("False positives\n(empty-GT patches)", confident_fp),
        ("True positives\n(high-Dice patches)", cleanest_tp),
    ]
    fig, axes = plt.subplots(2, ncols, figsize=(2.3 * ncols, 5.2))
    axes = np.atleast_2d(axes)
    for r, (label, items) in enumerate(rows):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(items):
                ax.axis("off")
                continue
            rec = items[c]
            img = _load_patch_image(rec["patch_path"])
            mask = _load_patch_image(rec["mask_path"])
            prob = _infer_patch_prob(img, rec, model, device, normalisation)
            ax.imshow(_prediction_overlay(img, mask, prob, threshold), interpolation="nearest")
            ax.set_title(f"Dice={rec['dice']:.2f}; mean p={rec['mean_prob']:.2f}", fontsize=7, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                ax.axis("off")
    fig.suptitle("FP / TP patch gallery: green = GT, magenta = prediction (overlap = magenta inside green)", fontsize=11, y=0.97)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.02, top=0.86, hspace=0.16, wspace=0.04)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _figure_per_image_fnr(hough_json: Path, out_path: Path, logger) -> bool:
    """Skip uninformative per-image bars when all positive images are detected."""
    data = json.loads(Path(hough_json).read_text())
    rows = [r for r in data["per_image"] if r["is_positive"]]
    if not rows:
        logger.warning("%s has no positive test images; skipping per-image FNR figure", hough_json)
        return False
    n = len(rows)
    pre = sum(bool(r["detected_pre_hough"]) for r in rows)
    post = sum(bool(r["detected_post_hough"]) for r in rows)
    if pre == n and post == n:
        logger.info(
            "Skipping per-image FNR bar chart for %s: all %d positive images detected pre- and post-Hough; report in text/table.",
            hough_json, n,
        )
        return False
    max_line_gap = data.get("max_line_gap")
    labels = [Path(r["source_image"]).stem.replace("_red.fits_full", "") for r in rows]
    pre_arr = np.asarray([1 if r["detected_pre_hough"] else 0 for r in rows])
    post_arr = np.asarray([1 if r["detected_post_hough"] else 0 for r in rows])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.3), 4))
    ax.bar(x - 0.2, pre_arr, width=0.4, label="pre-Hough")
    ax.bar(x + 0.2, post_arr, width=0.4, label="post-Hough")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Detected (1) / Missed (0)")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        f"Per-image detection (pre vs post-Hough). FNR pre={data.get('fnr_pre_hough')} "
        f"post={data.get('fnr_post_hough')} | max_line_gap={max_line_gap}",
        fontsize=10,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _resize_for_display(arr: np.ndarray, max_dim: int = 1500, nearest: bool = False) -> np.ndarray:
    import cv2
    h, w = arr.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return arr
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(arr.astype(np.float32), (int(round(w * scale)), int(round(h * scale))), interpolation=interpolation)


def _largest_component_bbox(mask: np.ndarray, margin: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    import cv2

    mask = np.asarray(mask, dtype=bool)
    h, w = shape
    if not mask.any():
        cy, cx = h // 2, w // 2
        return slice(max(0, cy - margin), min(h, cy + margin)), slice(max(0, cx - margin), min(w, cx + margin))
    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n_labels > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == int(np.argmax(sizes))
    yy, xx = np.nonzero(mask)
    y0 = max(0, int(yy.min()) - margin)
    y1 = min(h, int(yy.max()) + margin + 1)
    x0 = max(0, int(xx.min()) - margin)
    x1 = min(w, int(xx.max()) + margin + 1)
    return slice(y0, y1), slice(x0, x1)


def _contour_if_any(ax, mask: np.ndarray, *, color: str, linewidth: float, linestyle: str = "solid", alpha: float = 0.9) -> None:
    if np.asarray(mask).any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth, linestyles=linestyle, alpha=alpha)



def _line_inset_bbox(target_canvas: np.ndarray, binary: np.ndarray, hough: np.ndarray, *, half_h: int = 105, half_w: int = 230) -> tuple[slice, slice]:
    shape = target_canvas.shape
    mask = target_canvas.astype(bool) & (binary.astype(bool) | hough.astype(bool))
    if not mask.any():
        mask = target_canvas.astype(bool)
    if not mask.any():
        mask = binary.astype(bool) | hough.astype(bool)
    if not mask.any():
        return _largest_component_bbox(mask, margin=max(half_h, half_w), shape=shape)
    labels, n = ndi_label(mask)
    if n:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == int(counts.argmax())
    yy, xx = np.nonzero(mask)
    # Landscape window centred on the middle of the streak.
    cy = int(np.median(yy))
    cx = int(np.median(xx))
    y0 = max(0, cy - half_h)
    y1 = min(shape[0], cy + half_h)
    x0 = max(0, cx - half_w)
    x1 = min(shape[1], cx + half_w)
    return slice(y0, y1), slice(x0, x1)

def _figure_full_image_heatmap(
    records: list[dict],
    model: torch.nn.Module,
    device: torch.device,
    normalisation: str,
    patch_dir: Path,
    threshold: float,
    hough_json: Path | None,
    out_path: Path,
) -> None:
    """Full-frame contour overview plus deterministic inset zoom."""
    from src.classical.hough_runner import _apply_hough

    pos = [r for r in records if r["is_positive"]]
    if not pos:
        return
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
    target_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)

    has_full = {"image_mean", "image_std"}.issubset(group.columns)
    rows = list(group.itertuples(index=False))
    with torch.no_grad():
        for row in rows:
            img = _load_patch_image(row.patch_path)
            mask = _load_patch_image(row.mask_path) > 0
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
            target_canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE] |= mask

    with Image.open(chosen_src) as img:
        raw = np.asarray(img.convert("L"), dtype=np.uint8)[:canvas_h, :canvas_w]
    hough_params = {
        "hough_input_threshold": 0.10,
        "hough_threshold": 50,
        "min_line_length": 100,
        "max_line_gap": 250,
        "line_thickness": 3,
    }
    if hough_json is not None and Path(hough_json).exists():
        hdata = json.loads(Path(hough_json).read_text())
        for key in ("hough_input_threshold", "hough_threshold", "min_line_length", "max_line_gap", "line_thickness"):
            if hdata.get(key) is not None:
                hough_params[key] = hdata[key]
    binary = prob_canvas >= threshold
    hough_input = (prob_canvas >= float(hough_params["hough_input_threshold"])).astype(np.uint8) * 255
    hough = _apply_hough(
        hough_input,
        hough_threshold=int(hough_params["hough_threshold"]),
        min_line_length=int(hough_params["min_line_length"]),
        max_line_gap=int(hough_params["max_line_gap"]),
        line_thickness=int(hough_params["line_thickness"]),
    ) > 0

    raw_d = _resize_for_display(raw, nearest=False)
    target_d = _resize_for_display(target_canvas.astype(np.uint8), nearest=True) > 0
    binary_d = _resize_for_display(binary.astype(np.uint8), nearest=True) > 0
    hough_d = _resize_for_display(hough.astype(np.uint8), nearest=True) > 0

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(raw_d, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    _contour_if_any(ax, target_d, color="#009E73", linewidth=0.55, alpha=0.88)
    _contour_if_any(ax, binary_d, color="#ff2f92", linewidth=0.72, alpha=0.95)
    _contour_if_any(ax, hough_d, color="#00c8ff", linewidth=0.58, linestyle="dashed", alpha=0.92)
    ax.set_title(f"Full-image contour overview: {Path(chosen_src).stem}", fontsize=10)
    ax.axis("off")

    zoom_y, zoom_x = _line_inset_bbox(target_canvas, binary, hough, half_h=105, half_w=230)
    sy = raw_d.shape[0] / raw.shape[0]
    sx = raw_d.shape[1] / raw.shape[1]
    ax.add_patch(Rectangle((zoom_x.start * sx, zoom_y.start * sy), (zoom_x.stop - zoom_x.start) * sx,
                           (zoom_y.stop - zoom_y.start) * sy, fill=False, edgecolor="white", lw=0.9, alpha=0.9))
    inset = ax.inset_axes([0.32, 0.05, 0.66, 0.30])
    inset.imshow(raw[zoom_y, zoom_x], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    _contour_if_any(inset, target_canvas[zoom_y, zoom_x], color="#009E73", linewidth=0.75, alpha=0.92)
    _contour_if_any(inset, binary[zoom_y, zoom_x], color="#ff2f92", linewidth=0.9, alpha=0.97)
    _contour_if_any(inset, hough[zoom_y, zoom_x], color="#00c8ff", linewidth=0.75, linestyle="dashed", alpha=0.94)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.9)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
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

    _figure_overlay_grid(
        records, model, device, normalisation, args.threshold,
        out_dir / f"overlay_grid_{tag}.png",
    )
    logger.info("Wrote overlay grid (%d patches)", len(OVERLAY_PATCHES))

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
        Path(args.patch_dir), args.threshold, Path(args.hough_json) if args.hough_json else None,
        out_dir / f"full_image_contour_{tag}.png",
    )
    logger.info("Wrote full-image contour overview")

    print(f"Figures written under {out_dir}/ with tag '{tag}'")


if __name__ == "__main__":
    main()
