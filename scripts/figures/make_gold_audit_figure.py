#!/usr/bin/env python
"""Supplementary qualitative figure for the blinded gold audit (thesis section 6.2).

Visualises the gold-audit result (results/classical/gold_audit_eval.json) on two
illustrative crops: the original mask, the blinded single-author REFERENCE
re-annotation, and the locked MODEL prediction (>= 0.45) overlaid in three
distinct colours on the audit crop image, showing the close three-way agreement.

INFERENCE-ONLY on the locked model: no retraining, re-tuning, reselection, or
threshold change. Reuses gold_audit_eval.py's exact loaders (_load_normalised_patch
+ _infer_batch at threshold 0.45, load_raw_mask, load_annotation_mask) so the
rendered masks match the scored JSON.

The two crops (c027, c060) are author-chosen illustrative examples, not a blind
selection: c027 is an interior "trail" crop (a normal streak) and c060 is an
"fp"-stratum control crop the blinded annotator judged to be a real trail. The
reference is a blinded single-author self-reannotation, NOT an independent gold
standard. Saves a single PDF (no SVG).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from PIL import Image

from scripts.evaluation.gold_audit_eval import CROP_SIZE, load_annotation_mask, width_median
from scripts.evaluation.hough_postprocess import _infer_batch, _load_normalised_patch
from scripts.figures._locked_winner_canvases import load_raw_mask
from src.models.loading import load_segmentation_model
from src.utils.seed import seed_everything

Image.MAX_IMAGE_PIXELS = None

CHECKPOINT = "results/checkpoints/model-best.pth"
THRESHOLD = 0.45
SEALED = "data/gold/sealed_crop_manifest.json"
VERDICTS = "data/gold/verdicts.csv"
ANNOT_DIR = Path("data/gold/gold_masks")
CROP_DIR = Path("data/gold/audit_crops")
PATCH_DIR = "data/patches"
OUT = "results/figures/supp_3_gold_audit_overlays.pdf"

EXAMPLES = ["c027.png", "c060.png"]
COL_ORIG = "#D55E00"   # vermillion
COL_REF = "#009E73"    # bluish green
COL_MODEL = "#0072B2"  # blue


def _load_verdicts(path: str) -> dict[str, str]:
    with open(path, newline="") as handle:
        return {row["crop_name"].strip(): row["verdict"].strip() for row in csv.DictReader(handle)}


def _overlay(mask: np.ndarray, colour: str, alpha: float = 0.55) -> np.ndarray:
    """RGBA image: ``colour`` at ``alpha`` where mask is set, transparent elsewhere."""
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask] = (*to_rgb(colour), alpha)
    return rgba


def _wfmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.0f}"


def main() -> None:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalisation = load_segmentation_model(CHECKPOINT, device)

    records = {r["crop_name"]: r for r in json.loads(Path(SEALED).read_text())["crops"]}
    verdicts = _load_verdicts(VERDICTS)

    patch_manifest = pd.read_csv(Path(PATCH_DIR) / "manifest.csv")
    stats: dict[str, tuple[float, float]] = {}
    if {"image_mean", "image_std"}.issubset(patch_manifest.columns):
        for source, group in patch_manifest.groupby("source_image"):
            stats[str(source)] = (float(group["image_mean"].iloc[0]), float(group["image_std"].iloc[0]))

    mask_cache: dict[str, np.ndarray] = {}

    def masks_for(name: str):
        rec = records[name]
        source = rec["source_image"]
        if source not in mask_cache:
            mask_cache[source] = load_raw_mask(source)
        y0, x0 = int(rec["y0"]), int(rec["x0"])
        original = mask_cache[source][y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]
        reference = load_annotation_mask(ANNOT_DIR / name)
        mean, std = stats.get(source, (None, None))
        patch = _load_normalised_patch(str(CROP_DIR / name), normalisation, mean, std)
        prediction = _infer_batch([patch], model, device)[0] >= THRESHOLD
        with Image.open(CROP_DIR / name) as im:
            crop = np.asarray(im.convert("L"), dtype=np.uint8)
        return rec, crop, original, reference, prediction

    # --- render ----------------------------------------------------------------
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8})
    n = len(EXAMPLES)
    fig, axes = plt.subplots(n, 2, figsize=(5.6, 2.95 * n))
    axes = np.atleast_2d(axes)

    def label_box(ax, text, loc_x, ha):
        ax.text(loc_x, 0.025, text, transform=ax.transAxes, ha=ha, va="bottom",
                color="white", fontsize=7,
                bbox=dict(facecolor="black", alpha=0.45, pad=1.4, edgecolor="none"))

    for i, name in enumerate(EXAMPLES):
        rec, crop, orig, ref, pred = masks_for(name)
        stem = name.replace(".png", "")
        verdict = verdicts[name]
        ax_img, ax_ovl = axes[i, 0], axes[i, 1]
        for ax in (ax_img, ax_ovl):
            ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
        # filled translucent masks (model under, human on top) + crisp per-mask
        # contour edges so all three stay distinguishable where they coincide.
        for mask, colour in ((pred, COL_MODEL), (ref, COL_REF), (orig, COL_ORIG)):
            if mask.any():
                ax_ovl.imshow(_overlay(mask, colour, alpha=0.40), interpolation="nearest")
        for mask, colour in ((pred, COL_MODEL), (ref, COL_REF), (orig, COL_ORIG)):
            if mask.any():
                ax_ovl.contour(mask.astype(float), levels=[0.5], colors=[colour], linewidths=1.0)

        label_box(ax_img, f"{stem}  ({rec['stratum']} · {verdict})", 0.025, "left")
        wo, wr, wm = width_median(orig), width_median(ref), width_median(pred)
        label_box(ax_ovl, f"width med px  o {_wfmt(wo)} · r {_wfmt(wr)} · m {_wfmt(wm)}", 0.975, "right")
        if i == 0:
            ax_img.set_title("crop", fontsize=9)
            ax_ovl.set_title("mask overlay", fontsize=9)

    handles = [
        Patch(facecolor=COL_ORIG, alpha=0.7, label="original"),
        Patch(facecolor=COL_REF, alpha=0.7, label="reference (blinded)"),
        Patch(facecolor=COL_MODEL, alpha=0.7, label="model (>= 0.45)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.075))
    fig.suptitle("Gold-audit mask overlays", fontsize=11, y=0.985)
    fig.text(
        0.5, 0.014,
        "Two illustrative crops: c027 (interior trail) and c060 (an fp-control crop judged a real trail).\n"
        "Reference is a blinded single-author self-reannotation, not an independent gold standard.",
        ha="center", va="bottom", fontsize=6.6,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.93, bottom=0.15, hspace=0.06, wspace=0.03)

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")
    for name in EXAMPLES:
        print(f"  {name.replace('.png','')}  stratum={records[name]['stratum']}  verdict={verdicts[name]}")


if __name__ == "__main__":
    main()
