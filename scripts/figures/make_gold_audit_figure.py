#!/usr/bin/env python
"""Supplementary qualitative figure for the blinded gold audit (thesis section 6.2).

Five crops chosen by a fixed, non-cherry-picked rule (stated in the caption), each
shown as the audit crop image next to a 3-way contour overlay of the ORIGINAL
mask, the blinded single-author REFERENCE re-annotation, and the locked MODEL
prediction (>= 0.45). Masks are drawn as thin contour outlines with nested line
widths so the near-coincident boundary masks read as a layered colour band; an
empty mask draws no contour, so its absence is visible.

INFERENCE-ONLY on the locked model: no retraining, re-tuning, reselection, or
threshold change. Reuses gold_audit_eval.py's exact loaders (_load_normalised_patch
+ _infer_batch at threshold 0.45, load_raw_mask, load_annotation_mask) so the
rendered masks match the scored JSON.

Deterministic selection (also stated in the caption):
  * 3 boundary examples: the three LOWEST-INDEX crops with stratum in
    {interior, endpoint}, verdict "trail", and original / reference / model all
    non-empty (agreement cases).
  * 2 FP-control examples: the two LOWEST-INDEX crops with stratum "fp" where the
    model fired (prediction non-empty) AND the reference is empty (over-firing).

The reference is a blinded single-author self-reannotation, NOT an independent
gold standard. Saves a single PDF (no SVG).
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
from matplotlib.lines import Line2D
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
BOUNDARY_STRATA = ("interior", "endpoint")

COL_ORIG = "#D55E00"   # vermillion
COL_REF = "#009E73"    # bluish green
COL_MODEL = "#0072B2"  # blue
# (colour, mask-name, contour linewidth) drawn widest-first so all three nest.
LAYERS = [(COL_ORIG, "original", 2.6), (COL_REF, "reference", 1.6), (COL_MODEL, "model", 0.8)]


def _load_verdicts(path: str) -> dict[str, str]:
    with open(path, newline="") as handle:
        return {row["crop_name"].strip(): row["verdict"].strip() for row in csv.DictReader(handle)}


def _wfmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.0f}"


def main() -> None:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalisation = load_segmentation_model(CHECKPOINT, device)

    records = sorted(json.loads(Path(SEALED).read_text())["crops"], key=lambda r: r["crop_name"])
    verdicts = _load_verdicts(VERDICTS)

    patch_manifest = pd.read_csv(Path(PATCH_DIR) / "manifest.csv")
    stats: dict[str, tuple[float, float]] = {}
    if {"image_mean", "image_std"}.issubset(patch_manifest.columns):
        for source, group in patch_manifest.groupby("source_image"):
            stats[str(source)] = (float(group["image_mean"].iloc[0]), float(group["image_std"].iloc[0]))

    mask_cache: dict[str, np.ndarray] = {}

    def masks_for(rec: dict):
        name = rec["crop_name"]
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
        return crop, original, reference, prediction

    # --- deterministic, lowest-index-first selection --------------------------
    boundary_sel: list[tuple] = []
    for rec in records:
        if len(boundary_sel) >= 3:
            break
        if rec["stratum"] in BOUNDARY_STRATA and verdicts[rec["crop_name"]] == "trail":
            crop, orig, ref, pred = masks_for(rec)
            if orig.any() and ref.any() and pred.any():
                boundary_sel.append((rec, crop, orig, ref, pred))

    fp_sel: list[tuple] = []
    for rec in records:
        if len(fp_sel) >= 2:
            break
        if rec["stratum"] == "fp":
            crop, orig, ref, pred = masks_for(rec)
            if pred.any() and not ref.any():
                fp_sel.append((rec, crop, orig, ref, pred))

    rows = boundary_sel + fp_sel
    if len(boundary_sel) < 3 or len(fp_sel) < 2:
        print(f"[warn] selection short: {len(boundary_sel)} boundary, {len(fp_sel)} fp")

    # --- render: crop | 3-way contour overlay (nested widths) -----------------
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8})
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(5.6, 2.55 * n + 0.5))
    axes = np.atleast_2d(axes)

    def corner(ax, text, ha="right", x=0.97):
        ax.text(x, 0.03, text, transform=ax.transAxes, ha=ha, va="bottom",
                color="white", fontsize=7,
                bbox=dict(facecolor="black", alpha=0.45, pad=1.2, edgecolor="none"))

    for i, (rec, crop, orig, ref, pred) in enumerate(rows):
        masks = {"original": orig, "reference": ref, "model": pred}
        stem = rec["crop_name"].replace(".png", "")
        ax_img, ax_ovl = axes[i, 0], axes[i, 1]
        for ax in (ax_img, ax_ovl):
            ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
        for colour, mname, lw in LAYERS:
            mask = masks[mname]
            if mask.any():
                ax_ovl.contour(mask.astype(float), levels=[0.5], colors=[colour], linewidths=lw, alpha=0.95)
        corner(ax_img, f"{stem}  {rec['stratum']}·{verdicts[rec['crop_name']]}", ha="left", x=0.03)
        w = {k: width_median(v) for k, v in masks.items()}
        corner(ax_ovl, f"width px  o {_wfmt(w['original'])} · r {_wfmt(w['reference'])} · m {_wfmt(w['model'])}")
        if i == 0:
            ax_img.set_title("audit crop", fontsize=9)
            ax_ovl.set_title("mask contours", fontsize=9)

    handles = [Line2D([0], [0], color=c, lw=2.2, label=lab)
               for c, lab in ((COL_ORIG, "original"), (COL_REF, "reference"), (COL_MODEL, "model >= 0.45"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.044))
    fig.suptitle("Gold-audit mask contours", fontsize=11, y=0.995)
    fig.text(
        0.5, 0.01,
        "Rows 1-3: lowest-index interior/endpoint 'trail' crops with original, reference and model all non-empty.\n"
        "Rows 4-5: lowest-index 'fp'-control crops where the model fired and the reference is empty. Crops chosen by\n"
        "this fixed rule, not by eye. Reference = blinded single-author self-reannotation, not an independent gold standard.",
        ha="center", va="bottom", fontsize=6.2,
    )
    fig.subplots_adjust(left=0.012, right=0.988, top=0.965, bottom=0.085, hspace=0.06, wspace=0.025)

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print("Selected crops (fixed rule, lowest-index-first):")
    for rec, *_ in rows:
        print(f"  {rec['crop_name'].replace('.png','')}  stratum={rec['stratum']}  verdict={verdicts[rec['crop_name']]}")


if __name__ == "__main__":
    main()
