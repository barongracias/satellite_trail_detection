#!/usr/bin/env python
"""Supplementary qualitative figure for the blinded gold audit (thesis section 6.2).

Five crops chosen by a fixed, non-cherry-picked rule, each shown as a 3-way contour
overlay of the ORIGINAL mask, the blinded single-author REFERENCE re-annotation,
and the locked MODEL prediction (>= 0.45) on the audit crop. Masks are thin contour
outlines with nested line widths so near-coincident boundary masks read as a
layered colour band; an empty mask draws no contour, so its absence is visible.
The figure is intentionally bare (panels + top legend only): the crop identities,
stroke widths, selection rule, and the blinded-reference caveat belong in the
thesis figure caption, not baked into the image.

INFERENCE-ONLY on the locked model: no retraining, re-tuning, reselection, or
threshold change. Reuses gold_audit_eval.py's exact loaders (_load_normalised_patch
+ _infer_batch at threshold 0.45, load_raw_mask, load_annotation_mask) so the
rendered masks match the scored JSON.

Deterministic selection (lowest-index-first; reproduced in stdout):
  * Top row -- 3 boundary crops: the three lowest-index crops with stratum in
    {interior, endpoint}, verdict "trail", and original / reference / model all
    non-empty (agreement).
  * Bottom row -- 2 FP-control crops: the two lowest-index crops with stratum
    "fp" where the model fired (prediction non-empty) and the reference is empty
    (over-firing).

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

    # --- render: square 2x3 grid of contour overlays, legend on top -----------
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8})
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6, 5.05))
    flat = axes.ravel()
    for k, (rec, crop, orig, ref, pred) in enumerate(rows):
        ax = flat[k]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        masks = {"original": orig, "reference": ref, "model": pred}
        for colour, mname, lw in LAYERS:
            mask = masks[mname]
            if mask.any():
                ax.contour(mask.astype(float), levels=[0.5], colors=[colour], linewidths=lw, alpha=0.95)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([0], [0], color=c, lw=2.8, label=lab)
               for c, lab in ((COL_ORIG, "original"), (COL_REF, "reference"), (COL_MODEL, "model >= 0.45"))]
    empties = list(flat[len(rows):])
    for ax in empties:
        ax.axis("off")
    # use the spare bottom-right cell for a vertically stacked legend
    if empties:
        empties[0].legend(handles=handles, loc="center", ncol=1, frameon=False,
                          fontsize=9.5, handlelength=1.8, labelspacing=1.0)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle("Gold-audit mask contours", fontsize=11, y=0.96)
    fig.subplots_adjust(left=0.008, right=0.992, top=0.915, bottom=0.008, hspace=0.04, wspace=0.025)

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Wrote {out_path}")
    print("Selected crops (fixed rule, lowest-index-first; panel order = reading order):")
    for rec, crop, orig, ref, pred in rows:
        w = {"original": width_median(orig), "reference": width_median(ref), "model": width_median(pred)}
        print(f"  {rec['crop_name'].replace('.png','')}  {rec['stratum']}·{verdicts[rec['crop_name']]}  "
              f"width o/r/m = {_wfmt(w['original'])}/{_wfmt(w['reference'])}/{_wfmt(w['model'])}")


if __name__ == "__main__":
    main()
