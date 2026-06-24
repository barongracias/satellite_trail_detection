#!/usr/bin/env python
"""Supplementary qualitative figure for the blinded gold audit (thesis section 6.2).

Visualises the gold-audit result (results/classical/gold_audit_eval.json) on a few
example crops: the original mask, the blinded single-author REFERENCE
re-annotation, and the locked MODEL prediction (>= 0.45) overlaid in three
distinct colours on the audit crop image. It shows the ~6 px three-way agreement
on boundary crops and the model over-firing on the false-positive controls.

INFERENCE-ONLY on the locked model: no retraining, re-tuning, reselection, or
threshold change. Reuses gold_audit_eval.py's exact loaders (_load_normalised_patch
+ _infer_batch at threshold 0.45, load_raw_mask, load_annotation_mask) so the
rendered masks match the scored JSON.

Deterministic, non-cherry-picked crop selection (also stated in the caption):
  * 3 boundary examples: the three lowest-index crops with stratum in
    {interior, endpoint}, verdict "trail", and original / reference / model all
    non-empty (agreement cases).
  * 2 FP-control examples: the two lowest-index crops with stratum "fp" where the
    model fired (prediction non-empty) and the reference is empty (over-firing).

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


def _load_verdicts(path: str) -> dict[str, str]:
    with open(path, newline="") as handle:
        return {row["crop_name"].strip(): row["verdict"].strip() for row in csv.DictReader(handle)}


def _wfmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.0f}"


def main() -> None:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalisation = load_segmentation_model(CHECKPOINT, device)

    sealed = json.loads(Path(SEALED).read_text())
    records = sorted(sealed["crops"], key=lambda r: r["crop_name"])
    verdicts = _load_verdicts(VERDICTS)

    patch_manifest = pd.read_csv(Path(PATCH_DIR) / "manifest.csv")
    stats: dict[str, tuple[float, float]] = {}
    if {"image_mean", "image_std"}.issubset(patch_manifest.columns):
        for source, group in patch_manifest.groupby("source_image"):
            stats[str(source)] = (float(group["image_mean"].iloc[0]), float(group["image_std"].iloc[0]))

    mask_cache: dict[str, np.ndarray] = {}

    def original_of(rec: dict) -> np.ndarray:
        source = rec["source_image"]
        if source not in mask_cache:
            mask_cache[source] = load_raw_mask(source)
        y0, x0 = int(rec["y0"]), int(rec["x0"])
        return mask_cache[source][y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]

    def predict(rec: dict) -> np.ndarray:
        mean, std = stats.get(rec["source_image"], (None, None))
        patch = _load_normalised_patch(str(CROP_DIR / rec["crop_name"]), normalisation, mean, std)
        probability = _infer_batch([patch], model, device)[0]
        return probability >= THRESHOLD

    # --- deterministic selection by the fixed rule -----------------------------
    boundary_sel: list[tuple] = []
    for rec in records:
        if len(boundary_sel) >= 3:
            break
        if rec["stratum"] in BOUNDARY_STRATA and verdicts[rec["crop_name"]] == "trail":
            ref = load_annotation_mask(ANNOT_DIR / rec["crop_name"])
            orig = original_of(rec)
            pred = predict(rec)
            if ref.any() and orig.any() and pred.any():
                boundary_sel.append((rec, orig, ref, pred))

    fp_sel: list[tuple] = []
    for rec in records:
        if len(fp_sel) >= 2:
            break
        if rec["stratum"] == "fp":
            ref = load_annotation_mask(ANNOT_DIR / rec["crop_name"])
            pred = predict(rec)
            if pred.any() and not ref.any():
                fp_sel.append((rec, original_of(rec), ref, pred))

    rows = boundary_sel + fp_sel
    if len(boundary_sel) < 3 or len(fp_sel) < 2:
        print(f"[warn] selection short: {len(boundary_sel)} boundary, {len(fp_sel)} fp")

    # --- render ----------------------------------------------------------------
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8})
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(6.2, 3.05 * n))
    axes = np.atleast_2d(axes)

    for i, (rec, orig, ref, pred) in enumerate(rows):
        name = rec["crop_name"].replace(".png", "")
        stratum = rec["stratum"]
        verdict = verdicts[rec["crop_name"]]
        with Image.open(CROP_DIR / rec["crop_name"]) as im:
            crop = np.asarray(im.convert("L"), dtype=np.uint8)

        ax_img, ax_ovl = axes[i, 0], axes[i, 1]
        for ax in (ax_img, ax_ovl):
            ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
        # three mask boundaries as contours on the overlay panel; distinct line
        # styles keep all three legible where they nearly coincide (agreement).
        for mask, colour, style, lw in (
            (orig, COL_ORIG, "solid", 1.5),
            (ref, COL_REF, "dashed", 1.2),
            (pred, COL_MODEL, "dotted", 1.2),
        ):
            if mask.any():
                ax_ovl.contour(mask.astype(float), levels=[0.5], colors=[colour],
                               linewidths=lw, linestyles=style)

        wo = width_median(orig); wr = width_median(ref); wm = width_median(pred)
        ax_img.set_ylabel(f"{name}\n{stratum} | {verdict}", fontsize=8)
        ax_img.set_title("audit crop", fontsize=8)
        ax_ovl.set_title(
            f"overlay   width med (px): orig {_wfmt(wo)} / ref {_wfmt(wr)} / model {_wfmt(wm)}",
            fontsize=7.5,
        )

    handles = [
        Line2D([0], [0], color=COL_ORIG, lw=1.6, ls="solid", label="original mask"),
        Line2D([0], [0], color=COL_REF, lw=1.6, ls="dashed", label="reference (blinded re-annotation)"),
        Line2D([0], [0], color=COL_MODEL, lw=1.6, ls="dotted", label="locked model (>= 0.45)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(
        "Gold audit: original vs blinded single-author reference vs locked model",
        fontsize=10, y=0.997,
    )
    fig.text(
        0.5, 0.03,
        "Rows 1-3: lowest-index interior/endpoint 'trail' crops with all three masks non-empty (agreement).\n"
        "Rows 4-5: lowest-index 'fp'-control crops where the model fired and the reference is empty (over-firing).\n"
        "Crops selected by a fixed rule, not by eye. Reference = blinded single-author self-reannotation, not an independent gold standard.",
        ha="center", va="bottom", fontsize=6.2,
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.96, bottom=0.085, hspace=0.16, wspace=0.04)

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Wrote {out_path}")
    print("Selected crops:")
    for rec, *_ in rows:
        print(f"  {rec['crop_name'].replace('.png','')}  stratum={rec['stratum']}  verdict={verdicts[rec['crop_name']]}")


if __name__ == "__main__":
    main()
