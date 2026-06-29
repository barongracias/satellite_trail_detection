#!/usr/bin/env python
"""Supplementary vector figures for the additive submission-polish diagnostics.

Two figures, both purely from already-written locked-model diagnostic JSONs
(inference-only; no retraining/reselection):

  supp_1_hough_thickness_recovery
      Strict pixel precision vs drawn Hough line thickness {1,2,3} on the parity
      canvas, decomposing the post-Hough precision fall into the share
      recoverable by thinning the rasterised line (line-drawing convention over
      the ~6 px masks) vs the residual not recovered by 1 px thinning.
      Companion panel shows trail-pixel completeness (post-Hough pixel recall)
      stays high across thicknesses -- the recovery reconnects gaps, it does not
      buy precision by dropping detections.

  supp_2_diagnostics_multiseed_bands
      Five-seed mean +/- sample SD bands for the three load-bearing diagnostics
      (FP inter-patch fraction, +-1 px boundary-tolerant precision, FP-distance
      within 1 px), each seed at its own validation-optimal threshold, with the
      published s2804 anchor highlighted.

These are supplementary: the thesis does not need to embed them. Run from repo
root with the project venv (matplotlib required).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.figures.make_thesis_figures import (
    CLASSICAL,
    COLORS,
    configure_style,
    load_json,
    save_vector,
)

THICK_FILES = {
    1: CLASSICAL / "hough_postprocess_winner_t44_s2804_parity_thick1.json",
    2: CLASSICAL / "hough_postprocess_winner_t44_s2804_parity_thick2.json",
    3: CLASSICAL / "hough_postprocess_winner_t44_s2804_parity.json",  # committed (thickness 3)
}
SUMMARY = CLASSICAL / "diagnostics_multiseed_summary.json"


def fig_hough_thickness_recovery() -> None:
    missing = [str(p.name) for p in THICK_FILES.values() if not p.exists()]
    if missing:
        print(f"[skip] supp_1 thickness recovery: missing {missing}")
        return
    data = {t: load_json(p) for t, p in THICK_FILES.items()}
    thicks = [1, 2, 3]
    prec_post = [data[t]["pixel_precision_post"] for t in thicks]
    recall_post = [data[t]["pixel_recall_post"] for t in thicks]
    pre = data[3]["pixel_precision_pre"]  # thickness-independent (U-Net only)

    p1, p3 = prec_post[0], prec_post[2]
    fall = pre - p3
    recovered = p1 - p3
    conv_share = recovered / fall if fall else float("nan")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    x = np.arange(len(thicks))

    # --- Left: precision vs thickness with a single recoverable band ----------
    axL.bar(x, prec_post, 0.55, color=COLORS["blue"], alpha=0.9, zorder=2)
    for xi, p in zip(x, prec_post):
        axL.text(xi, p - 0.025, f"{p:.2f}", ha="center", va="top",
                 color="black", fontsize=7, zorder=3)
    # two light bands: green = recoverable by thinning 3 px -> 1 px;
    # orange = residual not recovered by 1 px thinning (thick1 -> pre-Hough).
    axL.axhspan(p3, p1, color=COLORS["green"], alpha=0.16, zorder=0)
    axL.axhspan(p1, pre, color=COLORS["red"], alpha=0.12, zorder=0)
    # pre-Hough reference + the two share labels, all right-aligned at one x
    axL.axhline(pre, color=COLORS["black"], linestyle="--", lw=1.0, zorder=1)
    lbl_x = 2.55
    axL.text(lbl_x, pre + 0.022, f"pre-Hough {pre:.2f}", ha="right", fontsize=6.5)
    axL.text(lbl_x, (pre + p1) / 2, f"{(1 - conv_share)*100:.0f}% residual",
             ha="right", va="center", fontsize=6.5, color=COLORS["red"])
    axL.text(lbl_x, (p1 + p3) / 2, f"{conv_share*100:.0f}% rasterisation",
             ha="right", va="center", fontsize=6.5, color=COLORS["green"])
    axL.set_xticks(x)
    axL.set_xticklabels(thicks)
    axL.set_xlabel("Line thickness (px)")
    axL.set_ylabel("Pixel precision (post)")
    axL.set_title("Strict-precision recovery")
    axL.set_xlim(-0.6, 2.6)
    axL.set_ylim(0.0, max(pre + 0.08, 0.88))

    # --- Right: completeness sanity check ------------------------------------
    axR.plot(x, recall_post, marker="o", color=COLORS["orange"], lw=1.4)
    n = len(recall_post)
    for i, (xi, r) in enumerate(zip(x, recall_post)):
        ha = "left" if i == 0 else ("right" if i == n - 1 else "center")
        dx = 0.07 if i == 0 else (-0.07 if i == n - 1 else 0.0)
        axR.text(xi + dx, r - 0.006, f"{r:.3f}", ha=ha, va="top", fontsize=7)
    axR.set_xticks(x)
    axR.set_xticklabels(thicks)
    axR.set_xlabel("Line thickness (px)")
    axR.set_ylabel("Pixel recall (post)")
    axR.set_title("Completeness stays high")
    axR.set_ylim(0.90, 1.0)

    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.18, top=0.86, wspace=0.30)
    save_vector(fig, "supp_1_hough_thickness_recovery")
    print(f"[ok]  supp_1_hough_thickness_recovery "
          f"(recovered {recovered:.3f} of {fall:.3f} fall = {conv_share*100:.0f}% rasterisation)")


def fig_multiseed_bands() -> None:
    if not SUMMARY.exists():
        print(f"[skip] supp_2 multiseed bands: missing {SUMMARY.name}")
        return
    d = load_json(SUMMARY)["diagnostics"]
    order = [
        ("fp_decomposition_inter_patch_fp_fraction", "Inter-patch FP"),
        ("boundary_tolerant_1px_precision", "±1 px precision"),
        ("fp_distance_fraction_within_1px_all_fp", "FP within 1 px"),
    ]
    anchor_seed = load_json(SUMMARY).get("anchor_seed", 2804)

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.7))
    band_h = mean_h = seed_h = anchor_h = None
    for ax, (key, title) in zip(axes, order):
        b = d[key]
        rows = b["per_seed"]
        seeds = [r["seed"] for r in rows]
        vals = [r["value"] for r in rows]
        mean, sd = b["mean"], b["sample_sd"]

        xs = np.arange(len(seeds))
        band_h = ax.axhspan(mean - sd, mean + sd, color=COLORS["blue"], alpha=0.14, zorder=0)
        mean_h = ax.axhline(mean, color=COLORS["blue"], lw=1.1, zorder=1)
        for xi, s, v in zip(xs, seeds, vals):
            if s == anchor_seed:
                anchor_h = ax.scatter([xi], [v], s=36, color=COLORS["red"], marker="D",
                                      zorder=3, edgecolor="black", lw=0.5)
            else:
                seed_h = ax.scatter([xi], [v], s=20, color=COLORS["orange"], zorder=3)
        # mean +/- sd as compact inline text, replaces the per-panel legend
        ax.text(0.5, 0.95, f"mean {mean:.3f} ± {sd:.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.5, color=COLORS["blue"])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=6)
        ax.set_title(title, fontsize=8)
        lo = min(min(vals), mean - sd)
        hi = max(max(vals), mean + sd)
        pad = 0.18 * (hi - lo) + 0.005
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("Value")
    fig.suptitle("Five-seed diagnostic bands", fontsize=9)
    fig.legend([mean_h, band_h, seed_h, anchor_h],
               ["5-seed mean", "±1 SD", "seed", f"s{anchor_seed} (model-best)"],
               loc="lower center", ncol=4, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 0.04))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.21, top=0.82, wspace=0.28)
    save_vector(fig, "supp_2_diagnostics_multiseed_bands")
    print("[ok]  supp_2_diagnostics_multiseed_bands")


def main() -> None:
    configure_style()
    fig_hough_thickness_recovery()
    fig_multiseed_bands()


if __name__ == "__main__":
    main()
