#!/usr/bin/env python
"""Build vector figures for the geometry-aware extension study.

Reads the locked geometry/Hough result JSONs and emits PDF + SVG figures
matching the thesis-figure style. Each figure is skipped (with a message) if
its inputs are absent, so the script is runnable after the Phase-1 evaluation
jobs land and again after the optional Phase-2 soft-label pilot.

Inputs (all force-added deliverables):
  results/classical/geometry_eval_t44_s2804.json        (base winner)
  results/classical/geometry_eval_attn_t7_s2804.json    (attention winner)
  results/classical/hough_prob_stratified_t44_s2804.json
  results/classical/geometry_eval_soft_dilated_t44_s2804.json   (Phase 2, optional)

Usage:
    venv/bin/python scripts/make_extension_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.make_thesis_figures import (
    CLASSICAL,
    COLORS,
    configure_style,
    load_json,
    save_vector,
)

BASE = CLASSICAL / "geometry_eval_t44_s2804.json"
ATTN = CLASSICAL / "geometry_eval_attn_t7_s2804.json"
HOUGH = CLASSICAL / "hough_prob_stratified_t44_s2804.json"
SOFT = CLASSICAL / "geometry_eval_soft_dilated_t44_s2804.json"
BASE_BND = CLASSICAL / "boundary_tolerant_unet_paper_arch_noise_topk_t44_s2804.json"
SOFT_BND = CLASSICAL / "boundary_tolerant_soft_dilated_t44_s2804.json"


def fig_fp_distance_histogram() -> None:
    if not BASE.exists():
        print(f"[skip] ext_1 FP-distance: missing {BASE.name}")
        return
    d = load_json(BASE)["fp_distance"]
    edges = np.asarray(d["histogram_bin_edges"], dtype=float)
    counts = np.asarray(d["histogram_counts"], dtype=float)
    total = counts.sum() + d["overflow_gt_50px"]
    frac = counts / total if total else counts
    centres = (edges[:-1] + edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xmax = 20
    keep = centres <= xmax
    ymax = max(float(frac[keep].max()) * 1.28, 0.05)
    ax.bar(centres[keep], frac[keep], width=0.9, color=COLORS["blue"], align="center")
    ax.set_ylim(0, ymax)
    label_positions = {1: (4.0, 0.90), 2: (6.1, 0.74), 3: (8.2, 0.58)}
    for px, label in ((1, "<=1 px"), (2, "<=2 px"), (3, "<=3 px")):
        f = d[f"fraction_within_{px}px"]
        ax.axvline(px, color=COLORS["red"], lw=0.8, ls="--", alpha=0.75)
        tx, ty_frac = label_positions[px]
        ax.annotate(
            f"{label}\n{f:.2f}", xy=(px, ymax * ty_frac), xytext=(tx, ymax * ty_frac),
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": COLORS["red"], "alpha": 0.75},
            fontsize=6, color=COLORS["red"], va="center", ha="left",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": COLORS["red"], "lw": 0.5, "alpha": 0.9},
        )
    ax.set_xlabel("Distance from false-positive pixel\nto nearest GT trail (px)")
    ax.set_ylabel("Fraction of distance-defined FP pixels")
    med = d.get("median_px_approx")
    g1 = d.get("fraction_within_1px_all_fp")
    if g1 is None:
        tot = d["n_fp_pixels_with_gt"] + d["whole_patch_fp_pixels"]
        g1 = d["fraction_within_1px"] * d["n_fp_pixels_with_gt"] / tot if tot else 0.0
    ax.set_title(
        f"FP distance-to-mask\nmedian approx. {med:.1f}px; <=1 px = "
        f"{d['fraction_within_1px']:.2f} distance-defined / {g1:.2f} all FP"
    )
    ax.set_xlim(0, xmax)
    fig.subplots_adjust(bottom=0.28, top=0.80)
    fig.text(
        0.5, 0.03,
        f"Histogram excludes {d['whole_patch_fp_pixels']:,} whole-patch FP with no GT in patch;\n"
        "those pixels are counted as non-boundary-adjacent in the all-FP fraction.",
        ha="center", fontsize=5.8, color="#555555",
    )
    save_vector(fig, "ext_1_fp_distance_histogram")
    print("[ok]  ext_1_fp_distance_histogram")


def fig_cldice_base_vs_attention() -> None:
    if not (BASE.exists() and ATTN.exists()):
        print(f"[skip] ext_2 clDice: need {BASE.name} and {ATTN.name}")
        return
    base, attn = load_json(BASE), load_json(ATTN)
    rows = [
        ["clDice", f"{base['centerline_dice']['mean']:.3f}", f"{attn['centerline_dice']['mean']:.3f}",
         f"{attn['centerline_dice']['mean'] - base['centerline_dice']['mean']:+.3f}"],
        ["exact Dice", f"{base['exact_metrics']['dice']:.3f}", f"{attn['exact_metrics']['dice']:.3f}",
         f"{attn['exact_metrics']['dice'] - base['exact_metrics']['dice']:+.3f}"],
    ]
    fig, ax = plt.subplots(figsize=(4.4, 1.75))
    ax.axis("off")
    ax.set_title("Topology and pixel overlap: base vs Attention U-Net", pad=8)
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Base U-Net\n(t44)", "Attention U-Net\n(t7)", "Delta\n(attn-base)"],
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    fig.text(
        0.5, 0.04,
        "Differences are marginal; table avoids visual overstatement. clDice is patch-averaged; exact Dice is micro over test pixels.",
        ha="center", fontsize=5.8, color="#555555",
    )
    save_vector(fig, "ext_2_cldice_base_vs_attention")
    print("[ok]  ext_2_cldice_base_vs_attention")


def fig_hough_prob_stratified() -> None:
    if not HOUGH.exists():
        print(f"[skip] ext_3 stratified Hough: missing {HOUGH.name}")
        return
    d = load_json(HOUGH)
    labels = ["U-Net\nonly", "binary\nHough", "stratified\nHough"]
    recall = [d["pixel_recall_pre"], d["pixel_recall_binary_hough"],
              d["pixel_recall_stratified_hough"]]
    colours = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

    fig, ax = plt.subplots(figsize=(4.2, 2.85))
    x = np.arange(len(labels))
    ax.bar(x, recall, 0.58, color=colours, alpha=0.82)
    for xi, r in zip(x, recall):
        ax.text(xi, r + 0.003, f"{r:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Pixel recall (positive images)")
    lo = min(recall) - 0.02
    ax.set_ylim(max(0.0, lo), 1.0)
    fp_b = d["hough_fp_pixels_binary"]; fp_s = d["hough_fp_pixels_stratified"]
    drec = d["pixel_recall_stratified_hough"] - d["pixel_recall_binary_hough"]
    ax.set_title("Probability-stratified Hough\nnegligible recall gain, added FP")
    fig.subplots_adjust(bottom=0.27, top=0.78)
    fig.text(
        0.5, 0.035,
        f"Delta recall vs binary Hough = +{drec:.4f}; patch-FNR unchanged.\n"
        f"Hough FP pixels {fp_b:,} -> {fp_s:,}; approximate strata union, not weighted accumulator.",
        ha="center", fontsize=5.8, color="#555555",
    )
    save_vector(fig, "ext_3_hough_prob_stratified")
    print("[ok]  ext_3_hough_prob_stratified")


def fig_soft_label_pilot() -> None:
    if not (BASE.exists() and SOFT.exists()):
        print(f"[skip] ext_4 soft-label pilot: need {BASE.name} and {SOFT.name} "
              "(Phase 2 — gated on the Phase-1 story)")
        return
    base, soft = load_json(BASE), load_json(SOFT)
    rows = [
        ["exact Dice", base["exact_metrics"]["dice"], soft["exact_metrics"]["dice"]],
        ["clDice", base["centerline_dice"]["mean"], soft["centerline_dice"]["mean"]],
    ]
    if BASE_BND.exists() and SOFT_BND.exists():
        rows.insert(1, ["+/-1 px F1", load_json(BASE_BND)["per_tolerance"]["1"]["f1"],
                       load_json(SOFT_BND)["per_tolerance"]["1"]["f1"]])
    table_rows = [[name, f"{base_v:.3f}", f"{soft_v:.3f}", f"{soft_v - base_v:+.3f}"]
                  for name, base_v, soft_v in rows]
    fig, ax = plt.subplots(figsize=(4.4, 1.95))
    ax.axis("off")
    ax.set_title("Soft-label pilot vs hard labels\nsingle seed 2804", pad=8)
    table = ax.table(
        cellText=table_rows,
        colLabels=["Metric", "Hard labels\n(winner)", "Dilated-soft\n(pilot)", "Delta\n(soft-hard)"],
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    fig.text(
        0.5, 0.035,
        "Single-seed protocol variant; table avoids overstating marginal/null differences.",
        ha="center", fontsize=5.8, color="#555555",
    )
    save_vector(fig, "ext_4_soft_label_pilot")
    print("[ok]  ext_4_soft_label_pilot")

def main() -> None:
    configure_style()
    fig_fp_distance_histogram()
    fig_cldice_base_vs_attention()
    fig_hough_prob_stratified()
    fig_soft_label_pilot()


if __name__ == "__main__":
    main()
