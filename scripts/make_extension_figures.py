#!/usr/bin/env python
"""Build vector figures for the geometry-aware extension study.

Reads the locked geometry/Hough result JSONs and emits only the thesis-useful
figures. Marginal/null comparisons that are clearer as report tables are
printed for the agent notes instead of being saved as image artifacts.
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
    PASTEL_COLORS,
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

    fig, ax = plt.subplots(figsize=(4.15, 2.7))
    xmax = 7.5
    keep = centres <= xmax
    ymax = max(float(frac[keep].max()) * 1.28, 0.05)
    ax.bar(centres[keep], frac[keep], width=0.9, color=COLORS["blue"], align="center")
    ax.set_ylim(0, ymax)
    label_positions = {1: (1.35, 0.90), 2: (2.55, 0.74), 3: (3.75, 0.58)}
    for px, label in ((1, "≤1 px"), (2, "≤2 px"), (3, "≤3 px")):
        f = d[f"fraction_within_{px}px"]
        ax.axvline(px, color=COLORS["red"], lw=0.8, ls="--", alpha=0.75)
        tx, ty_frac = label_positions[px]
        ax.annotate(
            f"{label}\n{f:.2f}", xy=(px, ymax * ty_frac), xytext=(tx, ymax * ty_frac),
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": COLORS["red"], "alpha": 0.75},
            fontsize=5.8, color=COLORS["red"], va="center", ha="left",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": COLORS["red"], "lw": 0.5, "alpha": 0.9},
        )
    ax.set_xlabel("Distance from FP pixel to nearest GT trail (px)", fontsize=7)
    ax.set_ylabel("Fraction of FP pixels", fontsize=7)
    med = d.get("median_px_approx")
    g1 = d.get("fraction_within_1px_all_fp")
    if g1 is None:
        tot = d["n_fp_pixels_with_gt"] + d["whole_patch_fp_pixels"]
        g1 = d["fraction_within_1px"] * d["n_fp_pixels_with_gt"] / tot if tot else 0.0
    ax.set_title(
        f"FP distance-to-mask\nmedian approx. {med:.1f}px; ≤1 px = "
        f"{d['fraction_within_1px']:.2f} distance-defined / {g1:.2f} all FP",
        fontsize=8,
    )
    ax.set_xlim(0, xmax)
    fig.subplots_adjust(left=0.16, bottom=0.18, top=0.78)
    save_vector(fig, "ext_1_fp_distance_histogram")
    print("[ok]  ext_1_fp_distance_histogram")


def fig_cldice_base_vs_attention() -> None:
    if not (BASE.exists() and ATTN.exists()):
        print(f"[skip] ext_2 clDice: need {BASE.name} and {ATTN.name}")
        return
    base, attn = load_json(BASE), load_json(ATTN)
    base_cl = base["centerline_dice"]["mean"]
    attn_cl = attn["centerline_dice"]["mean"]
    base_dice = base["exact_metrics"]["dice"]
    attn_dice = attn["exact_metrics"]["dice"]
    print("[table-only] ext_2_cldice_base_vs_attention")
    print(f"  clDice: base={base_cl:.3f}, attention={attn_cl:.3f}, delta={attn_cl - base_cl:+.3f}")
    print(f"  exact Dice: base={base_dice:.3f}, attention={attn_dice:.3f}, delta={attn_dice - base_dice:+.3f}")


def fig_hough_prob_stratified() -> None:
    if not HOUGH.exists():
        print(f"[skip] ext_3 stratified Hough: missing {HOUGH.name}")
        return
    d = load_json(HOUGH)
    labels = ["U-Net\nonly", "binary\nHough", "stratified\nHough"]
    recall = [d["pixel_recall_pre"], d["pixel_recall_binary_hough"], d["pixel_recall_stratified_hough"]]
    colours = [PASTEL_COLORS["blue"], PASTEL_COLORS["orange"], PASTEL_COLORS["green"]]

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    x = np.arange(len(labels))
    ax.bar(x, recall, 0.58, color=colours, alpha=0.9)
    for xi, r in zip(x, recall):
        ax.text(xi, r + 0.003, f"{r:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pixel recall (positive images)")
    ax.set_ylim(max(0.0, min(recall) - 0.02), 1.0)
    ax.set_title("Probability-stratified Hough\npixel recall on positive images")
    fig.subplots_adjust(bottom=0.16, top=0.80)
    fp_b = d["hough_fp_pixels_binary"]
    fp_s = d["hough_fp_pixels_stratified"]
    drec = d["pixel_recall_stratified_hough"] - d["pixel_recall_binary_hough"]
    print(f"[note] ext_3 delta recall vs binary Hough = +{drec:.4f}; Hough FP pixels {fp_b:,} -> {fp_s:,}")
    save_vector(fig, "ext_3_hough_prob_stratified")
    print("[ok]  ext_3_hough_prob_stratified")


def fig_soft_label_pilot() -> None:
    if not (BASE.exists() and SOFT.exists()):
        print(f"[skip] ext_4 soft-label pilot: need {BASE.name} and {SOFT.name} (Phase 2 — gated on the Phase-1 story)")
        return
    base, soft = load_json(BASE), load_json(SOFT)
    rows = [
        ("exact Dice", base["exact_metrics"]["dice"], soft["exact_metrics"]["dice"]),
        ("clDice", base["centerline_dice"]["mean"], soft["centerline_dice"]["mean"]),
    ]
    if BASE_BND.exists() and SOFT_BND.exists():
        rows.insert(1, ("±1 px F1", load_json(BASE_BND)["per_tolerance"]["1"]["f1"], load_json(SOFT_BND)["per_tolerance"]["1"]["f1"]))
    print("[table-only] ext_4_soft_label_pilot")
    for name, base_v, soft_v in rows:
        print(f"  {name}: hard={base_v:.3f}, soft={soft_v:.3f}, delta={soft_v - base_v:+.3f}")


def main() -> None:
    configure_style()
    fig_fp_distance_histogram()
    fig_cldice_base_vs_attention()
    fig_hough_prob_stratified()
    fig_soft_label_pilot()


if __name__ == "__main__":
    main()
