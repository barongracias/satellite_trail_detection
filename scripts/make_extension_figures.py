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

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    xmax = 20
    keep = centres <= xmax
    ax.bar(centres[keep], frac[keep], width=0.9, color=COLORS["blue"], align="center")
    for px, label in ((1, "≤1px"), (2, "≤2px"), (3, "≤3px")):
        f = d[f"fraction_within_{px}px"]
        ax.axvline(px, color=COLORS["red"], lw=0.8, ls="--", alpha=0.7)
        ax.text(px + 0.15, ax.get_ylim()[1] * (0.95 - 0.12 * px), f"{label}: {f:.2f}",
                fontsize=6, color=COLORS["red"])
    ax.set_xlabel("Distance from false-positive pixel to nearest GT trail (px)")
    ax.set_ylabel("Fraction of FP pixels")
    med = d.get("median_px_approx")
    ax.set_title(f"FP distance-to-mask  (median ≈ {med:.1f}px, "
                 f"{d['whole_patch_fp_pixels']:,} whole-patch FP excluded)")
    ax.set_xlim(0, xmax)
    save_vector(fig, "ext_1_fp_distance_histogram")
    print("[ok]  ext_1_fp_distance_histogram")


def fig_cldice_base_vs_attention() -> None:
    if not (BASE.exists() and ATTN.exists()):
        print(f"[skip] ext_2 clDice: need {BASE.name} and {ATTN.name}")
        return
    base, attn = load_json(BASE), load_json(ATTN)
    metrics = ["clDice", "exact Dice"]
    base_vals = [base["centerline_dice"]["mean"], base["exact_metrics"]["dice"]]
    attn_vals = [attn["centerline_dice"]["mean"], attn["exact_metrics"]["dice"]]
    x = np.arange(len(metrics))
    w = 0.36

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.bar(x - w / 2, base_vals, w, label="Base U-Net (t44)", color=COLORS["blue"])
    ax.bar(x + w / 2, attn_vals, w, label="Attention U-Net (t7)", color=COLORS["orange"])
    for xi, b, a in zip(x, base_vals, attn_vals):
        ax.text(xi - w / 2, b + 0.005, f"{b:.3f}", ha="center", fontsize=6)
        ax.text(xi + w / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Topology (clDice) vs pixel Dice")
    ax.legend(loc="lower center")
    save_vector(fig, "ext_2_cldice_base_vs_attention")
    print("[ok]  ext_2_cldice_base_vs_attention")


def fig_hough_prob_stratified() -> None:
    if not HOUGH.exists():
        print(f"[skip] ext_3 stratified Hough: missing {HOUGH.name}")
        return
    d = load_json(HOUGH)
    labels = ["U-Net only", "binary Hough", "stratified Hough"]
    recall = [d["pixel_recall_pre"], d["pixel_recall_binary_hough"],
              d["pixel_recall_stratified_hough"]]
    colours = [COLORS["black"], COLORS["sky"], COLORS["green"]]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x = np.arange(len(labels))
    ax.bar(x, recall, 0.6, color=colours)
    for xi, r in zip(x, recall):
        ax.text(xi, r + 0.003, f"{r:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Pixel recall (positive images)")
    lo = min(recall) - 0.02
    ax.set_ylim(max(0.0, lo), 1.0)
    fp_b = d["hough_fp_pixels_binary"]; fp_s = d["hough_fp_pixels_stratified"]
    ax.set_title(f"Stratified-Hough recall  (Hough FP px: {fp_b:,}→{fp_s:,})")
    save_vector(fig, "ext_3_hough_prob_stratified")
    print("[ok]  ext_3_hough_prob_stratified")


def fig_soft_label_pilot() -> None:
    if not (BASE.exists() and SOFT.exists()):
        print(f"[skip] ext_4 soft-label pilot: need {BASE.name} and {SOFT.name} "
              "(Phase 2 — gated on the Phase-1 story)")
        return
    base, soft = load_json(BASE), load_json(SOFT)
    metrics = ["exact Dice", "clDice"]
    base_vals = [base["exact_metrics"]["dice"], base["centerline_dice"]["mean"]]
    soft_vals = [soft["exact_metrics"]["dice"], soft["centerline_dice"]["mean"]]
    x = np.arange(len(metrics))
    w = 0.36

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.bar(x - w / 2, base_vals, w, label="Hard labels (winner)", color=COLORS["blue"])
    ax.bar(x + w / 2, soft_vals, w, label="Dilated-soft (pilot)", color=COLORS["purple"])
    for xi, b, s in zip(x, base_vals, soft_vals):
        ax.text(xi - w / 2, b + 0.005, f"{b:.3f}", ha="center", fontsize=6)
        ax.text(xi + w / 2, s + 0.005, f"{s:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Soft-label pilot vs hard labels (single seed 2804)")
    ax.legend(loc="lower center")
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
