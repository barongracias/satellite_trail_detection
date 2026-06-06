#!/usr/bin/env python
"""Build vector figures for the geometry-aware extension study.

Reads the locked geometry/Hough result JSONs and emits only the thesis-useful
figures. Marginal/null comparisons that are clearer as report tables are
documented in agents/plots.md instead of being saved as image artifacts.
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

BASE = CLASSICAL / "geometry_eval_t44_s2804.json"
HOUGH = CLASSICAL / "hough_prob_stratified_t44_s2804.json"


def fig_fp_distance_histogram() -> None:
    if not BASE.exists():
        print(f"[skip] ext_1 FP-distance: missing {BASE.name}")
        return
    d = load_json(BASE)["fp_distance"]
    edges = np.asarray(d["histogram_bin_edges"], dtype=float)
    counts = np.asarray(d["histogram_counts"], dtype=float)
    total = counts.sum() + d["overflow_gt_50px"]
    right_edges = edges[1:]

    # The authoritative cumulative values are the exact "<= N px" fractions. FP-to-GT
    # distances never fall below 1 px and cluster at 1.0, sqrt(2), 2.0, ..., so a raw
    # cumsum of the 1 px right-open bins drops the dominant 1.0 px spike into the [1, 2)
    # bin and reads a full bin short of the markers. Anchor the curve on the exact
    # <= N points and extend with the histogram tail (offset to meet <= 3 px) so the
    # blue step passes through the red markers instead of lagging a bin behind them.
    thresholds = np.array([1.0, 2.0, 3.0])
    values = np.array([
        d["fraction_within_1px"],
        d["fraction_within_2px"],
        d["fraction_within_3px"],
    ])
    cum_hist = np.cumsum(counts) / total if total else np.zeros_like(counts)
    idx3 = int(np.searchsorted(right_edges, 3.0))
    tail = right_edges > 3.0
    offset = values[-1] - cum_hist[idx3]
    x_curve = np.r_[0.0, thresholds, right_edges[tail]]
    y_curve = np.r_[0.0, values, np.clip(cum_hist[tail] + offset, 0.0, 1.0)]

    fig, ax = plt.subplots(figsize=(4.15, 2.7))
    xmax = 6.0
    keep = x_curve <= xmax
    ax.step(x_curve[keep], y_curve[keep], where="post", color=COLORS["blue"], lw=1.5)
    ax.scatter(thresholds, values, s=18, color=COLORS["red"], zorder=3)
    for px, value in zip(thresholds, values):
        ax.annotate(
            f"≤{int(px)} px\n{value:.2f}",
            xy=(px, value),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=6,
            color=COLORS["red"],
        )
    ax.set_xlabel("Distance from FP pixel to nearest GT trail (px)", fontsize=7)
    ax.set_ylabel("Cumulative fraction of FP pixels", fontsize=7)
    ax.set_title("Cumulative FP distance to nearest GT trail", fontsize=8)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    fig.subplots_adjust(left=0.16, bottom=0.18, top=0.78)
    save_vector(fig, "ext_1_fp_distance_histogram")
    print("[ok]  ext_1_fp_distance_histogram")

def fig_hough_prob_stratified() -> None:
    if not HOUGH.exists():
        print(f"[skip] ext_3 stratified Hough: missing {HOUGH.name}")
        return
    d = load_json(HOUGH)
    labels = ["U-Net\nonly", "binary\nHough", "stratified\nHough"]
    recall = [d["pixel_recall_pre"], d["pixel_recall_binary_hough"], d["pixel_recall_stratified_hough"]]
    colours = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

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


def main() -> None:
    configure_style()
    fig_fp_distance_histogram()
    fig_hough_prob_stratified()


if __name__ == "__main__":
    main()
