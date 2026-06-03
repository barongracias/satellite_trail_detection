#!/usr/bin/env python
"""Qualitative Hough gap-bridge figure for the locked winner.

This is a post-hoc mechanism illustration only. It ranks positive test images
by Hough-added GT coverage and plots the first candidate with an interior gap,
without changing the locked model, threshold, or Hough settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from scripts._locked_winner_canvases import (
    LOCKED_CHECKPOINT,
    LOCKED_THRESHOLD,
    iter_positive_source_groups,
    load_locked_model,
    provenance,
    read_test_manifest,
    reconstruct_locked_canvases,
    write_json,
)
from scripts.faint_streak_analysis import _bbox_from_mask, component_axis
from scripts.make_thesis_figures import COLORS, configure_style, save_vector

CLASSICAL = Path("results/classical")
HOUGH_JSON = CLASSICAL / "hough_postprocess_winner_t44_s2804.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=LOCKED_CHECKPOINT)
    p.add_argument("--patch-dir", default="data/patches")
    p.add_argument("--threshold", type=float, default=LOCKED_THRESHOLD)
    p.add_argument("--hough-json", default=str(HOUGH_JSON))
    p.add_argument("--out-json", default=str(CLASSICAL / "hough_gap_bridge_example_t44_s2804.json"))
    p.add_argument("--min-recovered-gt-px", type=int, default=500)
    p.add_argument("--min-interior-fraction", type=float, default=0.25)
    p.add_argument("--crop-margin", type=int, default=120)
    p.add_argument("--max-candidates", type=int, default=10)
    return p.parse_args()


def rank_hough_candidates(per_image: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for row in per_image:
        if not row.get("is_positive"):
            continue
        recovered = int(row.get("pixels_covered_post", 0)) - int(row.get("pixels_covered_pre", 0))
        new = dict(row)
        new["hough_recovered_gt_px"] = recovered
        ranked.append(new)
    ranked.sort(key=lambda r: r["hough_recovered_gt_px"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank_by_hough_recovered_gt_px"] = idx
    return ranked


def _interior_gap_mask(
    target: np.ndarray,
    recovered: np.ndarray,
    *,
    min_interior_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_labels, labels = cv2.connectedComponents(target.astype(np.uint8), connectivity=8)
    best_mask = np.zeros_like(target, dtype=bool)
    best_info: dict[str, Any] = {
        "criterion": "no_component_with_recovered_gt",
        "recovered_gt_px": 0,
        "interior_fraction": 0.0,
    }
    for label in range(1, n_labels):
        comp = labels == label
        rec = np.logical_and(recovered, comp)
        rec_n = int(rec.sum())
        if rec_n == 0:
            continue
        coords = np.argwhere(comp)
        rec_coords = np.argwhere(rec)
        centre, axis, _ = component_axis(coords)
        comp_proj = (coords.astype(float) - centre) @ axis
        rec_proj = (rec_coords.astype(float) - centre) @ axis
        lo, hi = np.percentile(comp_proj, [15, 85]) if comp_proj.size >= 4 else (comp_proj.min(), comp_proj.max())
        interior = np.logical_and(rec_proj >= lo, rec_proj <= hi)
        interior_fraction = float(interior.mean()) if interior.size else 0.0
        if rec_n > best_info["recovered_gt_px"]:
            best_mask = rec
            best_info = {
                "criterion": "interior_gap" if interior_fraction >= min_interior_fraction else "recovered_gt_fallback",
                "recovered_gt_px": rec_n,
                "interior_fraction": interior_fraction,
                "component_gt_px": int(comp.sum()),
            }
    return best_mask, best_info


def _overlay(raw: np.ndarray, masks: list[tuple[np.ndarray, tuple[float, float, float], float]]) -> np.ndarray:
    base = raw.astype(float) / 255.0
    rgb = np.dstack([base, base, base])
    for mask, color, alpha in masks:
        for channel in range(3):
            rgb[..., channel] = np.where(mask, (1 - alpha) * rgb[..., channel] + alpha * color[channel], rgb[..., channel])
    return np.clip(rgb, 0.0, 1.0)


def _local_gap_crop(
    gap_mask: np.ndarray,
    fallback_mask: np.ndarray,
    raw_shape: tuple[int, int],
    half_size: int,
) -> tuple[slice, slice, tuple[int, int]]:
    mask = gap_mask if gap_mask.any() else fallback_mask
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return _bbox_from_mask(mask, half_size, raw_shape)[0], _bbox_from_mask(mask, half_size, raw_shape)[1], (0, 0)
    cy = int(np.median(yy))
    cx = int(np.median(xx))
    h, w = raw_shape
    y0 = max(0, cy - half_size)
    y1 = min(h, cy + half_size)
    x0 = max(0, cx - half_size)
    x1 = min(w, cx + half_size)
    if y1 - y0 < 2 * half_size:
        y0 = max(0, y1 - 2 * half_size)
        y1 = min(h, y0 + 2 * half_size)
    if x1 - x0 < 2 * half_size:
        x0 = max(0, x1 - 2 * half_size)
        x1 = min(w, x0 + 2 * half_size)
    return slice(y0, y1), slice(x0, x1), (cy - y0, cx - x0)


def _make_figure(canvases, gap_mask: np.ndarray, crop_margin: int) -> tuple[tuple[slice, slice], dict[str, int]]:
    fallback = np.logical_and(canvases.hough_canvas, canvases.target_canvas)
    half_size = max(520, crop_margin * 3)
    sl_y, sl_x, local_center = _local_gap_crop(gap_mask, fallback, canvases.raw_image.shape, half_size)
    raw = canvases.raw_image[sl_y, sl_x]
    gt = canvases.target_canvas[sl_y, sl_x]
    binary = canvases.binary_canvas[sl_y, sl_x]
    hough = canvases.hough_canvas[sl_y, sl_x]
    added = np.logical_and(hough, ~binary)
    gap_crop = gap_mask[sl_y, sl_x]

    panels = [
        (raw.astype(float) / 255.0, "Raw local crop", "gray"),
        (_overlay(raw, [(gt, (0.0, 0.62, 0.45), 0.70), (binary, (0.80, 0.18, 0.58), 0.70)]),
         "GT (green) + U-Net (magenta)", None),
        (_overlay(raw, [(binary, (0.80, 0.18, 0.58), 0.60), (added, (0.9, 0.35, 0.0), 0.88)]),
         "Hough-added pixels (orange)", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.75))
    for ax, (image, title, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    if gap_crop.any():
        cy, cx = local_center
        box = 90
        x0 = max(0, cx - box // 2)
        y0 = max(0, cy - box // 2)
        for ax in axes:
            ax.add_patch(Rectangle((x0, y0), box, box, fill=False, edgecolor=COLORS["red"], lw=1.0))
        axes[-1].annotate(
            "bridged gap", xy=(cx, cy), xytext=(0.62, 0.12), textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["red"]},
            color=COLORS["red"], fontsize=7,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": COLORS["red"], "lw": 0.5, "alpha": 0.88},
        )
    fig.subplots_adjust(bottom=0.18, top=0.82, wspace=0.06)
    fig.text(
        0.5, 0.04,
        "Local crop from the rank-1 recovered-GT candidate; orange marks Hough additions not present in U-Net binary. Mechanism illustration only.",
        ha="center", fontsize=6, color="#555555",
    )
    save_vector(fig, "hough_gap_bridge_example")
    return (sl_y, sl_x), {
        "crop_y0": sl_y.start, "crop_y1": sl_y.stop,
        "crop_x0": sl_x.start, "crop_x1": sl_x.stop,
        "local_gap_y": int(local_center[0]), "local_gap_x": int(local_center[1]),
        "display_crop_half_size": int(half_size),
    }

def main() -> None:
    args = parse_args()
    configure_style()
    hough_data = json.loads(Path(args.hough_json).read_text())
    ranked = rank_hough_candidates(hough_data["per_image"])
    test_df = read_test_manifest(args.patch_dir)
    groups = {src: group.reset_index(drop=True) for src, group in iter_positive_source_groups(test_df)}
    model, normalisation, device = load_locked_model(args.checkpoint)
    selected = None
    selected_canvases = None
    selected_gap = None
    selected_info = None
    for row in ranked[: args.max_candidates]:
        if row["hough_recovered_gt_px"] < args.min_recovered_gt_px:
            continue
        source_image = row["source_image"]
        if source_image not in groups:
            continue
        print(f"Trying rank {row['rank_by_hough_recovered_gt_px']}: {Path(source_image).name}", flush=True)
        canvases = reconstruct_locked_canvases(
            source_image, groups[source_image], model, device, normalisation, threshold=args.threshold,
        )
        recovered = np.logical_and.reduce((canvases.hough_canvas, ~canvases.binary_canvas,
                                           canvases.target_canvas, canvases.support_canvas))
        gap_mask, info = _interior_gap_mask(
            np.logical_and(canvases.target_canvas, canvases.support_canvas), recovered,
            min_interior_fraction=args.min_interior_fraction,
        )
        if info["recovered_gt_px"] >= args.min_recovered_gt_px and info["criterion"] == "interior_gap":
            selected, selected_canvases, selected_gap, selected_info = row, canvases, gap_mask, info
            break
        if selected is None and info["recovered_gt_px"] >= args.min_recovered_gt_px:
            selected, selected_canvases, selected_gap, selected_info = row, canvases, gap_mask, info
    if selected is None or selected_canvases is None or selected_gap is None or selected_info is None:
        raise RuntimeError("No Hough gap candidate met the recovered-GT threshold.")
    _, crop = _make_figure(selected_canvases, selected_gap, args.crop_margin)
    payload = {
        "analysis": "hough_gap_bridge_example",
        "provenance": provenance(
            checkpoint=args.checkpoint,
            threshold=args.threshold,
            normalisation=normalisation,
            patch_dir=args.patch_dir,
        ),
        "method_note": (
            "Candidate selected post hoc by Hough-added GT pixels and an interior-gap criterion. "
            "Qualitative mechanism illustration only; not selection evidence."
        ),
        "selected": {
            **selected,
            **selected_info,
            **crop,
        },
    }
    write_json(args.out_json, payload)
    print(f"Saved {args.out_json}")
    print(f"Selected rank {selected['rank_by_hough_recovered_gt_px']}: {Path(selected['source_image']).name}")


if __name__ == "__main__":
    main()
