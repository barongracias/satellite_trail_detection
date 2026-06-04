#!/usr/bin/env python
"""Qualitative Hough gap-bridge figure for the locked winner.

Post-hoc mechanism illustration only. Two predeclared positive test images are
shown where the U-Net left whole trail patches undetected and the Hough stage
bridged them (patch-FN reduction 5->0 and 2->0). For each image a local crop is
shown three ways: raw, ground truth with the U-Net prediction, and the GT pixels
recovered only by Hough. The locked model, threshold, and Hough settings are
unchanged; this is not a selection claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
from scripts.make_thesis_figures import configure_style, save_vector

CLASSICAL = Path("results/classical")

# Predeclared examples chosen by patch-FN reduction: whole trail patches the
# U-Net missed entirely that the Hough stage then recovered. Hand-picked from an
# author-reviewed candidate sheet; this is an illustration, not an automatic
# selection claim.
PREDECLARED = [
    {"stem": "ML1_20220629_171546", "note": "U-Net missed 5 patches", "patch_fn_pre": 5, "patch_fn_post": 0},
]

GREEN = (0.0, 0.62, 0.45)
MAGENTA = (0.80, 0.18, 0.58)
ORANGE = (0.98, 0.45, 0.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=LOCKED_CHECKPOINT)
    p.add_argument("--patch-dir", default="data/patches")
    p.add_argument("--threshold", type=float, default=LOCKED_THRESHOLD)
    p.add_argument("--out-json", default=str(CLASSICAL / "hough_gap_bridge_example_t44_s2804.json"))
    p.add_argument("--crop-margin", type=int, default=150)
    return p.parse_args()


def _overlay(raw: np.ndarray, masks: list[tuple[np.ndarray, tuple[float, float, float], float]]) -> np.ndarray:
    base = raw.astype(float) / 255.0
    rgb = np.dstack([base, base, base])
    for mask, color, alpha in masks:
        m = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        for channel in range(3):
            rgb[..., channel] = np.where(m, (1 - alpha) * rgb[..., channel] + alpha * color[channel], rgb[..., channel])
    return np.clip(rgb, 0.0, 1.0)


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _recovered_crop(recovered: np.ndarray, fallback: np.ndarray, raw_shape: tuple[int, int], margin: int) -> tuple[slice, slice]:
    # Centre on the fattest bridged stretch: erode the recovered-GT mask to drop thin
    # boundary-thickening slivers, keep the solid gap, and take its largest component.
    seed = cv2.erode(recovered.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    comp = _largest_component_mask(seed if seed.any() else (recovered if recovered.any() else fallback))
    yy, xx = np.nonzero(comp)
    h, w = raw_shape
    if yy.size == 0:
        return slice(0, h), slice(0, w)
    y0 = max(0, int(yy.min()) - margin)
    y1 = min(h, int(yy.max()) + margin)
    x0 = max(0, int(xx.min()) - margin)
    x1 = min(w, int(xx.max()) + margin)
    return slice(y0, y1), slice(x0, x1)


def _match_source(stem: str, groups: dict[str, Any]) -> str | None:
    for src in groups:
        if stem in Path(src).name:
            return src
    return None


def main() -> None:
    args = parse_args()
    configure_style()
    test_df = read_test_manifest(args.patch_dir)
    groups = {src: group.reset_index(drop=True) for src, group in iter_positive_source_groups(test_df)}
    model, normalisation, device = load_locked_model(args.checkpoint)

    fig, axes = plt.subplots(len(PREDECLARED), 3, figsize=(8.1, 2.75 * len(PREDECLARED)), squeeze=False)
    col_titles = ["Raw local crop", "Ground truth (green) + U-Net (magenta)", "U-Net (magenta) + Hough (orange)"]
    records: list[dict[str, Any]] = []

    for r, item in enumerate(PREDECLARED):
        src = _match_source(item["stem"], groups)
        if src is None:
            raise RuntimeError(f"Predeclared image {item['stem']} not found among positive test groups")
        canvases = reconstruct_locked_canvases(
            src, groups[src], model, device, normalisation, threshold=args.threshold,
        )
        target = np.logical_and(canvases.target_canvas, canvases.support_canvas)
        recovered = np.logical_and.reduce((canvases.hough_canvas, ~canvases.binary_canvas, target))
        sl_y, sl_x = _recovered_crop(recovered, target, canvases.raw_image.shape, args.crop_margin)
        raw = canvases.raw_image[sl_y, sl_x]
        gt = target[sl_y, sl_x]
        binary = canvases.binary_canvas[sl_y, sl_x]
        rec = recovered[sl_y, sl_x]
        # Wider GT band so the ground truth is visible along the whole trail (the gap
        # then reads as the green-only stretch the U-Net missed).
        gt_band = cv2.dilate(gt.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0

        panels = [
            (raw.astype(float) / 255.0, "gray"),
            (_overlay(raw, [(gt_band, GREEN, 0.5), (binary, MAGENTA, 0.85)]), None),
            (_overlay(raw, [(binary, MAGENTA, 0.55), (rec, ORANGE, 0.95)]), None),
        ]
        for c, (image, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(image, cmap=cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=7.8)
        axes[r, 0].set_ylabel(f"{item['stem']}\n{item['note']}", fontsize=7.5)

        records.append({
            "source_image": src,
            "stem": item["stem"],
            "note": item["note"],
            "patch_fn_pre": item["patch_fn_pre"],
            "patch_fn_post": item["patch_fn_post"],
            "hough_recovered_gt_px": int(recovered.sum()),
            "crop_y0": sl_y.start, "crop_y1": sl_y.stop,
            "crop_x0": sl_x.start, "crop_x1": sl_x.stop,
        })

    fig.suptitle(
        "Hough bridging a U-Net gap",
        fontsize=10.5, y=0.97,
    )
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.02, top=0.82, hspace=0.08, wspace=0.07)
    save_vector(fig, "hough_gap_bridge_example")

    payload = {
        "analysis": "hough_gap_bridge_example",
        "selection_rule": (
            "Two predeclared positive test images chosen by patch-FN reduction "
            "(whole trail patches the U-Net missed that the Hough stage recovered), "
            "hand-picked from an author-reviewed candidate sheet."
        ),
        "method_note": (
            "Qualitative post-hoc mechanism illustration only; not selection evidence. "
            "Locked model, threshold, and Hough settings unchanged."
        ),
        "provenance": provenance(
            checkpoint=args.checkpoint,
            threshold=args.threshold,
            normalisation=normalisation,
            patch_dir=args.patch_dir,
        ),
        "examples": records,
    }
    write_json(args.out_json, payload)
    print(f"Saved {args.out_json}")
    for rec in records:
        print(f"  {rec['stem']}: recovered {rec['hough_recovered_gt_px']} px, patch FN {rec['patch_fn_pre']}->{rec['patch_fn_post']}")


if __name__ == "__main__":
    main()
