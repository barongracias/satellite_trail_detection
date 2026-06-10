"""Render the data-flow / split schematic for Chapter 2 of the thesis.

One self-contained matplotlib figure showing the path from the 178 labelled
frames to the evaluation manifests: image-level quartile-stratified split,
non-overlapping 528-pixel patch grid, 1:3 sampled manifests, and the
all-negative parity test manifest. Counts match the CSD3-verified manifests
recorded in the dataset-composition table (Chapter 2). Interpretation lives
in the thesis caption, not in on-figure footers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY = "#17232D"
TEXT_GREY = "#59666F"
ENCODER = "#1C8F8A"
DECODER = "#C98221"
BOTTLENECK = "#A54034"

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "results" / "figures" / "data_split_schematic.pdf"


def _box(ax, x, y, w, h, title, lines, face):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02",
            facecolor=face, edgecolor=NAVY, linewidth=1.2,
        )
    )
    ax.text(x + w / 2, y + h - 0.09, title, ha="center", va="top",
            fontsize=9, fontweight="bold", color="white")
    if lines:
        ax.text(x + w / 2, y + (h - 0.34) / 2, "\n".join(lines), ha="center",
                va="center", fontsize=8, color="white", linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.2, color=TEXT_GREY, shrinkA=3, shrinkB=3,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    row_y = {"train": 3.05, "val": 1.85, "test": 0.65}
    box_h = 0.78

    # Source corpus.
    _box(ax, 0.15, 1.45, 2.05, 1.45, "178 image–mask pairs",
         ["10,560 × 10,560 px", "8-bit PNG renders", "hand-annotated masks"],
         face=NAVY)

    # Image-level split.
    split_x, split_w = 3.45, 2.05
    _box(ax, split_x, row_y["train"], split_w, box_h, "Train", ["122 images (0.685)"], face=ENCODER)
    _box(ax, split_x, row_y["val"], split_w, box_h, "Validation", ["24 images (0.135)"], face=ENCODER)
    _box(ax, split_x, row_y["test"], split_w, box_h, "Test", ["32 images (0.180)"], face=ENCODER)

    for key in row_y:
        _arrow(ax, 2.25, 2.18, split_x - 0.06, row_y[key] + box_h / 2)
    ax.text(2.58, 3.78, "image-level split,\nstratified by quartile",
            ha="center", va="center", fontsize=7.5, color=TEXT_GREY, style="italic")

    # Sampled manifests.
    samp_x, samp_w = 6.55, 2.25
    _box(ax, samp_x, row_y["train"], samp_w, box_h, "Sampled train",
         ["13,192 patches", "3,298 pos + 9,894 neg"], face=DECODER)
    _box(ax, samp_x, row_y["val"], samp_w, box_h, "Sampled validation",
         ["2,456 patches", "614 pos + 1,842 neg"], face=DECODER)
    _box(ax, samp_x, row_y["test"], samp_w, box_h, "Sampled test",
         ["3,488 patches", "872 pos + 2,616 neg"], face=DECODER)

    for key in row_y:
        _arrow(ax, split_x + split_w + 0.02, row_y[key] + box_h / 2,
               samp_x - 0.06, row_y[key] + box_h / 2)
    ax.text(6.02, 2.84, "528 × 528 patches,\n1:3 pos:neg sampling",
            ha="center", va="center", fontsize=7.5, color=TEXT_GREY, style="italic")

    # Parity manifest (from the test split, bypassing sampling).
    par_x, par_w = 8.95, 1.70
    _box(ax, par_x, row_y["test"], par_w, box_h + 0.14, "Parity test",
         ["12,800 patches", "872 pos + 11,928 neg"], face=BOTTLENECK)
    _arrow(ax, split_x + split_w + 0.02, row_y["test"] + 0.05,
           par_x - 0.04, row_y["test"] + 0.07, rad=0.30)
    ax.text(7.25, 0.05, "all negatives kept", ha="center", va="center",
            fontsize=7.5, color=TEXT_GREY, style="italic")

    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
