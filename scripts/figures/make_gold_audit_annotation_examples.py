#!/usr/bin/env python
"""Render two blinded re-annotation crop/mask examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CROPS = ("c019.png", "c015.png")
CROP_DIR = Path("data/gold/audit_crops")
MASK_DIR = Path("data/gold/gold_masks")
OUT = Path("results/figures/supp_4_gold_audit_annotation_examples.pdf")


def load_luminance(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 5.6))

    for row, name in enumerate(CROPS):
        crop = load_luminance(CROP_DIR / name)
        mask = load_luminance(MASK_DIR / name)

        axes[row, 0].imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[row, 0].set_ylabel(f"Example {row + 1}", rotation=90, labelpad=10)

    axes[0, 0].set_title("Audit crop")
    axes[0, 1].set_title("Exported mask")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.02, hspace=0.08, wspace=0.04)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
