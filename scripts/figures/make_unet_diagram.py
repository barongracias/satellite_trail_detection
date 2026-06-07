"""Render the architecture-faithful U-Net schematic for the thesis.

The diagram is kept self-contained so that Figure 4.1 can be regenerated without
extra plotting dependencies. Encoder widths, spatial sizes, dropout rates, and
operation labels match ``src/models/unet.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch, Rectangle


NAVY = "#17232D"
TEXT_GREY = "#59666F"
POOL_GREY = "#6F7B83"
ENCODER = "#1C8F8A"
DECODER = "#C98221"
BOTTLENECK = "#A54034"
SKIP = "#147A75"
HEAD = "#E6EDF2"

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "results" / "figures" / "unet_architecture.pdf"


@dataclass(frozen=True)
class Block:
    """Drawn feature-map block."""

    name: str
    channels: int
    spatial: int
    dropout: float
    x: float
    y: float
    family: str


SPATIAL_HEIGHT = {
    528: 1.08,
    264: 0.88,
    132: 0.72,
    66: 0.60,
    33: 0.50,
}

CHANNEL_WIDTH = {
    8: 0.92,
    16: 1.02,
    32: 1.14,
    64: 1.28,
    128: 1.42,
}

BLOCKS = {
    "enc0": Block("DoubleConv", 8, 528, 0.1, 2.0, 5.70, "encoder"),
    "enc1": Block("DoubleConv", 16, 264, 0.1, 2.0, 4.35, "encoder"),
    "enc2": Block("DoubleConv", 32, 132, 0.2, 2.0, 3.15, "encoder"),
    "enc3": Block("DoubleConv", 64, 66, 0.2, 2.0, 1.95, "encoder"),
    "bottleneck": Block("DoubleConv", 128, 33, 0.3, 6.65, 0.95, "bottleneck"),
    "dec3": Block("DoubleConv", 64, 66, 0.2, 10.95, 1.95, "decoder"),
    "dec2": Block("DoubleConv", 32, 132, 0.2, 10.95, 3.15, "decoder"),
    "dec1": Block("DoubleConv", 16, 264, 0.1, 10.95, 4.35, "decoder"),
    "dec0": Block("DoubleConv", 8, 528, 0.1, 10.95, 5.70, "decoder"),
}


def _mix_with_white(hex_colour: str, white_fraction: float) -> tuple[float, float, float]:
    """Return a lighter version of ``hex_colour``."""
    base = mcolors.to_rgb(hex_colour)
    return tuple((1.0 - white_fraction) * c + white_fraction for c in base)


def _block_colour(block: Block) -> tuple[float, float, float]:
    """Use darker colour for wider-channel feature maps."""
    if block.family == "bottleneck":
        base = BOTTLENECK
    elif block.family == "decoder":
        base = DECODER
    else:
        base = ENCODER

    # Channel range is fixed by src/models/unet.py: 8, 16, 32, 64, 128.
    channel_rank = {8: 0.72, 16: 0.62, 32: 0.50, 64: 0.38, 128: 0.26}[block.channels]
    return _mix_with_white(base, channel_rank)


def _dimensions(block: Block) -> tuple[float, float]:
    return CHANNEL_WIDTH[block.channels], SPATIAL_HEIGHT[block.spatial]


def _edge(block: Block, side: str) -> tuple[float, float]:
    width, height = _dimensions(block)
    if side == "left":
        return block.x - width / 2, block.y
    if side == "right":
        return block.x + width / 2 + 0.09, block.y + 0.09
    if side == "top":
        return block.x + 0.045, block.y + height / 2 + 0.09
    if side == "bottom":
        return block.x, block.y - height / 2
    raise ValueError(f"Unknown side: {side}")


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str,
    dashed: bool = False,
    lw: float = 1.35,
    mutation_scale: float = 12,
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color=colour,
            linestyle=(0, (4, 3)) if dashed else "-",
            linewidth=lw,
            mutation_scale=mutation_scale,
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def _draw_feature_block(ax: plt.Axes, block: Block) -> None:
    """Draw a small stack of feature-map sheets."""
    width, height = _dimensions(block)
    offset = 0.07
    fill = _block_colour(block)
    edge = NAVY

    for i in (2, 1, 0):
        rect = Rectangle(
            (block.x - width / 2 + i * offset, block.y - height / 2 + i * offset),
            width,
            height,
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.85,
            zorder=3 + (2 - i),
        )
        ax.add_patch(rect)

    if block.family == "bottleneck":
        label = f"{block.channels} ch\n${block.spatial}^2$, p={block.dropout:.1f}"
    else:
        label = f"{block.channels} ch\n${block.spatial}^2$\np={block.dropout:.1f}"
    ax.text(
        block.x,
        block.y + 0.02,
        label,
        ha="center",
        va="center",
        fontsize=7.1,
        color=NAVY,
        linespacing=0.95,
        zorder=8,
    )


def _draw_head(ax: plt.Axes) -> None:
    dec0 = BLOCKS["dec0"]
    head_x = 12.55
    head_width = 0.38
    head_height = SPATIAL_HEIGHT[528]

    _arrow(ax, _edge(dec0, "right"), (head_x - head_width / 2 - 0.08, dec0.y), colour=NAVY, lw=1.1)
    ax.add_patch(
        Rectangle(
            (head_x - head_width / 2, dec0.y - head_height / 2),
            head_width,
            head_height,
            facecolor=HEAD,
            edgecolor=NAVY,
            linewidth=0.85,
            zorder=6,
        )
    )
    ax.text(head_x, dec0.y, "$1\\times1$\nconv", ha="center", va="center", fontsize=7.0, color=NAVY, zorder=8)

    output_x = 13.35
    _arrow(ax, (head_x + head_width / 2 + 0.06, dec0.y), (output_x - 0.22, dec0.y), colour=NAVY, lw=1.1)
    ax.text(
        output_x,
        dec0.y + 0.10,
        "output logits\n$1\\times528^2$",
        ha="left",
        va="center",
        fontsize=8.3,
        color=NAVY,
        linespacing=1.08,
    )
    ax.text(
        output_x,
        dec0.y - 0.55,
        "sigmoid in\nloss/metrics",
        ha="left",
        va="center",
        fontsize=6.7,
        color=TEXT_GREY,
        linespacing=1.05,
    )


def _draw_operations(ax: plt.Axes) -> None:
    encoder_order = ["enc0", "enc1", "enc2", "enc3"]
    decoder_order = ["dec3", "dec2", "dec1", "dec0"]

    for upper, lower in zip(encoder_order[:-1], encoder_order[1:], strict=True):
        _arrow(ax, _edge(BLOCKS[upper], "bottom"), _edge(BLOCKS[lower], "top"), colour=POOL_GREY, zorder=9)
    _arrow(
        ax,
        _edge(BLOCKS["enc3"], "bottom"),
        (BLOCKS["bottleneck"].x - 0.54, BLOCKS["bottleneck"].y + 0.35),
        colour=POOL_GREY,
        zorder=9,
    )

    _arrow(
        ax,
        (BLOCKS["bottleneck"].x + 0.56, BLOCKS["bottleneck"].y + 0.35),
        _edge(BLOCKS["dec3"], "bottom"),
        colour=DECODER,
        zorder=9,
    )
    for lower, upper in zip(decoder_order[:-1], decoder_order[1:], strict=True):
        _arrow(ax, _edge(BLOCKS[lower], "top"), _edge(BLOCKS[upper], "bottom"), colour=DECODER, zorder=9)

    for enc, dec in (("enc0", "dec0"), ("enc1", "dec1"), ("enc2", "dec2"), ("enc3", "dec3")):
        _arrow(ax, _edge(BLOCKS[enc], "right"), _edge(BLOCKS[dec], "left"), colour=SKIP, dashed=True, lw=1.05)

    ax.text(6.55, 5.95, "skip connections concatenate matching encoder maps", ha="center", va="center", fontsize=7.5, color=SKIP)


def _draw_input(ax: plt.Axes) -> None:
    enc0 = BLOCKS["enc0"]
    input_x = 0.50
    ax.text(input_x, enc0.y + 0.15, "input\n$1\\times528^2$", ha="left", va="center", fontsize=8.3, color=NAVY)
    _arrow(ax, (input_x + 0.75, enc0.y), _edge(enc0, "left"), colour=NAVY, lw=1.1)


def _draw_legend(ax: plt.Axes) -> None:
    y = 6.78
    x = 3.15
    ax.text(x - 0.65, y, "Legend", ha="right", va="center", fontsize=8.5, color=NAVY, fontweight="bold")

    ax.add_patch(Rectangle((x, y - 0.11), 0.22, 0.22, facecolor=_mix_with_white(ENCODER, 0.50), edgecolor=NAVY, linewidth=0.75))
    ax.text(x + 0.30, y, "DoubleConv", ha="left", va="center", fontsize=7.5, color=NAVY)

    x2 = x + 1.75
    _arrow(ax, (x2, y), (x2 + 0.42, y), colour=POOL_GREY, lw=1.1, mutation_scale=10)
    ax.text(x2 + 0.52, y, "AvgPool down", ha="left", va="center", fontsize=7.5, color=NAVY)

    x3 = x2 + 2.15
    _arrow(ax, (x3, y), (x3 + 0.42, y), colour=DECODER, lw=1.1, mutation_scale=10)
    ax.text(x3 + 0.52, y, "ConvTranspose up", ha="left", va="center", fontsize=7.5, color=NAVY)

    x4 = x3 + 2.65
    _arrow(ax, (x4, y), (x4 + 0.42, y), colour=SKIP, dashed=True, lw=1.1, mutation_scale=10)
    ax.text(x4 + 0.52, y, "skip concat", ha="left", va="center", fontsize=7.5, color=NAVY)


def render() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    ax.set_xlim(0.05, 14.05)
    ax.set_ylim(0.10, 7.15)
    ax.axis("off")

    _draw_legend(ax)
    _draw_input(ax)
    _draw_operations(ax)

    for key in ("enc0", "enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1", "dec0"):
        _draw_feature_block(ax, BLOCKS[key])
    _draw_head(ax)

    ax.text(6.65, 1.58, "bottleneck", ha="center", va="center", fontsize=8.2, color=TEXT_GREY)
    ax.text(
        6.65,
        0.24,
        "DoubleConv = Conv3x3 -> LeakyReLU(0.3) -> Dropout(p) -> Conv3x3 -> LeakyReLU(0.3); "
        "no BatchNorm; 485,673 trainable parameters.",
        ha="center",
        va="center",
        fontsize=7.4,
        color=TEXT_GREY,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    render()
