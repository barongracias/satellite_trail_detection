"""Dataset split helpers."""

from __future__ import annotations

import random

import numpy as np

from src.data.indexing import ProcessedImagePair


def create_image_level_splits(
    image_pairs: list[ProcessedImagePair],
    trail_pixel_counts: list[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 2804,
) -> tuple[list[ProcessedImagePair], list[ProcessedImagePair], list[ProcessedImagePair]]:
    """Split image pairs into train/val/test stratified by trail-pixel-count quartile.

    Images are binned into four quartile groups by their trail pixel count, then
    each group is independently shuffled and split proportionally.  This spreads
    heavy-trail and empty images evenly across the three partitions.

    Note on realized fractions: ``n_train`` and ``n_val`` are floored per quartile
    (``int(n * ratio)``) and test takes the remainder, so flooring across four
    groups under-allocates train/val and the test set runs slightly over nominal.
    For the 178-image MeerLICHT subset the realized split at the default 0.70/0.15
    ratios is 122/24/32 (0.685/0.135/0.180), not an exact 70/15/15. This is
    leakage-safe and conservative (a larger test set widens CIs); the methods
    section should quote the realized 122/24/32 counts rather than the nominal
    ratios.

    Parameters
    ----------
    image_pairs:
        All image/mask pairs to partition.
    trail_pixel_counts:
        Trail pixel count for each pair (same order).
    train_ratio:
        Fraction of images reserved for training.
    val_ratio:
        Fraction of images reserved for validation.
    seed:
        Controls the shuffle within each quartile group.

    Returns
    -------
    tuple[list, list, list]
        train, val, and test lists of ProcessedImagePair.
    """
    if len(image_pairs) != len(trail_pixel_counts):
        raise ValueError("image_pairs and trail_pixel_counts must have the same length")
    if not image_pairs:
        raise ValueError("image_pairs must not be empty")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must define a valid three-way split")

    counts = np.array(trail_pixel_counts, dtype=float)
    q25, q50, q75 = np.percentile(counts, [25, 50, 75])

    def _quartile(c: float) -> int:
        if c <= q25:
            return 0
        if c <= q50:
            return 1
        if c <= q75:
            return 2
        return 3

    groups: dict[int, list[ProcessedImagePair]] = {0: [], 1: [], 2: [], 3: []}
    for pair, count in zip(image_pairs, trail_pixel_counts):
        groups[_quartile(float(count))].append(pair)

    rng = random.Random(seed)
    train_pairs: list[ProcessedImagePair] = []
    val_pairs: list[ProcessedImagePair] = []
    test_pairs: list[ProcessedImagePair] = []

    for q in range(4):
        # Sort deterministically before shuffle so the seed fully determines the result.
        group = sorted(groups[q], key=lambda p: str(p.image_path))
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_pairs.extend(group[:n_train])
        val_pairs.extend(group[n_train : n_train + n_val])
        test_pairs.extend(group[n_train + n_val :])

    return train_pairs, val_pairs, test_pairs
