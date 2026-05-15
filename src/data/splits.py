"""Dataset split helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split

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


def create_splits(
    dataset: Dataset[Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset[Any], Subset[Any], Subset[Any]]:
    """
    Create deterministic train, validation, and test splits.

    Parameters
    ----------
    dataset:
        Dataset to split.
    train_ratio:
        Fraction reserved for training.
    val_ratio:
        Fraction reserved for validation.
    seed:
        Random seed used for deterministic splitting.
    """
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must define a valid split")

    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )
