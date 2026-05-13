"""Dataset split helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset, Subset, random_split


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
