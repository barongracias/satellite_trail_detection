import pytest


torch = pytest.importorskip("torch")

from src.data.splits import create_splits  # noqa: E402


def test_create_splits_is_reproducible_for_a_fixed_seed() -> None:
    dataset = list(range(20))

    split_a = create_splits(dataset, train_ratio=0.6, val_ratio=0.2, seed=7)
    split_b = create_splits(dataset, train_ratio=0.6, val_ratio=0.2, seed=7)

    assert split_a[0].indices == split_b[0].indices
    assert split_a[1].indices == split_b[1].indices
    assert split_a[2].indices == split_b[2].indices


def test_create_splits_rejects_invalid_ratios() -> None:
    dataset = list(range(10))

    with pytest.raises(ValueError):
        create_splits(dataset, train_ratio=0.8, val_ratio=0.3)

    with pytest.raises(ValueError):
        create_splits(dataset, train_ratio=0.0, val_ratio=0.2)
