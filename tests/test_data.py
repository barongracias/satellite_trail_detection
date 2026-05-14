"""Tests for the data layer: catalogue, dataset, indexing, and splits."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.catalog import SatelliteCatalog
from src.data.indexing import (
    PatchDatasetConfig,
    discover_image_mask_pairs,
    parse_observation_datetime,
)


torch = pytest.importorskip("torch")

from src.data.dataset import SatelliteTrailPatchDataset  # noqa: E402
from src.data.splits import create_splits  # noqa: E402


def _write_pair(
    root_dir: Path,
    stem: str,
    image_array: np.ndarray,
    mask_array: np.ndarray,
) -> None:
    Image.fromarray(image_array).save(root_dir / f"{stem}_red.fits_full.png")
    Image.fromarray(mask_array).save(root_dir / f"{stem}_red_mask.png")


def test_satellite_catalog_summary_methods(tmp_path: Path) -> None:
    csv_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "Length": [10.0, 20.0, 30.0],
            "Start_RA": [1.0, 2.0, 3.0],
            "End_RA": [1.5, 2.5, 3.5],
            "Start_DEC": [-1.0, 0.0, 1.0],
            "End_DEC": [-0.5, 0.5, 1.5],
            "Satellite_Name": ["A", "B", "A"],
        }
    ).to_csv(csv_path, index=False)

    catalog = SatelliteCatalog(csv_path)

    assert catalog.get_unique_satellites() == ["A", "B"]
    assert catalog.get_length_summary()["median"] == 20.0
    assert catalog.get_ra_dec_ranges() == {"ra": (1.0, 3.5), "dec": (-1.0, 1.5)}
    assert catalog.get_metadata_summary()["rows"] == 3


def test_parse_observation_datetime_recognises_meerlicht_filenames() -> None:
    parsed = parse_observation_datetime("ML1_20190602_181454_red.fits_full.png")
    assert parsed == datetime(2019, 6, 2, 18, 14, 54)

    assert parse_observation_datetime("unexpected_name.png") is None


def test_discover_image_pairs_validates_sizes_and_missing_masks(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
        processed_dir / "A_red.fits_full.png"
    )

    with pytest.raises(FileNotFoundError):
        discover_image_mask_pairs(
            PatchDatasetConfig(
                root_dir=processed_dir,
                patch_size=2,
                stride=2,
                strict_pairing=True,
            )
        )

    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
        processed_dir / "A_red_mask.png"
    )
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
        processed_dir / "B_red.fits_full.png"
    )
    Image.fromarray(np.zeros((5, 4), dtype=np.uint8)).save(
        processed_dir / "B_red_mask.png"
    )

    with pytest.raises(ValueError):
        discover_image_mask_pairs(
            PatchDatasetConfig(
                root_dir=processed_dir,
                patch_size=2,
                stride=2,
                strict_pairing=True,
            )
        )


def test_dataset_config_validates_root_and_patch_parameters(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PatchDatasetConfig(root_dir=tmp_path / "missing")

    with pytest.raises(ValueError):
        PatchDatasetConfig(root_dir=tmp_path, patch_size=0)

    with pytest.raises(ValueError):
        PatchDatasetConfig(root_dir=tmp_path, patch_size=2, stride=0)


def test_dataset_builds_deterministic_patch_index_and_bookkeeping(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    image_array = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask_array = np.array(
        [
            [255, 0, 0, 0],
            [255, 0, 0, 0],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )
    _write_pair(processed_dir, "A", image_array, mask_array)

    dataset = SatelliteTrailPatchDataset(
        root_dir=processed_dir,
        patch_size=2,
        stride=2,
    )

    assert len(dataset) == 4

    records = dataset.get_patch_records()
    assert (records[0].y, records[0].x) == (0, 0)
    assert (records[-1].y, records[-1].x) == (2, 2)

    sample = dataset[3]
    assert tuple(sample["coords"].tolist()) == (2, 2)
    assert tuple(sample["grid_coords"].tolist()) == (1, 1)
    assert np.array_equal(np.asarray(sample["image"]), image_array[2:4, 2:4])
    assert np.array_equal(np.asarray(sample["mask"]), mask_array[2:4, 2:4])


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
