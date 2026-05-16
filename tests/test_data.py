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

from src.data.dataset import SatelliteTrailPatchDataset, PatchDirectoryDataset  # noqa: E402
from src.data.splits import create_splits, create_image_level_splits  # noqa: E402
from src.data.transforms import get_train_transforms, get_eval_transforms  # noqa: E402


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


def _make_fake_pairs(
    tmp_path: Path, n: int
) -> tuple[list, list]:
    """Create n dummy ProcessedImagePair objects and associated trail pixel counts."""
    from src.data.indexing import ProcessedImagePair

    pairs = []
    counts = []
    for i in range(n):
        img = Image.fromarray(np.zeros((4, 4), dtype=np.uint8))
        msk = Image.fromarray(np.zeros((4, 4), dtype=np.uint8))
        img_path = tmp_path / f"img_{i:03d}_red.fits_full.png"
        msk_path = tmp_path / f"img_{i:03d}_red_mask.png"
        img.save(img_path)
        msk.save(msk_path)
        pairs.append(
            ProcessedImagePair(image_path=img_path, mask_path=msk_path, width=4, height=4)
        )
        counts.append(i * 10)  # 0, 10, 20, …  — spread across all quartiles
    return pairs, counts


def test_create_image_level_splits_no_overlap_and_reproducible(tmp_path: Path) -> None:
    pairs, counts = _make_fake_pairs(tmp_path, 20)

    train_a, val_a, test_a = create_image_level_splits(pairs, counts, seed=42)
    train_b, val_b, test_b = create_image_level_splits(pairs, counts, seed=42)

    # Reproducibility
    assert [p.image_path for p in train_a] == [p.image_path for p in train_b]
    assert [p.image_path for p in val_a] == [p.image_path for p in val_b]
    assert [p.image_path for p in test_a] == [p.image_path for p in test_b]

    # No overlap between splits
    all_paths = (
        [p.image_path for p in train_a]
        + [p.image_path for p in val_a]
        + [p.image_path for p in test_a]
    )
    assert len(all_paths) == len(set(all_paths)), "Same image appeared in two splits"


def test_get_train_transforms_applies_identical_geometry_to_image_and_mask() -> None:
    # Asymmetric pattern — a single bright pixel at top-left so any flip or
    # rotation moves it to a different corner that is detectable.
    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[0, 0] = 255
    img_pil = Image.fromarray(arr, mode="L")
    mask_pil = Image.fromarray(arr.copy(), mode="L")

    transform = get_train_transforms()

    # Run many iterations to exercise all four rotations and flip combinations.
    for _ in range(40):
        img_t, mask_t = transform(img_pil, mask_pil)
        img_nonzero = (img_t.squeeze() > 0).numpy()
        mask_nonzero = (mask_t.squeeze() > 0).numpy()
        assert np.array_equal(img_nonzero, mask_nonzero), (
            "Image and mask received different geometric transforms"
        )


def test_patch_directory_dataset_returns_correct_dict_format(tmp_path: Path) -> None:
    import csv

    # Create synthetic patch files and a manifest CSV directly.
    train_dir = tmp_path / "patches" / "train"
    train_dir.mkdir(parents=True)

    img_array = np.full((8, 8), 128, dtype=np.uint8)
    msk_array = np.zeros((8, 8), dtype=np.uint8)
    msk_array[0, 0] = 255

    img_path = train_dir / "patch0_image.png"
    msk_path = train_dir / "patch0_mask.png"
    Image.fromarray(img_array).save(img_path)
    Image.fromarray(msk_array).save(msk_path)

    manifest_path = tmp_path / "patches" / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "source_image", "patch_path", "mask_path",
                "positive_pixel_fraction", "pos_weight",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "split": "train",
            "source_image": "fake.png",
            "patch_path": str(img_path),
            "mask_path": str(msk_path),
            "positive_pixel_fraction": 1.0 / 64,
            "pos_weight": 63.0,
        })

    dataset = PatchDirectoryDataset(train_dir, manifest_path)
    assert len(dataset) == 1

    sample = dataset[0]
    assert set(sample.keys()) >= {"image", "mask", "coords"}
    assert sample["image"].shape[0] == 1  # grayscale channel
    assert sample["mask"].shape[0] == 1
    assert sample["image"].shape[1:] == sample["mask"].shape[1:]
    assert sample["coords"].shape == (2,)


def test_patch_directory_dataset_per_image_normalisation_differs_from_fixed(
    tmp_path: Path,
) -> None:
    import csv

    train_dir = tmp_path / "patches" / "train"
    train_dir.mkdir(parents=True)

    img_array = np.arange(64, dtype=np.uint8).reshape(8, 8)
    msk_array = np.zeros((8, 8), dtype=np.uint8)
    img_path = train_dir / "patch0_image.png"
    msk_path = train_dir / "patch0_mask.png"
    Image.fromarray(img_array).save(img_path)
    Image.fromarray(msk_array).save(msk_path)

    manifest_path = tmp_path / "patches" / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "source_image", "patch_path", "mask_path",
                        "positive_pixel_fraction", "pos_weight"],
        )
        writer.writeheader()
        writer.writerow({
            "split": "train", "source_image": "fake.png",
            "patch_path": str(img_path), "mask_path": str(msk_path),
            "positive_pixel_fraction": 0.0, "pos_weight": 1.0,
        })

    ds_fixed = PatchDirectoryDataset(train_dir, manifest_path, normalisation="fixed")
    ds_per = PatchDirectoryDataset(train_dir, manifest_path, normalisation="per_image")

    img_fixed = ds_fixed[0]["image"]
    img_per = ds_per[0]["image"]

    assert not torch.allclose(img_fixed, img_per)
    assert abs(img_per.mean().item()) < 1e-4


def test_get_eval_transforms_returns_tensor_with_no_augmentation() -> None:
    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[0, 0] = 200
    img_pil = Image.fromarray(arr, mode="L")
    mask_pil = Image.fromarray(arr.copy(), mode="L")

    transform = get_eval_transforms()
    img_t, mask_t = transform(img_pil, mask_pil)

    # No rotation or flip — bright pixel must remain at top-left (0, 0).
    assert img_t[0, 0, 0].item() > 0
    assert mask_t[0, 0, 0].item() > 0
