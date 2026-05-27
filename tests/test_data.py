"""Tests for the data layer: catalogue, dataset, indexing, and splits."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.catalog import SatelliteCatalog
from src.config.constants import PATCH_SIZE
from src.data.indexing import (
    PatchDatasetConfig,
    discover_image_mask_pairs,
    parse_observation_datetime,
)


torch = pytest.importorskip("torch")

from src.data.dataset import PatchDirectoryDataset  # noqa: E402
from src.data.splits import create_image_level_splits  # noqa: E402
from src.data.transforms import (  # noqa: E402
    SignalDependentNoise,
    get_eval_transforms,
    get_train_transforms,
)


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
    default_config = PatchDatasetConfig(root_dir=tmp_path)
    assert default_config.patch_size == PATCH_SIZE == 528
    assert default_config.resolved_stride == PATCH_SIZE

    with pytest.raises(FileNotFoundError):
        PatchDatasetConfig(root_dir=tmp_path / "missing")

    with pytest.raises(ValueError):
        PatchDatasetConfig(root_dir=tmp_path, patch_size=0)

    with pytest.raises(ValueError):
        PatchDatasetConfig(root_dir=tmp_path, patch_size=2, stride=0)


def test_csd3_patch_build_wrapper_defaults_to_active_patch_geometry() -> None:
    sbatch_path = Path(__file__).resolve().parents[1] / "slurm" / "build_patches.sbatch"
    contents = sbatch_path.read_text()

    assert f'PATCH_SIZE="${{PATCH_SIZE:-{PATCH_SIZE}}}"' in contents
    assert f'STRIDE="${{STRIDE:-{PATCH_SIZE}}}"' in contents


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
                "positive_pixel_fraction",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "split": "train",
            "source_image": "fake.png",
            "patch_path": str(img_path),
            "mask_path": str(msk_path),
            "positive_pixel_fraction": 1.0 / 64,
        })

    dataset = PatchDirectoryDataset(train_dir, manifest_path)
    assert len(dataset) == 1

    sample = dataset[0]
    assert set(sample.keys()) == {"image", "mask"}
    assert sample["image"].shape[0] == 1  # grayscale channel
    assert sample["mask"].shape[0] == 1
    assert sample["image"].shape[1:] == sample["mask"].shape[1:]


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
                        "positive_pixel_fraction"],
        )
        writer.writeheader()
        writer.writerow({
            "split": "train", "source_image": "fake.png",
            "patch_path": str(img_path), "mask_path": str(msk_path),
            "positive_pixel_fraction": 0.0,
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


def test_build_patch_dataset_writes_train_patches_without_augmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import csv

    from scripts import build_patch_dataset as builder
    from src.data import transforms as transform_module

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    image_array = np.array([[0, 80], [160, 255]], dtype=np.uint8)
    mask_array = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    _write_pair(processed_dir, "A", image_array, mask_array)

    monkeypatch.setattr(
        builder,
        "create_image_level_splits",
        lambda pairs, trail_counts, train_ratio, val_ratio, seed: (pairs, [], []),
    )
    monkeypatch.setattr(
        transform_module,
        "get_train_transforms",
        lambda: pytest.fail("build_patch_dataset must not use train augmentation"),
    )

    out_dir = tmp_path / "patches"
    manifest_path = builder.build(
        data_root=processed_dir,
        out_dir=out_dir,
        patch_size=2,
        stride=2,
        splits=("train",),
    )

    assert not hasattr(builder, "get_train_transforms")

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    saved_image = np.asarray(Image.open(rows[0]["patch_path"]).convert("L"))
    saved_mask = np.asarray(Image.open(rows[0]["mask_path"]).convert("L"))

    assert np.allclose(saved_image, image_array, atol=1)
    assert np.array_equal(saved_mask, mask_array)


def test_patch_directory_dataset_augmentation_active_for_train_split(tmp_path: Path) -> None:
    import csv

    train_dir = tmp_path / "patches" / "train"
    train_dir.mkdir(parents=True)

    img_array = np.zeros((8, 8), dtype=np.uint8)
    img_array[0:4, 0:4] = 200
    mask_array = np.zeros((8, 8), dtype=np.uint8)
    mask_array[0:4, 0:4] = 255

    img_path = train_dir / "patch0_image.png"
    msk_path = train_dir / "patch0_mask.png"
    Image.fromarray(img_array).save(img_path)
    Image.fromarray(mask_array).save(msk_path)

    manifest_path = tmp_path / "patches" / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "source_image", "patch_path", "mask_path",
                        "positive_pixel_fraction"],
        )
        writer.writeheader()
        writer.writerow({
            "split": "train", "source_image": "fake.png",
            "patch_path": str(img_path), "mask_path": str(msk_path),
            "positive_pixel_fraction": 1.0/64,
        })

    dataset = PatchDirectoryDataset(train_dir, manifest_path, normalisation="fixed")

    # Seed before sampling so a flaky outcome is deterministic. The 4×4 patch
    # admits 8 distinct dihedral orientations; with N=12 draws we expect at
    # least two distinct configurations with overwhelming probability.
    torch.manual_seed(0)
    images = [dataset[0]["image"] for _ in range(12)]

    distinct = {tuple(img.flatten().tolist()) for img in images}
    assert len(distinct) >= 2, "Augmentation should produce >=2 distinct outputs in train split"


def test_patch_directory_dataset_augment_train_false_freezes_augmentation(
    tmp_path: Path,
) -> None:
    import csv

    train_dir = tmp_path / "patches" / "train"
    train_dir.mkdir(parents=True)

    img_array = np.zeros((8, 8), dtype=np.uint8)
    img_array[0:4, 0:4] = 200
    mask_array = np.zeros((8, 8), dtype=np.uint8)
    mask_array[0:4, 0:4] = 255

    img_path = train_dir / "patch0_image.png"
    msk_path = train_dir / "patch0_mask.png"
    Image.fromarray(img_array).save(img_path)
    Image.fromarray(mask_array).save(msk_path)

    manifest_path = tmp_path / "patches" / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "source_image", "patch_path", "mask_path",
                        "positive_pixel_fraction"],
        )
        writer.writeheader()
        writer.writerow({
            "split": "train", "source_image": "fake.png",
            "patch_path": str(img_path), "mask_path": str(msk_path),
            "positive_pixel_fraction": 1.0/64,
        })

    dataset = PatchDirectoryDataset(
        train_dir, manifest_path, normalisation="fixed", augment_train=False,
    )
    torch.manual_seed(0)
    images = [dataset[0]["image"] for _ in range(8)]

    # With augment_train=False, repeated draws must be byte-identical.
    for i, img in enumerate(images[1:], start=1):
        assert torch.allclose(images[0], img), f"draw {i} differs with augment_train=False"


def test_patch_directory_dataset_eval_splits_never_augment_by_default(
    tmp_path: Path,
) -> None:
    image_array = np.zeros((8, 8), dtype=np.uint8)
    image_array[0:4, 0:4] = 200
    mask_array = np.zeros((8, 8), dtype=np.uint8)
    mask_array[0:4, 0:4] = 255

    for split in ("val", "test"):
        split_dir, manifest_path, _ = _write_directory_patch(
            tmp_path / split,
            split,
            image_array,
            mask_array,
        )
        dataset = PatchDirectoryDataset(
            split_dir,
            manifest_path,
            normalisation="fixed",
        )

        torch.manual_seed(0)
        images = [dataset[0]["image"] for _ in range(8)]

        for i, img in enumerate(images[1:], start=1):
            assert torch.allclose(images[0], img), (
                f"{split} draw {i} differs; augmentation leaked into eval split"
            )


def test_patch_directory_dataset_full_image_normalisation_uses_manifest_stats(
    tmp_path: Path,
) -> None:
    import csv

    test_dir = tmp_path / "patches" / "test"
    test_dir.mkdir(parents=True)

    img_array = np.full((8, 8), 200, dtype=np.uint8)
    mask_array = np.zeros((8, 8), dtype=np.uint8)
    img_path = test_dir / "patch0_image.png"
    msk_path = test_dir / "patch0_mask.png"
    Image.fromarray(img_array).save(img_path)
    Image.fromarray(mask_array).save(msk_path)

    # image_mean/std stored on the [0, 1] scale (matching JointTransform output).
    target_mean = 0.5
    target_std = 0.25

    manifest_path = tmp_path / "patches" / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "source_image", "patch_path", "mask_path",
                        "positive_pixel_fraction",
                        "image_mean", "image_std"],
        )
        writer.writeheader()
        writer.writerow({
            "split": "test", "source_image": "fake.png",
            "patch_path": str(img_path), "mask_path": str(msk_path),
            "positive_pixel_fraction": 0.0,
            "image_mean": target_mean, "image_std": target_std,
        })

    dataset = PatchDirectoryDataset(test_dir, manifest_path, normalisation="full_image")
    image = dataset[0]["image"]

    # PIL value 200 → 200/255 ≈ 0.784 after JointTransform.
    # full_image normalisation then gives (0.784 - 0.5) / 0.25 ≈ 1.137.
    expected = (200.0 / 255.0 - target_mean) / (target_std + 1e-6)
    assert torch.allclose(image, torch.full_like(image, expected), atol=1e-4)



def _write_directory_patch(
    tmp_path: Path,
    split: str,
    image_array: np.ndarray,
    mask_array: np.ndarray,
) -> tuple[Path, Path, Path]:
    import csv

    split_dir = tmp_path / "patches" / split
    split_dir.mkdir(parents=True)
    img_path = split_dir / "patch0_image.png"
    mask_path = split_dir / "patch0_mask.png"
    Image.fromarray(image_array).save(img_path)
    Image.fromarray(mask_array).save(mask_path)

    manifest_path = tmp_path / "patches" / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "source_image", "patch_path", "mask_path",
                "positive_pixel_fraction",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "split": split,
            "source_image": "fake.png",
            "patch_path": str(img_path),
            "mask_path": str(mask_path),
            "positive_pixel_fraction": float(mask_array.max() > 0),
        })
    return split_dir, manifest_path, mask_path


def _write_noise_calibration(tmp_path: Path, alpha: float = 0.0, beta: float = 0.1) -> Path:
    import json

    path = tmp_path / "noise_calibration.json"
    path.write_text(json.dumps({"alpha": alpha, "beta": beta}))
    return path


def test_patch_directory_dataset_noise_default_preserves_images(tmp_path: Path) -> None:
    image_array = np.full((8, 8), 128, dtype=np.uint8)
    mask_array = np.zeros((8, 8), dtype=np.uint8)
    split_dir, manifest_path, _ = _write_directory_patch(tmp_path, "train", image_array, mask_array)

    dataset = PatchDirectoryDataset(
        split_dir,
        manifest_path,
        normalisation="fixed",
        augment_train=False,
    )
    image = dataset[0]["image"]

    expected = (128.0 / 255.0 - 0.5) / 0.5
    assert torch.allclose(image, torch.full_like(image, expected), atol=1e-6)


def test_patch_directory_dataset_noise_changes_train_image_not_mask(tmp_path: Path) -> None:
    image_array = np.full((8, 8), 128, dtype=np.uint8)
    mask_array = np.zeros((8, 8), dtype=np.uint8)
    mask_array[2:4, 3:5] = 255
    split_dir, manifest_path, _ = _write_directory_patch(tmp_path, "train", image_array, mask_array)
    calibration = _write_noise_calibration(tmp_path)

    plain = PatchDirectoryDataset(
        split_dir,
        manifest_path,
        normalisation="fixed",
        augment_train=False,
    )
    noisy = PatchDirectoryDataset(
        split_dir,
        manifest_path,
        normalisation="fixed",
        augment_train=False,
        noise_augment=True,
        noise_std_multiplier=1.0,
        noise_calibration_path=calibration,
    )

    torch.manual_seed(0)
    noisy_sample = noisy[0]
    plain_sample = plain[0]

    assert not torch.allclose(noisy_sample["image"], plain_sample["image"])
    assert torch.equal(noisy_sample["mask"], plain_sample["mask"])


def test_patch_directory_dataset_noise_never_applies_to_eval_splits(tmp_path: Path) -> None:
    image_array = np.full((8, 8), 128, dtype=np.uint8)
    mask_array = np.zeros((8, 8), dtype=np.uint8)
    split_dir, manifest_path, _ = _write_directory_patch(tmp_path, "val", image_array, mask_array)
    calibration = _write_noise_calibration(tmp_path)

    plain = PatchDirectoryDataset(
        split_dir,
        manifest_path,
        normalisation="fixed",
        augment_train=False,
    )
    requested_noise = PatchDirectoryDataset(
        split_dir,
        manifest_path,
        normalisation="fixed",
        augment_train=False,
        noise_augment=True,
        noise_std_multiplier=1.0,
        noise_calibration_path=calibration,
    )

    torch.manual_seed(0)
    image_a = plain[0]["image"]
    torch.manual_seed(123)
    image_b = requested_noise[0]["image"]

    assert torch.allclose(image_a, image_b)


def test_patch_directory_dataset_noise_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    image_array = np.full((8, 8), 128, dtype=np.uint8)
    mask_array = np.zeros((8, 8), dtype=np.uint8)
    split_dir, manifest_path, _ = _write_directory_patch(tmp_path, "train", image_array, mask_array)
    calibration = _write_noise_calibration(tmp_path)
    dataset = PatchDirectoryDataset(
        split_dir,
        manifest_path,
        normalisation="fixed",
        augment_train=False,
        noise_augment=True,
        noise_std_multiplier=1.0,
        noise_calibration_path=calibration,
    )

    torch.manual_seed(2804)
    first = dataset[0]["image"]
    torch.manual_seed(2804)
    second = dataset[0]["image"]

    assert torch.allclose(first, second)


def test_signal_dependent_noise_calibration_matches_background_sigma() -> None:
    mu_bg = 0.5
    sigma_bg = 0.05
    alpha = sigma_bg ** 2 / (2.0 * mu_bg)
    beta = sigma_bg / np.sqrt(2.0)
    transform = SignalDependentNoise(alpha=alpha, beta=beta, multiplier=1.0)
    image = torch.full((1, 256, 256), mu_bg)

    torch.manual_seed(0)
    noisy = transform(image)
    empirical_std = (noisy - image).std(unbiased=False).item()

    assert abs(empirical_std - sigma_bg) < 0.003
