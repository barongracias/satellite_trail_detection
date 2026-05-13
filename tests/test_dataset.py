from pathlib import Path

import numpy as np
from PIL import Image
import pytest


torch = pytest.importorskip("torch")

from src.data.dataset import SatelliteTrailPatchDataset  # noqa: E402


def _write_pair(
    root_dir: Path,
    stem: str,
    image_array: np.ndarray,
    mask_array: np.ndarray,
) -> None:
    Image.fromarray(image_array).save(root_dir / f"{stem}_red.fits_full.png")
    Image.fromarray(mask_array).save(root_dir / f"{stem}_red_mask.png")


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
