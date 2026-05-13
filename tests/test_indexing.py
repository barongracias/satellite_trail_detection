from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.data.indexing import PatchDatasetConfig, discover_image_mask_pairs


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
