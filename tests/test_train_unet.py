from pathlib import Path

import numpy as np
from PIL import Image
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.training.train_unet import (  # noqa: E402
    TrainingConfig,
    estimate_positive_pixel_fraction,
)


def _write_pair(
    root_dir: Path,
    stem: str,
    image_array: np.ndarray,
    mask_array: np.ndarray,
) -> None:
    Image.fromarray(image_array).save(root_dir / f"{stem}_red.fits_full.png")
    Image.fromarray(mask_array).save(root_dir / f"{stem}_red_mask.png")


def test_training_config_normalises_paths_for_programmatic_use(tmp_path: Path) -> None:
    config = TrainingConfig(
        data_root=str(tmp_path / "processed"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )

    assert isinstance(config.data_root, Path)
    assert isinstance(config.checkpoint_dir, Path)
    assert isinstance(config.log_dir, Path)


def test_estimate_positive_pixel_fraction_matches_patch_coverage(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    image_array = np.full((4, 4), 100, dtype=np.uint8)
    mask_array = np.array(
        [
            [255, 255, 0, 0],
            [255, 255, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    _write_pair(processed_dir, "A", image_array, mask_array)

    config = TrainingConfig(
        data_root=processed_dir,
        patch_size=2,
        stride=2,
    )

    assert estimate_positive_pixel_fraction(config) == 0.25
