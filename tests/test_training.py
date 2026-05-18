from pathlib import Path

import numpy as np
from PIL import Image
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.training.train_unet import (  # noqa: E402
    ComboBCEDiceLoss,
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


def test_combo_loss_squared_denominator_matches_paper_formula() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[[[2.0, -2.0], [2.0, -2.0]]]])  # probs ≈ [0.88, 0.12, 0.88, 0.12]
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    probs = torch.sigmoid(logits)
    spatial = (2, 3)
    intersection = (probs * targets).sum(dim=spatial)
    squared_denom = (probs ** 2).sum(dim=spatial) + (targets ** 2).sum(dim=spatial)
    expected_dice = 1.0 - (2 * intersection + 1e-4) / (squared_denom + 1e-4)

    loss = ComboBCEDiceLoss(
        bce_weight=0.0, dice_weight=1.0,
        dice_smooth=1e-4, dice_denominator_squared=True,
    )
    assert torch.allclose(loss(logits, targets), expected_dice.mean(), atol=1e-6)


def test_combo_loss_linear_denominator_matches_legacy_formula() -> None:
    logits = torch.tensor([[[[2.0, -2.0], [2.0, -2.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    probs = torch.sigmoid(logits)
    spatial = (2, 3)
    intersection = (probs * targets).sum(dim=spatial)
    linear_denom = probs.sum(dim=spatial) + targets.sum(dim=spatial)
    expected_dice = 1.0 - (2 * intersection + 1e-6) / (linear_denom + 1e-6)

    loss = ComboBCEDiceLoss(
        bce_weight=0.0, dice_weight=1.0,
        dice_smooth=1e-6, dice_denominator_squared=False,
    )
    assert torch.allclose(loss(logits, targets), expected_dice.mean(), atol=1e-6)


def test_combo_loss_formulas_disagree_on_partial_probabilities() -> None:
    # For binary targets the two formulas should differ; otherwise the squared-vs-linear
    # distinction would be inert and the new config field would be pointless.
    logits = torch.full((1, 1, 4, 4), 0.5)  # mid-probability so p**2 < p
    targets = torch.zeros((1, 1, 4, 4))
    targets[..., :2] = 1.0

    sq = ComboBCEDiceLoss(bce_weight=0.0, dice_weight=1.0,
                          dice_smooth=1e-4, dice_denominator_squared=True)
    lin = ComboBCEDiceLoss(bce_weight=0.0, dice_weight=1.0,
                           dice_smooth=1e-6, dice_denominator_squared=False)
    assert not torch.isclose(sq(logits, targets), lin(logits, targets), atol=1e-3)
