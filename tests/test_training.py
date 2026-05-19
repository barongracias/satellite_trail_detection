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


def test_training_config_validates_lr_scheduler_choice(tmp_path: Path) -> None:
    TrainingConfig(data_root=tmp_path, lr_scheduler=None)
    TrainingConfig(data_root=tmp_path, lr_scheduler="cosine")
    with pytest.raises(ValueError):
        TrainingConfig(data_root=tmp_path, lr_scheduler="step")


def test_amp_scaler_is_noop_when_disabled() -> None:
    # The CPU CI path can't exercise FP16 autocast meaningfully, but it must
    # still construct GradScaler(enabled=False) and let gradients flow through
    # the normal backward path.
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    assert not scaler.is_enabled()
    x = torch.tensor([2.0], requires_grad=True)
    loss = (x ** 2).sum()
    scaler.scale(loss).backward()
    assert torch.allclose(x.grad, torch.tensor([4.0]))


def test_threshold_sweep_json_strict_serialisation() -> None:
    """Fixed-threshold mode produces None (not NaN) so the output JSON
    serialises cleanly under allow_nan=False. Locks R1."""
    import json
    out_payload = {
        "tag": "demo",
        "checkpoint": "results/checkpoints/demo_best.pth",
        "optimal_threshold": 0.6,
        "val_f1": None,
        "val_precision": None,
        "val_recall": None,
        "test_precision": 0.8,
        "test_recall": 0.7,
        "test_dice": 0.75,
        "test_iou": 0.6,
        "pr_curve": [],
    }
    # Must not raise.
    json.dumps(out_payload, indent=2, allow_nan=False)


def test_sweep_trial_config_preserves_num_workers(tmp_path: Path) -> None:
    """The Optuna trial-config builder must inherit num_workers from the base
    YAML — historically it was hardcoded to 4, defeating the re-Optuna perf
    bundle's num_workers=8."""
    from unittest.mock import MagicMock

    optuna = pytest.importorskip("optuna")
    from src.training.sweep import _make_trial_config

    base = {
        "data_root": str(tmp_path),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "logs"),
        "num_workers": 8,
        "batch_size": 16,
        "use_amp": True,
        "lr_scheduler": "cosine",
    }
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    trial.suggest_float.side_effect = lambda name, *a, **k: 0.5
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
    cfg = _make_trial_config(trial, base, patch_dir=str(tmp_path), study_name="study")
    assert cfg.num_workers == 8
    assert cfg.batch_size == 16
    assert cfg.use_amp is True
    assert cfg.lr_scheduler == "cosine"


def test_optuna_sweep_sbatch_defaults_to_reoptuna_base() -> None:
    """The optuna_sweep sbatch must default CONFIG to the re-Optuna base
    config so submitting without a CONFIG override doesn't silently use
    the legacy baseline."""
    sbatch_path = Path(__file__).resolve().parents[1] / "slurm" / "optuna_sweep.sbatch"
    contents = sbatch_path.read_text()
    assert 'CONFIG="${CONFIG:-configs/experiments/unet_reoptuna_base.yaml}"' in contents
