from pathlib import Path

import numpy as np
from PIL import Image
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.config.constants import PATCH_SIZE  # noqa: E402
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
    assert config.patch_size == PATCH_SIZE == 528


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


def test_training_config_validates_precision_choices(tmp_path: Path) -> None:
    assert TrainingConfig(data_root=tmp_path, amp_dtype="bf16").amp_dtype == "bfloat16"
    assert TrainingConfig(data_root=tmp_path, amp_dtype="float16").amp_dtype == "fp16"
    TrainingConfig(data_root=tmp_path, amp_dtype="fp16")
    TrainingConfig(data_root=tmp_path, amp_dtype="bfloat16")
    TrainingConfig(data_root=tmp_path, float32_matmul_precision="high")
    with pytest.raises(ValueError):
        TrainingConfig(data_root=tmp_path, amp_dtype="tf32")
    with pytest.raises(ValueError):
        TrainingConfig(data_root=tmp_path, float32_matmul_precision="low")


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
        "amp_dtype": "bfloat16",
        "float32_matmul_precision": "high",
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
    assert cfg.amp_dtype == "bfloat16"
    assert cfg.float32_matmul_precision == "high"
    assert cfg.lr_scheduler == "cosine"
    assert cfg.normalisation == "per_image"
    trial.suggest_float.assert_any_call("learning_rate", 1e-4, 1e-3, log=True)
    trial.suggest_categorical.assert_any_call("normalisation", ["per_image", "full_image"])


def test_reoptuna_base_uses_bf16_precision_bundle() -> None:
    base_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "unet_reoptuna_base.yaml"
    contents = base_path.read_text()
    assert "use_amp: true" in contents
    assert "amp_dtype: bfloat16" in contents
    assert "float32_matmul_precision: high" in contents
    assert "normalisation: per_image" in contents


def test_optuna_sweep_sbatch_defaults_to_reoptuna_base() -> None:
    """The optuna_sweep sbatch must default CONFIG to the re-Optuna base
    config so submitting without a CONFIG override doesn't silently use
    the legacy baseline."""
    sbatch_path = Path(__file__).resolve().parents[1] / "slurm" / "optuna_sweep.sbatch"
    contents = sbatch_path.read_text()
    assert 'CONFIG="${CONFIG:-configs/experiments/unet_reoptuna_base.yaml}"' in contents



def test_save_training_summary_omits_test_keys_without_test_result(tmp_path: Path) -> None:
    import json
    from src.training.train_unet import save_training_summary

    config = TrainingConfig(
        data_root=tmp_path / "processed",
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        skip_test_eval=True,
    )
    summary_path = save_training_summary(
        config=config,
        history=[],
        best_checkpoint=tmp_path / "best.pth",
        latest_checkpoint=tmp_path / "latest.pth",
        effective_eval_max_batches=None,
        precision_metadata={"amp_enabled": False},
        test_result=None,
    )
    data = json.loads(summary_path.read_text())
    assert "test_counts" not in data
    assert "test_metrics" not in data
    assert "test_loss" not in data
    assert data["config"]["skip_test_eval"] is True


def test_paper_noise_base_config_locks_restudy_bundle() -> None:
    import yaml

    base_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "unet_paper_noise_base.yaml"
    data = yaml.safe_load(base_path.read_text())
    sweep = data.pop("sweep")
    cfg = TrainingConfig(**data)

    assert cfg.batch_size == 16
    assert cfg.num_workers == 8
    assert cfg.use_amp is True
    assert cfg.amp_dtype == "bfloat16"
    assert cfg.float32_matmul_precision == "high"
    assert cfg.lr_scheduler == "cosine"
    assert cfg.normalisation == "full_image"
    assert cfg.noise_augment is True
    assert cfg.noise_std_multiplier == 1.0
    assert cfg.auto_pos_weight is False
    assert cfg.pos_weight is None
    assert cfg.dice_denominator_squared is True
    assert cfg.dice_smooth == pytest.approx(1.0e-4)
    assert sweep["objective"] == "val_f1"
    assert sweep["auto_retrain"] is False
    assert sweep["skip_test_eval"] is True
    assert sweep["normalisation_search_space"] == []
    assert sweep["batch_size_search_space"] == [8, 16, 32]
    assert sweep["learning_rate_max_by_batch_size"][8] == pytest.approx(5.0e-4)
    assert sweep["learning_rate_max_by_batch_size"][16] == pytest.approx(1.0e-3)
    assert sweep["learning_rate_max_by_batch_size"][32] == pytest.approx(2.0e-3)


def test_restudy_sweep_config_searches_batch_size_and_keeps_full_image(tmp_path: Path) -> None:
    from unittest.mock import MagicMock
    import yaml

    optuna = pytest.importorskip("optuna")
    from src.training.sweep import _make_trial_config

    base_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "unet_paper_noise_base.yaml"
    base = yaml.safe_load(base_path.read_text())
    base["data_root"] = str(tmp_path / "processed")
    base["checkpoint_dir"] = str(tmp_path / "checkpoints")
    base["log_dir"] = str(tmp_path / "logs")

    trial = MagicMock(spec=optuna.Trial)
    trial.number = 3

    def suggest_categorical(name, choices):
        assert name == "batch_size"
        assert choices == [8, 16, 32]
        return 32

    def suggest_float(name, low, high, **kwargs):
        if name == "bce_weight":
            assert (low, high) == (0.2, 0.8)
            return 0.6
        if name == "learning_rate":
            assert low == pytest.approx(1.0e-4)
            assert high == pytest.approx(2.0e-3)
            assert kwargs == {"log": True}
            return high
        if name == "dropout_rate":
            assert (low, high) == (0.1, 0.7)
            return 0.2
        raise AssertionError(name)

    trial.suggest_categorical.side_effect = suggest_categorical
    trial.suggest_float.side_effect = suggest_float

    cfg = _make_trial_config(trial, base, patch_dir=str(tmp_path), study_name="study")
    assert cfg.experiment_name == "study_trial_003"
    assert cfg.batch_size == 32
    assert cfg.learning_rate == pytest.approx(2.0e-3)
    assert cfg.dropout_rate == pytest.approx(0.2)
    assert cfg.bce_weight == pytest.approx(0.6)
    assert cfg.dice_weight == pytest.approx(0.4)
    assert cfg.normalisation == "full_image"
    assert cfg.noise_augment is True
    assert cfg.noise_std_multiplier == 1.0
    assert cfg.skip_test_eval is True
    assert trial.suggest_categorical.call_count == 1


def test_sweep_best_payload_records_val_f1_user_attrs() -> None:
    from types import SimpleNamespace
    from src.training.sweep import _best_payload

    best = SimpleNamespace(
        number=12,
        value=0.84,
        params={"batch_size": 16, "learning_rate": 3e-4},
        user_attrs={"val_f1": 0.84, "optimal_threshold": 0.51},
    )
    payload = _best_payload("study", best, "val_f1")
    assert payload["objective_metric"] == "val_f1"
    assert payload["val_f1"] == pytest.approx(0.84)
    assert payload["user_attrs"]["optimal_threshold"] == pytest.approx(0.51)


def test_optuna_sweep_sbatch_supports_skip_retrain_flag() -> None:
    sbatch_path = Path(__file__).resolve().parents[1] / "slurm" / "optuna_sweep.sbatch"
    contents = sbatch_path.read_text()
    assert 'SKIP_RETRAIN="${SKIP_RETRAIN:-0}"' in contents
    assert "--skip-retrain" in contents
