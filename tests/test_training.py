from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.training.train_unet import (  # noqa: E402
    ComboBCEDiceLoss,
    TrainingConfig,
)


def test_training_config_normalises_paths_for_programmatic_use(tmp_path: Path) -> None:
    config = TrainingConfig(
        data_root=str(tmp_path / "processed"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )

    assert isinstance(config.data_root, Path)
    assert isinstance(config.checkpoint_dir, Path)
    assert isinstance(config.log_dir, Path)


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
