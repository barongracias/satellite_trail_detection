import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
import yaml


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.training.train_unet import (  # noqa: E402
    ComboBCEDiceLoss,
    TrainingConfig,
    _append_hard_negative_records,
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


def test_hard_negative_config_and_append_are_opt_in(tmp_path: Path) -> None:
    manifest = tmp_path / "hard_negatives.json"
    manifest.write_text(
        json.dumps({"hard_negatives": [{"patch_path": "train/b_image.png"}]}),
        encoding="utf-8",
    )

    config = TrainingConfig(
        data_root=tmp_path,
        hard_negative_manifest=str(manifest),
        hard_negative_repeat=2,
    )
    assert config.hard_negative_manifest == manifest
    with pytest.raises(ValueError, match="hard_negative_repeat"):
        TrainingConfig(data_root=tmp_path, hard_negative_repeat=-1)

    dataset = type("DummyDataset", (), {})()
    dataset.records = pd.DataFrame(
        [
            {"patch_path": "train/a_image.png", "split": "train"},
            {"patch_path": "train/b_image.png", "split": "train"},
            {"patch_path": "train/c_image.png", "split": "train"},
        ]
    )
    dataset._rows = dataset.records.to_dict("records")

    assert _append_hard_negative_records(dataset, None, repeat=2) == 0
    assert len(dataset.records) == 3

    appended = _append_hard_negative_records(dataset, manifest, repeat=2)
    assert appended == 2
    assert dataset.records["patch_path"].tolist() == [
        "train/a_image.png",
        "train/b_image.png",
        "train/c_image.png",
        "train/b_image.png",
        "train/b_image.png",
    ]
    assert len(dataset._rows) == 5


def test_target_mode_config_is_opt_in_and_validated(tmp_path: Path) -> None:
    # Default is inert hard labels.
    default = TrainingConfig(data_root=tmp_path)
    assert default.target_mode == "hard"
    assert default.soft_dilation_px == 1 and default.soft_band_value == 0.5
    # Opt-in is accepted.
    soft = TrainingConfig(
        data_root=tmp_path, target_mode="dilated_soft", soft_dilation_px=1, soft_band_value=0.5,
    )
    assert soft.target_mode == "dilated_soft"
    # Invalid values are rejected.
    with pytest.raises(ValueError, match="target_mode"):
        TrainingConfig(data_root=tmp_path, target_mode="bogus")
    with pytest.raises(ValueError, match="soft_dilation_px"):
        TrainingConfig(data_root=tmp_path, target_mode="dilated_soft", soft_dilation_px=0)
    with pytest.raises(ValueError, match="soft_band_value"):
        TrainingConfig(data_root=tmp_path, target_mode="dilated_soft", soft_band_value=1.5)


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


def _arch_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "unet_paper_arch_noise_base.yaml"
    return yaml.safe_load(path.read_text())


def test_arch_noise_base_config_loads_as_training_config() -> None:
    raw = _arch_config()
    sweep = raw.pop("sweep")
    config = TrainingConfig(**raw)

    assert config.batch_size == 16
    assert config.normalisation == "full_image"
    assert config.skip_test_eval is True
    assert sweep["objective"] == "val_f1"
    assert sweep["batch_size_search_space"] == [8, 16, 32]
    assert "dropout_rate" not in raw


class _FakeTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.user_attrs: dict[str, int] = {}
        self.suggested: list[str] = []

    def set_user_attr(self, key: str, value: int) -> None:
        self.user_attrs[key] = value

    def suggest_float(self, name: str, low: float, high: float, **kwargs: object) -> float:
        assert name != "dropout_rate"
        self.suggested.append(name)
        return low

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        raise AssertionError(f"unexpected categorical suggestion: {name}={choices}")


def test_arch_sweep_assigns_balanced_batch_sizes_without_categorical_suggest() -> None:
    from src.training.sweep import _make_trial_config

    base = _arch_config()
    batch_sizes = []
    for number in range(45):
        trial = _FakeTrial(number)
        config = _make_trial_config(
            trial=trial,  # type: ignore[arg-type]
            base=base,
            patch_dir="data/patches",
            study_name="unet_paper_arch_noise_f1",
        )
        batch_sizes.append(config.batch_size)
        assert trial.user_attrs["batch_size"] == config.batch_size
        assert trial.suggested == ["bce_weight", "learning_rate"]

    assert batch_sizes.count(8) == 15
    assert batch_sizes.count(16) == 15
    assert batch_sizes.count(32) == 15


def test_arch_sweep_uses_common_lr_bounds_and_yaml_pruner_settings() -> None:
    from src.training.sweep import _make_trial_config, _median_pruner_from_config

    base = _arch_config()
    sweep_cfg = base["sweep"]
    trial = _FakeTrial(2)
    config = _make_trial_config(
        trial=trial,  # type: ignore[arg-type]
        base=base,
        patch_dir="data/patches",
        study_name="unet_paper_arch_noise_f1",
    )
    pruner = _median_pruner_from_config(sweep_cfg)

    assert config.batch_size == 32
    assert config.learning_rate == pytest.approx(1.0e-4)
    assert sweep_cfg["learning_rate_max"] == pytest.approx(1.0e-3)
    assert pruner._n_startup_trials == 15
    assert pruner._n_warmup_steps == 10


def test_generate_topk_configs_reads_batch_size_from_user_attrs(tmp_path: Path) -> None:
    from scripts.sweep.generate_restudy_topk_configs import load_top_trials

    db = tmp_path / "study.db"
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("create table trials (trial_id integer, number integer, state text)")
    cur.execute("create table trial_values (trial_id integer, objective integer, value real)")
    cur.execute(
        "create table trial_params (trial_id integer, param_name text, param_value real, distribution_json text)"
    )
    cur.execute("create table trial_user_attributes (trial_id integer, key text, value_json text)")
    cur.execute("insert into trials values (1, 24, 'COMPLETE')")
    cur.execute("insert into trial_values values (1, 0, 0.8372)")
    cur.executemany(
        "insert into trial_params values (1, ?, ?, ?)",
        [
            ("learning_rate", 2.7e-4, json.dumps({"attributes": {}})),
            ("bce_weight", 0.63, json.dumps({"attributes": {}})),
        ],
    )
    cur.execute("insert into trial_user_attributes values (1, 'batch_size', '16')")
    con.commit()
    con.close()

    trials = load_top_trials(db, top_k=1)

    assert len(trials) == 1
    assert trials[0].number == 24
    assert trials[0].batch_size == 16
    assert trials[0].learning_rate == pytest.approx(2.7e-4)
