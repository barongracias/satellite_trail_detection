"""Optuna hyperparameter sweep for the U-Net satellite trail detector.

M5.6 uses balanced batch-size allocation from the YAML sweep block and optimises
threshold-swept validation F1 without inspecting test metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.threshold_sweep import streaming_sweep_thresholds
from src.config.constants import GLOBAL_SEED
from src.data.dataset import PatchDirectoryDataset
from src.models.loading import load_segmentation_model
from src.training.train_unet import TrainingConfig, resolve_device, run_training
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


_SWEEP_EPOCHS = 20
_RETRAIN_EPOCHS = 75
_SWEEP_TIMEOUT_SECONDS = int(11.5 * 3600)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="configs/experiments/unet_paper_arch_noise_base.yaml",
        help="Base YAML config. A nested 'sweep' block may define Optuna-only settings.",
    )
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--n_trials", type=int, default=45)
    p.add_argument(
        "--study_name",
        default="unet_sweep",
        help="Optuna study name. Drives SQLite storage, best-params JSON, and retrain tag.",
    )
    p.add_argument(
        "--storage",
        default=None,
        help="SQLite URL. Defaults to results/classical/<study_name>.db.",
    )
    p.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Skip the historical single-best automatic retrain after the sweep.",
    )
    return p.parse_args()


def _training_base(base: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base)
    cfg.pop("sweep", None)
    return cfg


def _sweep_config(base: dict[str, Any]) -> dict[str, Any]:
    raw = base.get("sweep", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("sweep config block must be a mapping")
    return raw


def _lr_max_for_batch(sweep_cfg: dict[str, Any], batch_size: int) -> float:
    lr_max = float(sweep_cfg.get("learning_rate_max", 1e-3))
    by_batch = sweep_cfg.get("learning_rate_max_by_batch_size", {}) or {}
    if not isinstance(by_batch, dict):
        raise ValueError("learning_rate_max_by_batch_size must be a mapping")
    return float(by_batch.get(batch_size, by_batch.get(str(batch_size), lr_max)))


def _normalisation_for_trial(
    trial: optuna.Trial,
    base_cfg: dict[str, Any],
    sweep_cfg: dict[str, Any],
) -> str:
    if "normalisation_search_space" in sweep_cfg:
        choices = sweep_cfg["normalisation_search_space"]
        if choices:
            return str(trial.suggest_categorical("normalisation", list(choices)))
    return str(base_cfg.get("normalisation", "fixed"))


def _batch_size_for_trial(
    trial: optuna.Trial,
    base_cfg: dict[str, Any],
    sweep_cfg: dict[str, Any],
) -> int:
    choices = sweep_cfg.get("batch_size_search_space")
    if choices:
        batch_sizes = [int(v) for v in choices]
        batch_size = batch_sizes[trial.number % len(batch_sizes)]
    else:
        batch_size = int(base_cfg.get("batch_size", 16))
    trial.set_user_attr("batch_size", batch_size)
    return batch_size


def _median_pruner_from_config(sweep_cfg: dict[str, Any]) -> optuna.pruners.MedianPruner:
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(sweep_cfg.get("pruner_n_startup_trials", 5)),
        n_warmup_steps=int(sweep_cfg.get("pruner_n_warmup_steps", 10)),
    )


def _make_trial_config(
    trial: optuna.Trial,
    base: dict[str, Any],
    patch_dir: str,
    study_name: str,
) -> TrainingConfig:
    sweep_cfg = _sweep_config(base)
    base_cfg = _training_base(base)
    batch_size = _batch_size_for_trial(trial, base_cfg, sweep_cfg)
    lr_min = float(sweep_cfg.get("learning_rate_min", 1e-4))
    lr_max = _lr_max_for_batch(sweep_cfg, batch_size)
    bce_w = trial.suggest_float(
        "bce_weight",
        float(sweep_cfg.get("bce_weight_min", 0.2)),
        float(sweep_cfg.get("bce_weight_max", 0.8)),
    )
    overrides: dict[str, Any] = {
        "patch_dir": patch_dir,
        "experiment_name": f"{study_name}_trial_{trial.number:03d}",
        "epochs": int(sweep_cfg.get("trial_epochs", _SWEEP_EPOCHS)),
        "batch_size": batch_size,
        "learning_rate": trial.suggest_float("learning_rate", lr_min, lr_max, log=True),
        "bce_weight": bce_w,
        "dice_weight": 1.0 - bce_w,
        "normalisation": _normalisation_for_trial(trial, base_cfg, sweep_cfg),
        "skip_test_eval": bool(sweep_cfg.get("skip_test_eval", False)),
    }
    return TrainingConfig(**{**base_cfg, **overrides})


def _validation_f1_for_checkpoint(
    checkpoint: Path,
    patch_dir: Path,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> dict[str, float]:
    device = resolve_device(device_name)
    model, normalisation = load_segmentation_model(checkpoint, device)
    val_ds = PatchDirectoryDataset(patch_dir / "val", patch_dir / "manifest.csv", normalisation)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    thresholds = np.round(np.arange(0.05, 0.955, 0.01), 2)
    sweep = streaming_sweep_thresholds(model, val_loader, device, thresholds)
    best = max(sweep, key=lambda row: row["f1"])
    return {
        "val_f1": float(best["f1"]),
        "optimal_threshold": float(best["threshold"]),
        "val_precision": float(best["precision"]),
        "val_recall": float(best["recall"]),
    }


def _objective(trial: optuna.Trial, base: dict[str, Any], patch_dir: str, study_name: str) -> float:
    sweep_cfg = _sweep_config(base)
    objective_metric = str(sweep_cfg.get("objective", "val_dice"))
    cfg = _make_trial_config(trial, base, patch_dir, study_name)
    pruned = False

    def _pruner(epoch: int, val_dice: float) -> bool:
        nonlocal pruned
        trial.report(val_dice, epoch)
        if trial.should_prune():
            pruned = True
            return True
        return False

    results = run_training(cfg, pruner_callback=_pruner, save_curves=False)
    if pruned:
        raise optuna.exceptions.TrialPruned()

    if objective_metric == "val_f1":
        val = _validation_f1_for_checkpoint(
            Path(results["best_checkpoint"]),
            Path(patch_dir),
            cfg.batch_size,
            cfg.num_workers,
            cfg.device,
        )
        for key, value in val.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("batch_size", cfg.batch_size)
        return val["val_f1"]

    if objective_metric != "val_dice":
        raise ValueError("sweep objective must be 'val_dice' or 'val_f1'")
    summary = json.loads(Path(results["summary_path"]).read_text())
    return max(e["val_dice"] for e in summary["history"])


def _best_payload(study_name: str, best: optuna.trial.FrozenTrial, objective_metric: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "study_name": study_name,
        "trial_number": best.number,
        "objective_metric": objective_metric,
        "objective_value": best.value,
        "params": best.params,
        "user_attrs": dict(best.user_attrs),
    }
    if objective_metric == "val_f1":
        payload["val_f1"] = best.value
    else:
        payload["val_dice"] = best.value
    return payload


def _retrain_config(
    base: dict[str, Any],
    patch_dir: str,
    study_name: str,
    best: optuna.trial.FrozenTrial,
) -> TrainingConfig:
    base_cfg = _training_base(base)
    bce_w = best.params["bce_weight"]
    kwargs: dict[str, Any] = {
        **base_cfg,
        "patch_dir": patch_dir,
        "experiment_name": study_name,
        "epochs": int(_sweep_config(base).get("retrain_epochs", _RETRAIN_EPOCHS)),
        "batch_size": int(
            best.user_attrs.get(
                "batch_size",
                best.params.get("batch_size", base_cfg.get("batch_size", 16)),
            )
        ),
        "learning_rate": best.params["learning_rate"],
        "bce_weight": bce_w,
        "dice_weight": 1.0 - bce_w,
        "normalisation": best.params.get("normalisation", base_cfg.get("normalisation", "fixed")),
        "skip_test_eval": bool(_sweep_config(base).get("skip_test_eval_retrain", False)),
    }
    return TrainingConfig(**kwargs)


def main() -> None:
    args = parse_args()
    logger = get_logger("optuna_sweep")
    seed_everything()

    with open(args.config) as f:
        base = yaml.safe_load(f)
    sweep_cfg = _sweep_config(base)
    objective_metric = str(sweep_cfg.get("objective", "val_dice"))

    Path("results/classical").mkdir(parents=True, exist_ok=True)

    storage = args.storage or f"sqlite:///results/classical/{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED),
        pruner=_median_pruner_from_config(sweep_cfg),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: _objective(trial, base, args.patch_dir, args.study_name),
        n_trials=args.n_trials,
        timeout=float(sweep_cfg.get("timeout_seconds", _SWEEP_TIMEOUT_SECONDS)),
    )

    best = study.best_trial
    logger.info(
        "Best trial %d: %s=%.4f  params=%s  attrs=%s",
        best.number, objective_metric, best.value, best.params, best.user_attrs,
    )

    out_path = Path("results/classical") / f"{args.study_name}_best.json"
    out_path.write_text(json.dumps(
        _best_payload(args.study_name, best, objective_metric),
        indent=2,
        allow_nan=False,
    ))
    logger.info("Saved best params to %s", out_path)

    auto_retrain = bool(sweep_cfg.get("auto_retrain", True)) and not args.skip_retrain
    if not auto_retrain:
        print(f"\nBest trial: {best.number}  {objective_metric}={best.value:.4f}")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print("\nAuto-retrain skipped.")
        return

    retrain_cfg = _retrain_config(base, args.patch_dir, args.study_name, best)
    logger.info("Retraining best config for %d epochs", retrain_cfg.epochs)
    retrain_results = run_training(retrain_cfg)
    logger.info("Retrain complete. Checkpoint: %s", retrain_results["best_checkpoint"])

    print(f"\nBest trial: {best.number}  {objective_metric}={best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"\nRetrain checkpoint: {retrain_results['best_checkpoint']}")


if __name__ == "__main__":
    main()
