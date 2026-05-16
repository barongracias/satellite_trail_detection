"""Optuna hyperparameter sweep for the U-Net satellite trail detector.

Runs N_TRIALS short (20-epoch) trials with MedianPruner, then automatically
retrains the best configuration for RETRAIN_EPOCHS epochs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import yaml

from src.training.train_unet import TrainingConfig, run_training
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


_SWEEP_EPOCHS = 20
_RETRAIN_EPOCHS = 75


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments/unet_baseline.yaml")
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--storage", default="sqlite:///results/classical/optuna_sweep.db")
    p.add_argument("--study_name", default="unet_sweep")
    return p.parse_args()


def _make_trial_config(trial: optuna.Trial, base: dict, patch_dir: str) -> TrainingConfig:
    bce_w = trial.suggest_float("bce_weight", 0.2, 0.8)
    return TrainingConfig(
        **{
            **base,
            "patch_dir": patch_dir,
            "experiment_name": f"sweep_trial_{trial.number:03d}",
            "epochs": _SWEEP_EPOCHS,
            "num_workers": 4,
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.7),
            "bce_weight": bce_w,
            "dice_weight": 1.0 - bce_w,
            "normalisation": trial.suggest_categorical("normalisation", ["fixed", "per_image"]),
        }
    )


def _objective(trial: optuna.Trial, base: dict, patch_dir: str) -> float:
    cfg = _make_trial_config(trial, base, patch_dir)
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

    summary = json.loads(Path(results["summary_path"]).read_text())
    return max(e["val_dice"] for e in summary["history"])


def main() -> None:
    args = parse_args()
    logger = get_logger("optuna_sweep")
    seed_everything()

    with open(args.config) as f:
        base = yaml.safe_load(f)

    Path("results/classical").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: _objective(trial, base, args.patch_dir),
        n_trials=args.n_trials,
        timeout=9 * 3600,
    )

    best = study.best_trial
    logger.info(
        "Best trial %d: val_dice=%.4f  params=%s",
        best.number, best.value, best.params,
    )

    out_path = Path("results/classical/optuna_best.json")
    out_path.write_text(json.dumps(
        {"trial_number": best.number, "val_dice": best.value, "params": best.params},
        indent=2,
    ))
    logger.info("Saved best params to %s", out_path)

    bce_w = best.params["bce_weight"]
    retrain_cfg = TrainingConfig(
        **{
            **base,
            "patch_dir": args.patch_dir,
            "experiment_name": "unet_sweep_best",
            "epochs": _RETRAIN_EPOCHS,
            "num_workers": 4,
            "learning_rate": best.params["learning_rate"],
            "dropout_rate": best.params["dropout_rate"],
            "bce_weight": bce_w,
            "dice_weight": 1.0 - bce_w,
            "normalisation": best.params["normalisation"],
        }
    )
    logger.info("Retraining best config for %d epochs", _RETRAIN_EPOCHS)
    retrain_results = run_training(retrain_cfg)
    logger.info("Retrain complete. Checkpoint: %s", retrain_results["best_checkpoint"])

    print(f"\nBest trial: {best.number}  val_dice={best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"\nRetrain checkpoint: {retrain_results['best_checkpoint']}")


if __name__ == "__main__":
    main()
