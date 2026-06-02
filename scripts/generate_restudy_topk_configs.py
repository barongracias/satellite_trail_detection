#!/usr/bin/env python
"""Generate materialised configs for M5.6 restudy top-K multi-seed retrains.

The training config loader accepts a single flat YAML, so this reads the
restudy base config, removes the Optuna-only sweep block, applies DB-derived
top-K hyperparameters plus each seed, and writes complete YAMLs.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrialParams:
    number: int
    value: float
    learning_rate: float
    bce_weight: float
    batch_size: int

    @property
    def dice_weight(self) -> float:
        return 1.0 - self.bce_weight


def _decode_param(value: float, distribution_json: str) -> Any:
    distribution = json.loads(distribution_json)
    choices = distribution.get("attributes", {}).get("choices")
    if choices is not None:
        return choices[int(value)]
    return float(value)


def _load_user_attrs(cur: sqlite3.Cursor, trial_id: int) -> dict[str, Any]:
    try:
        rows = cur.execute(
            """
            select key, value_json
            from trial_user_attributes
            where trial_id = ?
            """,
            (trial_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    attrs: dict[str, Any] = {}
    for key, value_json in rows:
        attrs[str(key)] = json.loads(value_json)
    return attrs


def load_top_trials(db_path: Path, top_k: int) -> list[TrialParams]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        """
        select t.trial_id, t.number, v.value
        from trials t
        join trial_values v on v.trial_id = t.trial_id and v.objective = 0
        where t.state = 'COMPLETE'
        order by v.value desc
        limit ?
        """,
        (top_k,),
    ).fetchall()
    trials: list[TrialParams] = []
    for trial_id, number, value in rows:
        params = {
            name: _decode_param(param_value, distribution_json)
            for name, param_value, distribution_json in cur.execute(
                """
                select param_name, param_value, distribution_json
                from trial_params
                where trial_id = ?
                """,
                (trial_id,),
            ).fetchall()
        }
        attrs = _load_user_attrs(cur, int(trial_id))
        batch_size = attrs.get("batch_size", params.get("batch_size"))
        missing = {"learning_rate", "bce_weight"}.difference(params)
        if batch_size is None:
            missing.add("batch_size")
        if missing:
            raise SystemExit(f"trial {number} is missing expected params: {sorted(missing)}")
        trials.append(
            TrialParams(
                number=int(number),
                value=float(value),
                learning_rate=float(params["learning_rate"]),
                bce_weight=float(params["bce_weight"]),
                batch_size=int(batch_size),
            )
        )
    con.close()
    return trials


def materialise_config(
    base: dict[str, Any], trial: TrialParams, seed: int, epochs: int,
    experiment_prefix: str = "unet_paper_arch_noise_topk",
) -> dict[str, Any]:
    cfg = dict(base)
    cfg.pop("sweep", None)
    cfg.update(
        {
            "experiment_name": f"{experiment_prefix}_t{trial.number}_s{seed}",
            "learning_rate": trial.learning_rate,
            "bce_weight": trial.bce_weight,
            "dice_weight": trial.dice_weight,
            "batch_size": trial.batch_size,
            "normalisation": "full_image",
            "noise_augment": True,
            "noise_std_multiplier": 1.0,
            "seed": seed,
            "epochs": epochs,
            "skip_test_eval": True,
        }
    )
    return cfg


def parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("results/classical/unet_paper_arch_noise_f1.db"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/experiments/unet_paper_arch_noise_base.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("configs/experiments/restudy_topk"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seeds", default="2804,1234,42,7,13")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--experiment-prefix", default="unet_paper_arch_noise_topk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_ints(args.seeds)
    base = yaml.safe_load(args.base_config.read_text())
    trials = load_top_trials(args.db, args.top_k)
    if len(trials) != args.top_k:
        raise SystemExit(f"expected {args.top_k} completed trials, found {len(trials)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for trial in trials:
        for seed in seeds:
            cfg = materialise_config(base, trial, seed, args.epochs, args.experiment_prefix)
            out = args.out_dir / f"topk_t{trial.number}_s{seed}.yaml"
            out.write_text(yaml.safe_dump(cfg, sort_keys=False))
            written.append(out)

    print(f"Generated {len(written)} configs in {args.out_dir}")
    for trial in trials:
        print(
            f"trial {trial.number:>3d}: val_f1={trial.value:.6f} "
            f"bs={trial.batch_size} lr={trial.learning_rate:.8g} "
            f"bce={trial.bce_weight:.4f}"
        )


if __name__ == "__main__":
    main()
