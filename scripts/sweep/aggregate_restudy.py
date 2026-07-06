#!/usr/bin/env python
"""Aggregate M5.6 restudy top-K multi-seed val-only threshold sweeps."""

from __future__ import annotations

import argparse
from glob import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TAG_RE = re.compile(
    r"^(?P<prefix>(?:attention_)?unet_paper_arch_noise_topk)_t(?P<trial>\d+)_s(?P<seed>\d+)$"
)
TEST_KEYS = {"test_precision", "test_recall", "test_dice", "test_iou", "test_counts", "test_metrics", "test_loss"}


@dataclass(frozen=True)
class SeedResult:
    tag_prefix: str
    trial: int
    seed: int
    tag: str
    val_f1: float | None
    optimal_threshold: float | None
    threshold_json: str
    summary_json: str | None
    nan_loss: bool
    final_val_dice: float | None
    max_consecutive_zero_val_dice: int
    val_loss_spike_count_2x: int
    config: dict[str, Any]


def parse_ints(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def parse_tag(tag: str) -> tuple[str, int, int]:
    match = TAG_RE.match(tag)
    if not match:
        raise ValueError(f"unexpected restudy tag: {tag}")
    return match.group("prefix"), int(match.group("trial")), int(match.group("seed"))


def count_max_consecutive_zero(history: list[dict[str, Any]]) -> int:
    best = 0
    current = 0
    for row in history:
        if row.get("val_dice") in (0, 0.0):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def count_val_loss_spikes(history: list[dict[str, Any]]) -> int:
    count = 0
    rolling_min = math.inf
    for row in history:
        val = row.get("val_loss")
        if not isinstance(val, int | float) or not math.isfinite(float(val)):
            continue
        val_f = float(val)
        if rolling_min < math.inf and val_f > 2.0 * rolling_min:
            count += 1
        rolling_min = min(rolling_min, val_f)
    return count


def summary_stability(tag: str, summary_dir: Path) -> tuple[str | None, bool, float | None, int, int, dict[str, Any]]:
    path = summary_dir / f"{tag}_summary.json"
    if not path.exists():
        return None, False, None, 0, 0, {}
    data = json.loads(path.read_text())
    present_test_keys = TEST_KEYS.intersection(data)
    if present_test_keys:
        raise SystemExit(f"{path} contains inadmissible test metric keys: {sorted(present_test_keys)}")
    history = data.get("history", [])
    nan_loss = False
    for row in history:
        for key in ("train_loss", "val_loss"):
            val = row.get(key)
            if not isinstance(val, int | float) or not math.isfinite(float(val)):
                nan_loss = True
    final_val_dice = None
    if history:
        val = history[-1].get("val_dice")
        final_val_dice = float(val) if isinstance(val, int | float) and math.isfinite(float(val)) else None
    return (
        str(path),
        nan_loss,
        final_val_dice,
        count_max_consecutive_zero(history),
        count_val_loss_spikes(history),
        data.get("config", {}),
    )


def bootstrap_ci(values: list[float], seed: int, n_resamples: int, alpha: float) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def load_results(pattern: str, summary_dir: Path) -> list[SeedResult]:
    results: list[SeedResult] = []
    for path_str in sorted(glob(pattern)):
        path = Path(path_str)
        data = json.loads(path.read_text())
        present_test_keys = TEST_KEYS.intersection(data)
        if present_test_keys:
            raise SystemExit(f"{path} contains inadmissible test metric keys: {sorted(present_test_keys)}")
        tag = data.get("tag")
        if not isinstance(tag, str):
            raise SystemExit(f"{path} is missing string tag")
        tag_prefix, trial, seed = parse_tag(tag)
        val_f1 = data.get("val_f1")
        threshold = data.get("optimal_threshold")
        summary_path, nan_loss, final_val_dice, max_zero, spike_count, cfg = summary_stability(tag, summary_dir)
        results.append(
            SeedResult(
                tag_prefix=tag_prefix,
                trial=trial,
                seed=seed,
                tag=tag,
                val_f1=float(val_f1) if isinstance(val_f1, int | float) else None,
                optimal_threshold=float(threshold) if isinstance(threshold, int | float) else None,
                threshold_json=str(path),
                summary_json=summary_path,
                nan_loss=nan_loss,
                final_val_dice=final_val_dice,
                max_consecutive_zero_val_dice=max_zero,
                val_loss_spike_count_2x=spike_count,
                config=cfg,
            )
        )
    return results


def params_from_items(items: list[SeedResult]) -> dict[str, Any]:
    for item in items:
        cfg = item.config
        if cfg:
            return {
                "batch_size": cfg.get("batch_size"),
                "learning_rate": cfg.get("learning_rate"),
                "bce_weight": cfg.get("bce_weight"),
                "dice_weight": cfg.get("dice_weight"),
                "normalisation": cfg.get("normalisation"),
                "noise_augment": cfg.get("noise_augment"),
                "noise_std_multiplier": cfg.get("noise_std_multiplier"),
            }
    return {}


def select_trial(rows: list[dict[str, Any]], tie_delta: float) -> dict[str, Any]:
    valid = [row for row in rows if row["val_f1_mean"] is not None]
    if not valid:
        raise SystemExit("no valid restudy trials to select")
    best_mean = max(float(row["val_f1_mean"]) for row in valid)
    tied = [row for row in valid if best_mean - float(row["val_f1_mean"]) <= tie_delta]
    return sorted(
        tied,
        key=lambda row: (
            int(row.get("params", {}).get("batch_size") or 10**9),
            float(row.get("params", {}).get("learning_rate") or math.inf),
            int(row["trial"]),
        ),
    )[0]


def aggregate(
    results: list[SeedResult],
    expected_trials: set[int] | None,
    expected_seeds: set[int],
    n_resamples: int,
    bootstrap_seed: int,
    ci_level: float,
    tie_delta: float,
    selection_scope: str,
) -> dict[str, Any]:
    by_trial: dict[int, list[SeedResult]] = {}
    for item in results:
        by_trial.setdefault(item.trial, []).append(item)

    if expected_trials is not None:
        missing_trials = sorted(expected_trials.difference(by_trial))
        if missing_trials:
            raise SystemExit(f"missing expected trials: {missing_trials}")
        trial_order = sorted(expected_trials)
    else:
        trial_order = sorted(by_trial)

    alpha = 1.0 - ci_level
    trial_rows: list[dict[str, Any]] = []
    for trial in trial_order:
        items = sorted(by_trial.get(trial, []), key=lambda r: r.seed)
        prefixes = sorted({item.tag_prefix for item in items})
        if len(prefixes) != 1:
            raise SystemExit(f"trial {trial} has mixed tag prefixes: {prefixes}")
        seeds_present = {item.seed for item in items}
        missing_seeds = sorted(expected_seeds.difference(seeds_present))
        values = [float(item.val_f1) for item in items if item.val_f1 is not None]
        ci_lo, ci_hi = bootstrap_ci(values, bootstrap_seed + trial, n_resamples, alpha)
        failure_cases = [
            item.tag
            for item in items
            if item.val_f1 is None or item.nan_loss or item.final_val_dice is None
        ]
        row = {
            "trial": trial,
            "tag_prefix": prefixes[0] if prefixes else None,
            "params": params_from_items(items),
            "seeds": [item.seed for item in items],
            "missing_seeds": missing_seeds,
            "val_f1_values": values,
            "val_f1_mean": float(np.mean(values)) if values else None,
            "val_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else None,
            "val_f1_ci_level": ci_level,
            "val_f1_ci_lo": ci_lo if values else None,
            "val_f1_ci_hi": ci_hi if values else None,
            "failure_count": len(failure_cases) + len(missing_seeds),
            "failure_cases": failure_cases,
            "max_consecutive_zero_val_dice": max((item.max_consecutive_zero_val_dice for item in items), default=0),
            "val_loss_spike_count_2x_total": sum(item.val_loss_spike_count_2x for item in items),
            "seed_results": [item.__dict__ for item in items],
        }
        trial_rows.append(row)

    selected = select_trial(trial_rows, tie_delta)
    return {
        "protocol_record": "docs/reproduction_map.rst",
        "selection_scope": selection_scope,
        "selection_rule": (
            "highest mean threshold-swept validation val_f1; if candidates are within "
            f"{tie_delta} val_f1, prefer lower batch_size, then lower learning_rate, "
            "then lower trial number; test metrics are inadmissible"
        ),
        "n_resamples": n_resamples,
        "bootstrap_seed": bootstrap_seed,
        "trials": trial_rows,
        "selected_trial": selected["trial"],
        "selected_tag_prefix": f"{selected['tag_prefix']}_t{selected['trial']}",
        "selected_mean_val_f1": selected["val_f1_mean"],
        "selected_ci_lo": selected["val_f1_ci_lo"],
        "selected_ci_hi": selected["val_f1_ci_hi"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="results/classical/threshold_sweep_unet_paper_arch_noise_topk_t*_s*.json")
    parser.add_argument("--summary-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("results/classical/restudy_topk_summary.json"))
    parser.add_argument("--expected-trials", default=None)
    parser.add_argument("--expected-seeds", default="2804,1234,42,7,13")
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2804)
    parser.add_argument("--ci-level", type=float, default=0.90)
    parser.add_argument("--tie-delta", type=float, default=0.001)
    parser.add_argument(
        "--selection-scope",
        default="top-K x five-seed validation-only restudy",
        help="Human-readable scope recorded in the summary provenance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.pattern, args.summary_dir)
    if not results:
        raise SystemExit(f"no restudy threshold JSONs matched {args.pattern!r}")
    payload = aggregate(
        results=results,
        expected_trials=parse_ints(args.expected_trials),
        expected_seeds=parse_ints(args.expected_seeds) or set(),
        n_resamples=args.n_resamples,
        bootstrap_seed=args.bootstrap_seed,
        ci_level=args.ci_level,
        tie_delta=args.tie_delta,
        selection_scope=args.selection_scope,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"Saved {args.out}")
    print("trial  bs  mean_val_f1  ci_lo  ci_hi  failures")
    for row in payload["trials"]:
        print(
            f"{row['trial']:>5}  {row['params'].get('batch_size')!s:>2}  "
            f"{row['val_f1_mean']!s:>11}  {row['val_f1_ci_lo']!s:>7}  "
            f"{row['val_f1_ci_hi']!s:>7}  {row['failure_count']}"
        )
    print(
        f"selected_trial: {payload['selected_trial']}  "
        f"mean_val_f1={payload['selected_mean_val_f1']}"
    )


if __name__ == "__main__":
    main()
