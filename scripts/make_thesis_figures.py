#!/usr/bin/env python
"""Build vector thesis figures from locked result artifacts."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config.constants import GLOBAL_SEED, PAPER_PRECISION, PAPER_RECALL

ROOT = Path(__file__).parent.parent
CLASSICAL = ROOT / "results" / "classical"
CHECKPOINTS = ROOT / "results" / "checkpoints"
FIGURES = ROOT / "results" / "figures"
SEEDS = [2804, 1234, 42, 7, 13]
METRICS = ["precision", "recall", "f1", "dice"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "dice": "Dice",
}
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
PAPER_F1 = (2.0 * PAPER_PRECISION * PAPER_RECALL) / (PAPER_PRECISION + PAPER_RECALL)
PAPER_VALUES = {
    "precision": PAPER_PRECISION,
    "recall": PAPER_RECALL,
    "f1": PAPER_F1,
    "dice": PAPER_F1,
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _strip_svg_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def save_vector(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("pdf", "svg"):
        out = FIGURES / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        if ext == "svg":
            _strip_svg_trailing_whitespace(out)
        outputs.append(out)
    plt.close(fig)
    return outputs


def bootstrap_mean_ci(values: list[float], seed: int = GLOBAL_SEED, n_resamples: int = 10000) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, math.nan, math.nan
    rng = np.random.default_rng(seed)
    sample_means = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(sample_means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def metric_value(data: dict[str, Any], metric: str) -> float:
    if metric == "f1":
        p = float(data["point_metrics"]["precision"])
        r = float(data["point_metrics"]["recall"])
        return 2.0 * p * r / (p + r) if (p + r) else 0.0
    return float(data["point_metrics"][metric])


def metric_series(prefix: str) -> dict[str, list[float]]:
    rows = [load_json(CLASSICAL / f"bootstrap_{prefix}_s{seed}.json") for seed in SEEDS]
    return {metric: [metric_value(row, metric) for row in rows] for metric in METRICS}


def figure_multiseed_replication() -> None:
    series = metric_series("winner_t44")
    x = np.arange(len(METRICS))
    means, lows, highs = [], [], []
    for idx, metric in enumerate(METRICS):
        mean, lo, hi = bootstrap_mean_ci(series[metric], seed=GLOBAL_SEED + idx)
        means.append(mean)
        lows.append(mean - lo)
        highs.append(hi - mean)

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    seed_handle = ax.bar(x, means, color=COLORS["blue"], alpha=0.75, width=0.58, label="Seed mean")
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", color=COLORS["black"], capsize=3, lw=0.9)
    for i, metric in enumerate(METRICS):
        jitter = np.linspace(-0.16, 0.16, len(SEEDS))
        ax.scatter(
            np.full(len(SEEDS), x[i]) + jitter, series[metric],
            s=16, color=COLORS["black"], alpha=0.7, zorder=3,
        )
        ax.scatter(x[i], PAPER_VALUES[metric], marker="D", s=24, color=COLORS["orange"], zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Test score")
    ax.set_ylim(0.78, 0.97)
    ax.set_title("Five-seed replication vs paper reference")
    paper_handle = ax.scatter([], [], marker="D", color=COLORS["orange"], label="Paper reference")
    ax.legend(
        [seed_handle, paper_handle],
        ["Seed mean", "Paper reference"],
        loc="upper right", ncol=2, frameon=True, framealpha=0.86, edgecolor="none",
    )
    save_vector(fig, "thesis_1_multiseed_replication")

def figure_boundary_tolerance() -> None:
    rows = [load_json(CLASSICAL / f"boundary_tolerant_unet_paper_arch_noise_topk_t44_s{seed}.json") for seed in SEEDS]
    tolerances = [0, 1, 2, 3]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    styles = [("precision", COLORS["blue"]), ("recall", COLORS["green"]), ("f1", COLORS["orange"])]
    for offset, (metric, color) in enumerate(styles):
        means, los, his = [], [], []
        for tol in tolerances:
            vals = [float(row["per_tolerance"][str(tol)][metric]) for row in rows]
            mean, lo, hi = bootstrap_mean_ci(vals, seed=GLOBAL_SEED + 10 * tol + offset)
            means.append(mean); los.append(lo); his.append(hi)
        means_arr = np.asarray(means)
        ax.plot(tolerances, means_arr, marker="o", color=color, label=METRIC_LABELS.get(metric, "F1"))
        ax.fill_between(tolerances, los, his, color=color, alpha=0.16, linewidth=0)
    ax.axhline(PAPER_PRECISION, color=COLORS["black"], linestyle="--", lw=0.9, label="Paper precision (0.94)")
    ax.set_xlabel("Boundary tolerance (px)")
    ax.set_ylabel("Micro-averaged score")
    ax.set_xticks(tolerances)
    ax.set_ylim(0.82, 1.0)
    ax.set_title("Precision/recall/F1 under boundary tolerance")
    ax.legend(loc="lower right", frameon=False)
    save_vector(fig, "thesis_2_boundary_tolerance")


def build_data_efficiency_json() -> dict[str, Any]:
    specs = [
        (30, CHECKPOINTS / "unet_data_efficiency_f30_s2804_summary.json"),
        (50, CHECKPOINTS / "unet_data_efficiency_f50_s2804_summary.json"),
        (70, CHECKPOINTS / "unet_data_efficiency_f70_s2804_summary.json"),
        (100, CHECKPOINTS / "unet_paper_arch_noise_topk_t44_s2804_summary.json"),
    ]
    points = []
    for fraction, path in specs:
        data = load_json(path)
        history = data.get("history", [])
        if not history:
            raise ValueError(f"{path} has no history")
        best = max(history, key=lambda row: float(row["val_dice"]))
        cfg = data.get("config", {})
        points.append({
            "train_fraction_percent": fraction,
            "summary": str(path.relative_to(ROOT)),
            "best_checkpoint": data.get("best_checkpoint"),
            "seed": cfg.get("seed", GLOBAL_SEED),
            "best_epoch": int(best["epoch"]),
            "val_dice_at_threshold_0p5": float(best["val_dice"]),
            "val_f1_at_threshold_0p5": float(best["val_dice"]),
            "val_iou_at_threshold_0p5": float(best["val_iou"]),
        })
    payload = {
        "split": "validation",
        "seed_protocol": "single seed 2804 at each fraction",
        "threshold": 0.5,
        "metric_convention": (
            "Best per-epoch validation Dice (= pixel F1) at the fixed training threshold 0.5. "
            "This is not directly comparable to validation-optimal threshold-swept val_f1 elsewhere."
        ),
        "points": points,
    }
    out = CLASSICAL / "data_efficiency_curve.json"
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def figure_data_efficiency(payload: dict[str, Any]) -> None:
    points = payload["points"]
    x = [p["train_fraction_percent"] for p in points]
    y = [p["val_dice_at_threshold_0p5"] for p in points]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(x, y, marker="o", color=COLORS["blue"], lw=1.4)
    ax.set_xlabel("Training images used (%)")
    ax.set_ylabel("Validation Dice @ 0.5")
    ax.set_xticks(x)
    ax.set_ylim(min(y) - 0.01, max(y) + 0.01)
    ax.set_title("Data-efficiency curve")
    fig.subplots_adjust(bottom=0.16)
    save_vector(fig, "thesis_4_data_efficiency")

def figure_two_stage_pareto() -> None:
    data = load_json(CLASSICAL / "two_stage_t44_s2804.json")
    pareto = data["pareto_front"]
    recall = [p["end_to_end_recall"] for p in pareto]
    precision = [p["end_to_end_precision"] for p in pareto]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(recall, precision, color=COLORS["blue"], lw=1.1, label="Two-stage Pareto")
    ax.scatter([data["baseline_unet_recall"]], [data["baseline_unet_precision"]], marker="D", s=35, color=COLORS["orange"], label="Single-stage baseline")
    ax.scatter([data["end_to_end_recall"]], [data["end_to_end_precision"]], marker="o", s=35, color=COLORS["green"], label="Locked two-stage point")
    dp = data["end_to_end_precision"] - data["baseline_unet_precision"]
    dr = data["end_to_end_recall"] - data["baseline_unet_recall"]
    ax.annotate(
        f"gate: {dp * 100:+.1f} pp precision,\n{dr * 100:+.1f} pp recall",
        xy=(data["end_to_end_recall"], data["end_to_end_precision"]),
        xytext=(0.885, 0.812), textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": COLORS["black"]},
        fontsize=6.5, ha="right", va="top",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.70, 0.98)
    ax.set_ylim(0.72, 0.98)
    ax.set_title("Classifier-gated operating characteristic")
    ax.legend(frameon=False, loc="upper right")
    save_vector(fig, "thesis_5_two_stage_pareto")


def load_optuna_trials() -> list[dict[str, Any]]:
    db = CLASSICAL / "unet_paper_arch_noise_f1.db"
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT t.trial_id, t.number, t.state, v.value AS objective_value
        FROM trials t
        LEFT JOIN trial_values v ON t.trial_id = v.trial_id
        ORDER BY t.number
        """
    ).fetchall()
    attrs = con.execute("SELECT trial_id, key, value_json FROM trial_user_attributes").fetchall()
    con.close()
    by_trial: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_trial[int(row["trial_id"])] = {
            "trial_id": int(row["trial_id"]),
            "trial": int(row["number"]),
            "state": row["state"],
            "objective_value": None if row["objective_value"] is None else float(row["objective_value"]),
        }
    for attr in attrs:
        trial_id = int(attr["trial_id"])
        if trial_id not in by_trial:
            continue
        value = json.loads(attr["value_json"])
        by_trial[trial_id][attr["key"]] = value
    trials = list(by_trial.values())
    if len(trials) != 45:
        raise ValueError(f"expected 45 Optuna trials, found {len(trials)}")
    return trials


def _trial_val_f1(trial: dict[str, Any]) -> float:
    return float(trial.get("val_f1", trial["objective_value"]))


def figure_sweep_selection() -> None:
    trials = load_optuna_trials()
    complete = [t for t in trials if t["state"] == "COMPLETE" and t.get("val_f1") is not None]
    pruned = [t for t in trials if t["state"] == "PRUNED"]
    batches = [8, 16, 32]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    rng = np.random.default_rng(GLOBAL_SEED)
    for idx, batch in enumerate(batches):
        vals = [_trial_val_f1(t) for t in complete if int(t["batch_size"]) == batch]
        xs = idx + rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(xs, vals, s=18, alpha=0.75, color=COLORS["blue"], label="Completed" if idx == 0 else None)
        pruned_vals = [float(t["objective_value"]) for t in pruned if int(t["batch_size"]) == batch and t.get("objective_value") is not None]
        if pruned_vals:
            pxs = idx + rng.uniform(-0.12, 0.12, size=len(pruned_vals))
            ax.scatter(
                pxs, pruned_vals, s=18, facecolors="none", edgecolors="#777777",
                alpha=0.8, label="Pruned" if idx == 0 else None,
            )
        mean, lo, hi = bootstrap_mean_ci(vals, seed=GLOBAL_SEED + batch)
        ax.errorbar(
            [idx], [mean], yerr=[[mean - lo], [hi - mean]], fmt="s",
            color=COLORS["orange"], capsize=3, lw=0.9, zorder=4,
            label="Completed mean" if idx == 0 else None,
        )
    selected = next(t for t in trials if int(t["trial"]) == 44)
    selected_x = batches.index(int(selected["batch_size"]))
    ax.scatter([selected_x], [_trial_val_f1(selected)], marker="*", s=90, color=COLORS["red"], label="Selected trial 44", zorder=5)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Validation F1 objective")
    ax.set_title("Balanced Optuna sweep by batch size")
    ax.legend(frameon=True, framealpha=0.86, edgecolor="none", loc="lower right")
    fig.subplots_adjust(bottom=0.16)
    save_vector(fig, "thesis_7_sweep_selection")

def main() -> None:
    configure_style()
    figure_multiseed_replication()
    figure_boundary_tolerance()
    data_eff = build_data_efficiency_json()
    figure_data_efficiency(data_eff)
    figure_two_stage_pareto()
    figure_sweep_selection()
    print("Saved thesis figures to", FIGURES)
    print("Saved", CLASSICAL / "data_efficiency_curve.json")


if __name__ == "__main__":
    main()
