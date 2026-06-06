#!/usr/bin/env python
"""Summarise hard-negative mining against the locked same-seed baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="results/classical/threshold_sweep_winner_t44_s2804.json",
    )
    parser.add_argument(
        "--hard-negative",
        default="results/classical/threshold_sweep_hard_negative_t44_s2804.json",
    )
    parser.add_argument(
        "--mined",
        default="results/classical/hard_negative_train_t44_s2804.json",
    )
    parser.add_argument(
        "--fp-decomposition",
        default="results/classical/fp_decomposition_unet_paper_arch_noise_topk_t44_s2804.json",
    )
    parser.add_argument(
        "--out",
        default="results/classical/hard_negative_t44_s2804_summary.json",
    )
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _test_metrics(payload: dict[str, Any]) -> dict[str, float]:
    precision = float(payload["test_precision"])
    recall = float(payload["test_recall"])
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": float(payload["test_dice"]),
        "iou": float(payload["test_iou"]),
    }


def main() -> None:
    args = parse_args()
    baseline = _load(args.baseline)
    hard = _load(args.hard_negative)
    mined = _load(args.mined)
    fp_decomp = _load(args.fp_decomposition)

    baseline_metrics = _test_metrics(baseline)
    hard_metrics = _test_metrics(hard)
    deltas = {
        key: hard_metrics[key] - baseline_metrics[key]
        for key in ("precision", "recall", "f1", "dice", "iou")
    }
    inter_fraction = float(fp_decomp["inter_patch_fp_fraction"])
    out = {
        "baseline": {
            "tag": baseline.get("tag"),
            "checkpoint": baseline.get("checkpoint"),
            "optimal_threshold": baseline.get("optimal_threshold"),
            "test_metrics": baseline_metrics,
        },
        "hard_negative": {
            "tag": hard.get("tag"),
            "checkpoint": hard.get("checkpoint"),
            "optimal_threshold": hard.get("optimal_threshold"),
            "val_f1": hard.get("val_f1"),
            "test_metrics": hard_metrics,
        },
        "delta_hard_negative_minus_baseline": deltas,
        "mining": {
            "train_manifest": str(args.mined),
            "threshold": mined.get("threshold"),
            "selected_count": mined.get("selected_count"),
            "candidate_empty_patches": mined.get("candidate_empty_patches"),
            "false_positive_empty_patches": mined.get("false_positive_empty_patches"),
        },
        "fp_decomposition_context": {
            "inter_patch_fp_fraction": inter_fraction,
            "intra_patch_fp_fraction": 1.0 - inter_fraction,
            "interpretation": fp_decomp.get("interpretation"),
            "limitation": (
                "Patch-level hard-negative mining is structurally bounded because "
                f"only {100.0 * inter_fraction:.1f}% of false-positive pixels are inter-patch; "
                "the remaining false positives arise within trail-containing patches."
            ),
        },
        "generated": date.today().isoformat(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(
        "Hard-negative delta vs baseline: "
        f"P={deltas['precision']:+.4f} R={deltas['recall']:+.4f} "
        f"F1={deltas['f1']:+.4f} Dice={deltas['dice']:+.4f} -> {out_path}"
    )


if __name__ == "__main__":
    main()
