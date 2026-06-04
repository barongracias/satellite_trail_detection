"""Grid search over Hough parameters on a subset of images.

Writes results to results/classical/hough_tuning.csv and prints a summary table.
Run with: python scripts/tune_hough.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import itertools

from src.classical.hough import HoughTransformConfig, evaluate_hough_baseline
from src.utils.logger import get_logger

LIMIT = 20  # images per trial — fast enough for a grid search
DATA_ROOT = Path("data/Processed")
OUTPUT_CSV = Path("results/classical/hough_tuning.csv")
LOG_DIR = Path("results/logs")

GRID = {
    "hough_threshold":    [50, 100, 200],
    "min_line_length":    [500, 1000, 2000, 5000],
    "canny_low_threshold":  [50],
    "canny_high_threshold": [150],
}


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    logger = get_logger(name="hough_tuning", run_dir=LOG_DIR)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    logger.info("Running %d parameter combinations on %d images each", len(combos), LIMIT)

    rows = []
    best_f1, best_cfg = -1.0, None

    for values in combos:
        params = dict(zip(keys, values))
        cfg = HoughTransformConfig(**params)
        logger.info("Trial: %s", params)

        result = evaluate_hough_baseline(
            root_dir=DATA_ROOT,
            config=cfg,
            limit=LIMIT,
            logger=None,
        )
        m = result.summary_metrics
        trial_f1 = f1(m.precision, m.recall)

        row = {**params, "precision": m.precision, "recall": m.recall,
               "f1": trial_f1, "dice": m.dice, "iou": m.iou}
        rows.append(row)

        logger.info(
            "  precision=%.4f recall=%.4f f1=%.4f dice=%.4f",
            m.precision, m.recall, trial_f1, m.dice,
        )

        if trial_f1 > best_f1:
            best_f1, best_cfg = trial_f1, params

    # Write CSV
    fieldnames = keys + ["precision", "recall", "f1", "dice", "iou"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Results saved to %s", OUTPUT_CSV)
    logger.info("Best config (F1=%.4f): %s", best_f1, best_cfg)

    print("\n=== TOP 5 CONFIGS BY F1 ===")
    for r in sorted(rows, key=lambda x: x["f1"], reverse=True)[:5]:
        print(
            f"  threshold={r['hough_threshold']:3d}  min_len={r['min_line_length']:5d}"
            f"  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}"
        )
    print(f"\nBest: {best_cfg}  F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
