"""Run the classical Hough-transform baseline on processed image-mask pairs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from src.classical.hough import HoughTransformConfig, evaluate_hough_baseline
from src.utils.logger import get_logger


def parse_args() -> dict:
    """Load run parameters from a YAML file given by --config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config file (e.g. configs/experiments/hough_baseline.yaml).",
    )
    args = parser.parse_args()
    with open(args.config) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Load config, run the Hough baseline, and save per-image and summary metrics."""
    cfg = parse_args()

    data_root = Path(cfg.get("data_root", "data/Processed"))
    metrics_path = Path(cfg.get("metrics_path", "results/classical/hough_metrics.csv"))
    summary_path = Path(cfg["summary_path"]) if "summary_path" in cfg else None
    log_dir = Path(cfg.get("log_dir", "results/logs"))
    limit = cfg.get("limit", None)

    hough_keys = {
        "canny_low_threshold", "canny_high_threshold", "hough_threshold",
        "min_line_length", "max_line_gap", "line_thickness",
        "clip_low_percentile", "clip_high_percentile",
    }
    config = HoughTransformConfig(**{k: cfg[k] for k in hough_keys if k in cfg})

    logger = get_logger(name="hough_baseline", run_dir=log_dir)
    logger.info("Running Hough baseline with config: %s", asdict(config))

    result = evaluate_hough_baseline(
        root_dir=data_root,
        config=config,
        limit=limit,
        logger=logger,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    result.per_image_metrics.to_csv(metrics_path, index=False)
    logger.info(
        "Aggregated: precision=%.4f recall=%.4f dice=%.4f iou=%.4f",
        result.summary_metrics.precision,
        result.summary_metrics.recall,
        result.summary_metrics.dice,
        result.summary_metrics.iou,
    )

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": asdict(config),
            "data_root": str(data_root),
            "counts": asdict(result.summary_counts),
            "metrics": asdict(result.summary_metrics),
        }
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
        logger.info("Saved summary JSON to %s", summary_path)


if __name__ == "__main__":
    main()
