"""Run the classical Hough-transform baseline on processed image-mask pairs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.classical.hough import HoughTransformConfig, evaluate_hough_baseline
from src.utils.logger import get_logger


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the Hough baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/subset/processed"),
        help="Directory containing processed image and mask PNG pairs.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("results/classical/hough_metrics.csv"),
        help="CSV path for per-image metrics.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/logs"),
        help="Directory where runtime logs will be saved.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--canny-low-threshold", type=int, default=50)
    parser.add_argument("--canny-high-threshold", type=int, default=150)
    parser.add_argument("--hough-threshold", type=int, default=50)
    parser.add_argument("--min-line-length", type=int, default=40)
    parser.add_argument("--max-line-gap", type=int, default=10)
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--clip-low-percentile", type=float, default=1.0)
    parser.add_argument("--clip-high-percentile", type=float, default=99.5)
    return parser


def main() -> None:
    """Parse CLI arguments, run the baseline, and save the metrics CSV."""
    args = build_parser().parse_args()
    logger = get_logger(name="hough_baseline", run_dir=args.log_dir)
    config = HoughTransformConfig(
        canny_low_threshold=args.canny_low_threshold,
        canny_high_threshold=args.canny_high_threshold,
        hough_threshold=args.hough_threshold,
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
        line_thickness=args.line_thickness,
        clip_low_percentile=args.clip_low_percentile,
        clip_high_percentile=args.clip_high_percentile,
    )
    logger.info("Running Hough baseline with config: %s", asdict(config))
    result = evaluate_hough_baseline(
        root_dir=args.data_root,
        config=config,
        limit=args.limit,
        logger=logger,
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    result.per_image_metrics.to_csv(args.metrics_path, index=False)
    logger.info("Saved Hough metrics to %s", args.metrics_path)
    logger.info("Summary counts: %s", asdict(result.summary_counts))
    logger.info("Summary metrics: %s", asdict(result.summary_metrics))


if __name__ == "__main__":
    main()
