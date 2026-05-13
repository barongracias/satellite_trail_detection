"""Classical Hough-transform baseline for satellite trail segmentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.indexing import PatchDatasetConfig, discover_image_mask_pairs
from src.evaluation.segmentation import (
    SegmentationCounts,
    SegmentationMetrics,
    combine_counts,
    compute_metrics_from_counts,
    compute_segmentation_counts,
)


Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class HoughTransformConfig:
    """Collect the configurable parameters for the Hough baseline."""

    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    hough_threshold: int = 50
    min_line_length: int = 40
    max_line_gap: int = 10
    line_thickness: int = 3
    clip_low_percentile: float = 1.0
    clip_high_percentile: float = 99.5

    def __post_init__(self) -> None:
        """Validate the small set of options that would break OpenCV."""
        if self.canny_low_threshold >= self.canny_high_threshold:
            raise ValueError("canny_low_threshold must be smaller than canny_high_threshold")
        if self.hough_threshold <= 0 or self.min_line_length <= 0:
            raise ValueError("Hough thresholds and line length must be positive")
        if self.line_thickness <= 0:
            raise ValueError("line_thickness must be positive")


@dataclass(frozen=True)
class HoughEvaluationResult:
    """Store per-image and aggregated evaluation results for the Hough baseline."""

    summary_counts: SegmentationCounts
    summary_metrics: SegmentationMetrics
    per_image_metrics: pd.DataFrame


def _load_grayscale_array(path: Path) -> np.ndarray:
    """Load a grayscale PNG as a NumPy array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _stretch_image(
    image: np.ndarray,
    config: HoughTransformConfig,
) -> np.ndarray:
    """Apply percentile clipping and rescale to 8-bit for edge detection."""
    lower = float(np.percentile(image, config.clip_low_percentile))
    upper = float(np.percentile(image, config.clip_high_percentile))
    if upper <= lower:
        return np.zeros_like(image, dtype=np.uint8)

    clipped = np.clip(image.astype(np.float32), lower, upper)
    stretched = (clipped - lower) / (upper - lower)
    return np.round(stretched * 255.0).astype(np.uint8)


def detect_trail_mask(
    image: np.ndarray,
    config: HoughTransformConfig | None = None,
) -> np.ndarray:
    """Detect likely satellite-trail pixels using Canny edges and Hough lines."""
    hough_config = HoughTransformConfig() if config is None else config
    stretched = _stretch_image(image, hough_config)
    blurred = cv2.GaussianBlur(stretched, (5, 5), 0)
    edges = cv2.Canny(
        blurred,
        threshold1=hough_config.canny_low_threshold,
        threshold2=hough_config.canny_high_threshold,
    )
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_config.hough_threshold,
        minLineLength=hough_config.min_line_length,
        maxLineGap=hough_config.max_line_gap,
    )

    prediction = np.zeros_like(stretched, dtype=np.uint8)
    if lines is None:
        return prediction

    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(value) for value in line]
        cv2.line(
            prediction,
            (x1, y1),
            (x2, y2),
            color=255,
            thickness=hough_config.line_thickness,
        )

    return prediction


def evaluate_hough_baseline(
    root_dir: str | Path,
    config: HoughTransformConfig | None = None,
    limit: int | None = None,
    logger: logging.Logger | None = None,
) -> HoughEvaluationResult:
    """Run the classical baseline on processed image-mask pairs and score it."""
    dataset_config = PatchDatasetConfig(root_dir=Path(root_dir), patch_size=1)
    hough_config = HoughTransformConfig() if config is None else config
    pairs = discover_image_mask_pairs(dataset_config)
    if limit is not None:
        pairs = pairs[:limit]

    all_counts: list[SegmentationCounts] = []
    rows = []

    for pair in pairs:
        if logger is not None:
            logger.info("Running Hough baseline for %s", pair.image_path.name)

        image = _load_grayscale_array(pair.image_path)
        target_mask = _load_grayscale_array(pair.mask_path)
        prediction_mask = detect_trail_mask(image, hough_config)

        counts = compute_segmentation_counts(prediction_mask, target_mask)
        metrics = compute_metrics_from_counts(counts)
        all_counts.append(counts)
        rows.append(
            {
                "image_name": pair.image_path.name,
                "true_positive": counts.true_positive,
                "false_positive": counts.false_positive,
                "true_negative": counts.true_negative,
                "false_negative": counts.false_negative,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "dice": metrics.dice,
                "iou": metrics.iou,
                "accuracy": metrics.accuracy,
                "specificity": metrics.specificity,
            }
        )

    combined_counts = combine_counts(all_counts)
    return HoughEvaluationResult(
        summary_counts=combined_counts,
        summary_metrics=compute_metrics_from_counts(combined_counts),
        per_image_metrics=pd.DataFrame(rows),
    )
