from pathlib import Path

import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")

from src.classical.hough import HoughTransformConfig, evaluate_hough_baseline  # noqa: E402
from tests.helpers import write_pair as _write_pair  # noqa: E402


def test_hough_baseline_detects_a_simple_synthetic_trail(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    image_array = np.zeros((64, 64), dtype=np.uint8)
    mask_array = np.zeros((64, 64), dtype=np.uint8)
    cv2.line(image_array, (6, 32), (58, 32), color=255, thickness=2)
    cv2.line(mask_array, (6, 32), (58, 32), color=255, thickness=2)
    _write_pair(processed_dir, "synthetic", image_array, mask_array)

    result = evaluate_hough_baseline(
        root_dir=processed_dir,
        config=HoughTransformConfig(
            canny_low_threshold=10,
            canny_high_threshold=40,
            hough_threshold=10,
            min_line_length=20,
            max_line_gap=5,
            line_thickness=2,
        ),
    )

    assert len(result.per_image_metrics) == 1
    assert result.summary_counts.true_positive > 0
    assert result.summary_metrics.recall > 0.0
