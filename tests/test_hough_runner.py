"""Equivalence tests for ``src/classical/hough_runner.py``.

These tests guarantee that ``run_hough_on_canvas`` is bitwise-equivalent
to the inline logic currently at ``scripts/hough_postprocess.py:280-329``.

The inline logic is replicated here as the reference implementation. When
``scripts/hough_postprocess.py`` is later refactored to call
``run_hough_on_canvas``, the refactor is "no behaviour change" iff these
tests still pass — they pin every operation that produces the canvases,
detection flags, and pixel counts to the original arithmetic.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.classical.hough_runner import (
    HoughCanvasResult,
    run_hough_on_canvas,
)


_HOUGH_PARAMS: dict = {
    "threshold": 0.58,
    "hough_input_threshold": 0.10,
    "hough_threshold": 50,
    "min_line_length": 100,
    "max_line_gap": 250,
    "line_thickness": 3,
}


# --- reference implementation (cherry-picked verbatim from
# scripts/hough_postprocess.py lines 280-329; do not modify without also
# updating the inline logic, or vice versa) -----------------------------------


def _reference_apply_hough(
    canvas: np.ndarray,
    hough_threshold: int,
    min_line_length: int,
    max_line_gap: int,
    line_thickness: int,
) -> np.ndarray:
    lines = cv2.HoughLinesP(
        canvas,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    result = np.zeros_like(canvas)
    if lines is None:
        return result
    for line in lines[:, 0]:
        x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        cv2.line(result, (x1, y1), (x2, y2), color=255, thickness=line_thickness)
    return result


def _reference_logic(
    prob_canvas: np.ndarray,
    target_canvas: np.ndarray,
    *,
    threshold: float,
    hough_input_threshold: float,
    hough_threshold: int,
    min_line_length: int,
    max_line_gap: int,
    line_thickness: int,
) -> dict:
    """Inline logic from scripts/hough_postprocess.py:280-329."""
    binary_canvas = (prob_canvas >= threshold).astype(np.uint8) * 255
    detected_pre = bool(
        (binary_canvas > 0).any() and (binary_canvas & target_canvas).any()
    )

    hough_input = (prob_canvas >= hough_input_threshold).astype(np.uint8) * 255
    hough_canvas = _reference_apply_hough(
        hough_input,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
        line_thickness=line_thickness,
    )
    detected_post = bool(((binary_canvas | hough_canvas) & target_canvas).any())

    binary_bool = binary_canvas > 0
    target_bool = target_canvas > 0
    combined_bool = (binary_canvas | hough_canvas) > 0

    gt_pixels = int(target_bool.sum())
    pixels_pre = int((binary_bool & target_bool).sum())
    pixels_post = int((combined_bool & target_bool).sum())

    return {
        "binary_canvas": binary_canvas,
        "hough_canvas": hough_canvas,
        "combined_canvas": binary_canvas | hough_canvas,
        "detected_pre": detected_pre,
        "detected_post": detected_post,
        "gt_pixels": gt_pixels,
        "pixels_pre": pixels_pre,
        "pixels_post": pixels_post,
    }


def _assert_equivalent(helper: HoughCanvasResult, reference: dict) -> None:
    """Assert that the helper result is bitwise-equal to the reference."""
    np.testing.assert_array_equal(helper.binary_canvas, reference["binary_canvas"])
    np.testing.assert_array_equal(helper.hough_canvas, reference["hough_canvas"])
    np.testing.assert_array_equal(helper.combined_canvas, reference["combined_canvas"])
    assert helper.detected_pre is reference["detected_pre"]
    assert helper.detected_post is reference["detected_post"]
    assert helper.gt_pixels == reference["gt_pixels"]
    assert helper.pixels_pre == reference["pixels_pre"]
    assert helper.pixels_post == reference["pixels_post"]


# --- synthetic fixtures ------------------------------------------------------


def _make_diagonal_trail(size: int = 1056, intensity: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 'image' with a diagonal trail in both prob_canvas and gt.

    The trail is a thick diagonal line; this guarantees Hough has enough
    aligned points to register a detection at the default thresholds.
    """
    rng = np.random.default_rng(2804)
    prob = rng.uniform(0.0, 0.05, size=(size, size)).astype(np.float32)
    gt = np.zeros((size, size), dtype=np.uint8)
    for offset in range(-3, 4):
        for i in range(100, size - 100):
            j = i + offset
            if 0 <= j < size:
                prob[i, j] = intensity
                gt[i, j] = 255
    return prob, gt


def _make_empty_canvas(size: int = 528) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic empty 'image' — no trail in either prob or gt."""
    prob = np.zeros((size, size), dtype=np.float32)
    gt = np.zeros((size, size), dtype=np.uint8)
    return prob, gt


def _make_faint_trail_below_main_threshold(size: int = 1056) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic faint trail: prob values between hough_input and main threshold.

    This is the regime the Hough recovery step is designed to handle:
    pixel-level overlap pre-Hough is zero (nothing crosses the main
    threshold), but post-Hough should recover the diagonal.
    """
    rng = np.random.default_rng(42)
    prob = rng.uniform(0.0, 0.05, size=(size, size)).astype(np.float32)
    gt = np.zeros((size, size), dtype=np.uint8)
    # Faint values between hough_input_threshold (0.10) and threshold (0.58).
    faint = 0.30
    for offset in range(-2, 3):
        for i in range(150, size - 150):
            j = i + offset
            if 0 <= j < size:
                prob[i, j] = faint
                gt[i, j] = 255
    return prob, gt


def _make_false_positive_only(size: int = 528) -> tuple[np.ndarray, np.ndarray]:
    """Bright diagonal in prob, but gt is empty — pure false positive.

    Image-level detection (which requires overlap with GT) must be False
    both pre- and post-Hough.
    """
    rng = np.random.default_rng(7)
    prob = rng.uniform(0.0, 0.05, size=(size, size)).astype(np.float32)
    for offset in range(-3, 4):
        for i in range(50, size - 50):
            j = i + offset
            if 0 <= j < size:
                prob[i, j] = 0.95
    gt = np.zeros((size, size), dtype=np.uint8)
    return prob, gt


# --- equivalence tests -------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_fn",
    [
        _make_diagonal_trail,
        _make_empty_canvas,
        _make_faint_trail_below_main_threshold,
        _make_false_positive_only,
    ],
)
def test_equivalent_to_reference(fixture_fn) -> None:
    """The helper produces bitwise-identical output to the inline reference."""
    prob, gt = fixture_fn()
    helper = run_hough_on_canvas(prob, gt, **_HOUGH_PARAMS)
    reference = _reference_logic(prob, gt, **_HOUGH_PARAMS)
    _assert_equivalent(helper, reference)


# --- contract / shape / threshold validation ---------------------------------


def test_shape_mismatch_raises() -> None:
    prob = np.zeros((10, 20), dtype=np.float32)
    gt = np.zeros((10, 30), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        run_hough_on_canvas(prob, gt, **_HOUGH_PARAMS)


@pytest.mark.parametrize(
    "params",
    [
        {**_HOUGH_PARAMS, "threshold": 1.5},                  # > 1
        {**_HOUGH_PARAMS, "hough_input_threshold": 0.0},      # not > 0
        {**_HOUGH_PARAMS, "hough_input_threshold": 0.99,
         "threshold": 0.5},                                    # hough > threshold
    ],
)
def test_threshold_validation(params) -> None:
    prob, gt = _make_empty_canvas()
    with pytest.raises(ValueError, match="hough_input_threshold|threshold"):
        run_hough_on_canvas(prob, gt, **params)


# --- behaviour checks --------------------------------------------------------


def test_empty_canvas_has_zero_counts() -> None:
    prob, gt = _make_empty_canvas()
    result = run_hough_on_canvas(prob, gt, **_HOUGH_PARAMS)
    assert result.gt_pixels == 0
    assert result.pixels_pre == 0
    assert result.pixels_post == 0
    assert result.detected_pre is False
    assert result.detected_post is False


def test_false_positive_not_counted_as_detection() -> None:
    """Without GT overlap, even a bright Hough-detected line is not a detection."""
    prob, gt = _make_false_positive_only()
    result = run_hough_on_canvas(prob, gt, **_HOUGH_PARAMS)
    # Hough may fire on the bright diagonal in prob, but gt is empty.
    assert result.gt_pixels == 0
    assert result.pixels_pre == 0
    assert result.pixels_post == 0
    assert result.detected_pre is False
    assert result.detected_post is False


def test_post_hough_pixel_recall_at_least_pre_hough() -> None:
    """Pixel-level coverage is monotone non-decreasing pre → post by construction."""
    prob, gt = _make_diagonal_trail()
    result = run_hough_on_canvas(prob, gt, **_HOUGH_PARAMS)
    assert result.pixels_post >= result.pixels_pre


def test_canvases_have_uint8_dtype_and_correct_values() -> None:
    prob, gt = _make_diagonal_trail()
    result = run_hough_on_canvas(prob, gt, **_HOUGH_PARAMS)
    for canvas in (result.binary_canvas, result.hough_canvas, result.combined_canvas):
        assert canvas.dtype == np.uint8
        unique = set(np.unique(canvas).tolist())
        assert unique <= {0, 255}, f"canvas contained unexpected values: {unique}"
