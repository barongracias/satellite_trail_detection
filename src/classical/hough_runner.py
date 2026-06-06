"""Per-image Hough post-processing helper used by evaluation scripts.

The script-level logic in ``scripts/evaluation/hough_postprocess.py`` is responsible for
manifest grouping, per-source-image canvas construction, batched U-Net
inference, and cross-image aggregation. The per-canvas operations — applying
the main threshold, running the probabilistic Hough transform on a lower-
threshold canvas, and computing image-level + pixel-level overlap counts —
are independent of the manifest grouping and the model, so they live here
instead of being duplicated across scripts.

The helper returns both the scalar counts and the boolean canvases. The
canvases are returned so the caller can carry out **patch-level** FNR
aggregation (which requires per-patch ``(y, x)`` slicing of
``binary_canvas`` / ``combined_canvas``) without the helper needing to know
about the manifest's patch extents.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _apply_hough(
    canvas: np.ndarray,
    *,
    hough_threshold: int,
    min_line_length: int,
    max_line_gap: int,
    line_thickness: int,
) -> np.ndarray:
    """Draw HoughLinesP detections onto a fresh ``uint8`` canvas at ``255``."""
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


@dataclass(frozen=True)
class HoughCanvasResult:
    """Per-image result of pre/post-Hough evaluation against a target canvas.

    Attributes are bitwise-equivalent to the in-place computation performed
    by ``scripts/evaluation/hough_postprocess.py`` per source image. The boolean
    canvases are exposed so the caller can compute patch-level FNR by
    slicing ``binary_canvas`` / ``combined_canvas`` at each patch's
    ``(y, x)`` extent.
    """

    binary_canvas: np.ndarray   # uint8, {0, 255}; threshold >= main threshold
    hough_canvas: np.ndarray    # uint8, {0, 255}; HoughLinesP output
    combined_canvas: np.ndarray # uint8, {0, 255}; binary | hough
    detected_pre: bool          # image-level: thresholded prediction overlaps GT
    detected_post: bool         # image-level: (binary | hough) overlaps GT
    gt_pixels: int              # |GT > 0|
    pixels_pre: int             # |binary & GT|
    pixels_post: int            # |(binary | hough) & GT|


def run_hough_on_canvas(
    prob_canvas: np.ndarray,
    gt_canvas: np.ndarray,
    *,
    threshold: float,
    hough_input_threshold: float,
    hough_threshold: int,
    min_line_length: int,
    max_line_gap: int,
    line_thickness: int,
) -> HoughCanvasResult:
    """Apply main threshold + probabilistic Hough recovery to one source image.

    Parameters
    ----------
    prob_canvas:
        ``(H, W)`` float probability map for a single source image. Values
        outside [0, 1] are tolerated (the comparisons against the thresholds
        do the right thing), but normally inputs are in [0, 1].
    gt_canvas:
        ``(H, W)`` uint8 ground-truth mask, ``> 0`` indicating trail pixels.
        Must have the same shape as ``prob_canvas``.
    threshold:
        Main binary threshold applied to ``prob_canvas`` for pixel-level
        coverage and image-level pre-Hough detection. In [0, 1] by
        convention.
    hough_input_threshold:
        Lower threshold used to construct the binary canvas fed into
        ``cv2.HoughLinesP``. Must satisfy
        ``0 < hough_input_threshold <= threshold <= 1`` so that Hough can
        recover faint detections below the main threshold without trivially
        widening the binary canvas.
    hough_threshold, min_line_length, max_line_gap, line_thickness:
        Forwarded to ``cv2.HoughLinesP`` / ``cv2.line``.

    Returns
    -------
    HoughCanvasResult
        Image-level detection booleans, pixel-level coverage counts, and
        the boolean canvases needed for patch-level aggregation in the
        caller.

    Notes
    -----
    The image-level detection flags require **overlap with ground truth**:
    a stray false-positive that does not touch a GT pixel does not count
    as detection. This matches the contract in
    ``scripts/evaluation/hough_postprocess.py``, where dropping the overlap
    requirement would trivially drive FNR to zero on any image with even
    one false-positive pixel.
    """
    if prob_canvas.shape != gt_canvas.shape:
        raise ValueError(
            f"prob_canvas.shape={prob_canvas.shape} != gt_canvas.shape={gt_canvas.shape}"
        )
    if not (0.0 < hough_input_threshold <= threshold <= 1.0):
        raise ValueError(
            f"Require 0 < hough_input_threshold ({hough_input_threshold}) "
            f"<= threshold ({threshold}) <= 1"
        )

    binary_canvas = (prob_canvas >= threshold).astype(np.uint8) * 255
    detected_pre = bool(
        (binary_canvas > 0).any() and (binary_canvas & gt_canvas).any()
    )

    hough_input = (prob_canvas >= hough_input_threshold).astype(np.uint8) * 255
    hough_canvas = _apply_hough(
        hough_input,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
        line_thickness=line_thickness,
    )
    combined_canvas = binary_canvas | hough_canvas
    detected_post = bool((combined_canvas & gt_canvas).any())

    gt_bool = gt_canvas > 0
    gt_pixels = int(gt_bool.sum())
    pixels_pre = int(((binary_canvas > 0) & gt_bool).sum())
    pixels_post = int(((combined_canvas > 0) & gt_bool).sum())

    return HoughCanvasResult(
        binary_canvas=binary_canvas,
        hough_canvas=hough_canvas,
        combined_canvas=combined_canvas,
        detected_pre=detected_pre,
        detected_post=detected_post,
        gt_pixels=gt_pixels,
        pixels_pre=pixels_pre,
        pixels_post=pixels_post,
    )
