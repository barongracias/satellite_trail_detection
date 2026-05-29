"""False-positive decomposition: inter-patch (empty) vs intra-patch (positive) FP pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FpDecompositionResult:
    inter_patch_fp_pixels: int
    intra_patch_fp_pixels: int
    inter_patch_fp_fraction: float
    empty_patch_fp_rate: float
    interpretation: str


def decompose_false_positives(
    samples: list[tuple[np.ndarray, np.ndarray, bool]],
    gate_favourable_threshold: float = 0.5,
) -> FpDecompositionResult:
    """Decompose FP pixels into recoverable (inter-patch) and unrecoverable (intra-patch).

    Parameters
    ----------
    samples:
        List of ``(pred_binary, gt, is_positive)`` tuples.  ``pred_binary`` and
        ``gt`` are boolean or 0/1 arrays of the same shape.  ``is_positive``
        is ``True`` when the patch contains at least one ground-truth trail pixel.
    gate_favourable_threshold:
        Minimum ``inter_patch_fp_fraction`` required for a "gate-favourable"
        interpretation.
    """
    inter_fp = 0
    intra_fp = 0
    n_empty = 0
    n_empty_with_fp = 0

    for pred, gt, is_positive in samples:
        pred_arr = np.asarray(pred, dtype=bool)
        gt_arr = np.asarray(gt, dtype=bool)
        fp_pixels = int(np.sum(pred_arr & ~gt_arr))

        if is_positive:
            intra_fp += fp_pixels
        else:
            inter_fp += fp_pixels
            n_empty += 1
            if fp_pixels > 0:
                n_empty_with_fp += 1

    total_fp = inter_fp + intra_fp

    if total_fp == 0:
        inter_fraction = 0.0
    else:
        inter_fraction = inter_fp / total_fp

    if n_empty == 0:
        empty_fp_rate = 0.0
    else:
        empty_fp_rate = n_empty_with_fp / n_empty

    if total_fp == 0:
        interpretation = "insufficient-data"
    elif inter_fraction >= gate_favourable_threshold:
        interpretation = "gate-favourable"
    else:
        interpretation = "gate-limited"

    return FpDecompositionResult(
        inter_patch_fp_pixels=inter_fp,
        intra_patch_fp_pixels=intra_fp,
        inter_patch_fp_fraction=inter_fraction,
        empty_patch_fp_rate=empty_fp_rate,
        interpretation=interpretation,
    )
