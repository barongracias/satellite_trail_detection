import numpy as np
import pytest

from src.analysis.fp_decomposition import FpDecompositionResult, decompose_false_positives


def _patch(pred_ones: int, gt_ones: int, size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros(size, dtype=bool)
    pred[:pred_ones] = True
    gt = np.zeros(size, dtype=bool)
    gt[:gt_ones] = True
    return pred, gt


def test_all_fp_in_empty_patches_gives_inter_fraction_one() -> None:
    gt_empty = np.zeros(16, dtype=bool)
    pred_fp = np.ones(16, dtype=bool)
    samples = [(pred_fp, gt_empty, False)] * 4
    result = decompose_false_positives(samples)
    assert result.inter_patch_fp_fraction == pytest.approx(1.0)
    assert result.interpretation == "gate-favourable"


def test_all_fp_in_positive_patches_gives_inter_fraction_zero() -> None:
    # Positive patch: GT has trail in first 4 pixels; prediction fires on last 8 (FP region)
    gt_pos = np.zeros(16, dtype=bool)
    gt_pos[:4] = True
    pred_fp = np.zeros(16, dtype=bool)
    pred_fp[8:] = True
    samples = [(pred_fp, gt_pos, True)] * 4
    result = decompose_false_positives(samples)
    assert result.inter_patch_fp_fraction == pytest.approx(0.0)
    assert result.interpretation == "gate-limited"


def test_fifty_fifty_split_gives_correct_fractions() -> None:
    gt_empty = np.zeros(16, dtype=bool)
    pred_fp = np.ones(16, dtype=bool)
    gt_pos = np.ones(8, dtype=bool)
    gt_pos_full = np.zeros(16, dtype=bool)
    gt_pos_full[:8] = True
    pred_intra = np.ones(16, dtype=bool)

    empty_samples = [(pred_fp, gt_empty, False)] * 8
    pos_samples = [(pred_intra, gt_pos_full, True)] * 8
    result = decompose_false_positives(empty_samples + pos_samples)

    assert result.inter_patch_fp_pixels == 8 * 16
    assert result.intra_patch_fp_pixels == 8 * 8
    assert result.inter_patch_fp_fraction == pytest.approx(
        (8 * 16) / (8 * 16 + 8 * 8)
    )
    assert result.interpretation == "gate-favourable"


def test_zero_total_fp_gives_insufficient_data() -> None:
    gt = np.ones(16, dtype=bool)
    pred = np.ones(16, dtype=bool)
    samples = [(pred, gt, True)] * 4 + [(np.zeros(16, dtype=bool), np.zeros(16, dtype=bool), False)] * 4
    result = decompose_false_positives(samples)
    assert result.inter_patch_fp_fraction == pytest.approx(0.0)
    assert result.empty_patch_fp_rate == pytest.approx(0.0)
    assert result.interpretation == "insufficient-data"


def test_all_empty_patches_with_fp_gives_empty_fp_rate_one() -> None:
    gt_empty = np.zeros(16, dtype=bool)
    pred_fp = np.ones(16, dtype=bool)
    samples = [(pred_fp, gt_empty, False)] * 5
    result = decompose_false_positives(samples)
    assert result.empty_patch_fp_rate == pytest.approx(1.0)


def test_custom_gate_favourable_threshold_respected() -> None:
    gt_empty = np.zeros(16, dtype=bool)
    pred_fp = np.ones(16, dtype=bool)
    gt_pos = np.zeros(16, dtype=bool)
    pred_pos_fp = np.ones(16, dtype=bool)
    # 50% inter, 50% intra
    samples = [(pred_fp, gt_empty, False)] * 4 + [(pred_pos_fp, gt_pos, True)] * 4

    result_strict = decompose_false_positives(samples, gate_favourable_threshold=0.6)
    result_lenient = decompose_false_positives(samples, gate_favourable_threshold=0.4)

    assert result_strict.interpretation == "gate-limited"
    assert result_lenient.interpretation == "gate-favourable"
