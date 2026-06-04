"""Tests for M6.1 locked-winner post-hoc analysis helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("cv2")
pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._locked_winner_canvases import (  # noqa: E402
    LOCKED_NORMALISATION,
    LOCKED_THRESHOLD,
    build_support_canvas,
    place_patch_max,
    provenance,
)
from scripts.faint_streak_analysis import (  # noqa: E402
    FP_RIDER_CATEGORIES,
    build_fp_intensity_rider,
    cleaned_gt_components,
    collect_fp_intensity_samples,
    component_pixel_metrics,
    fp_distance_strata,
    profile_metrics,
)
from scripts.hough_gap_figure import _local_gap_crop, rank_hough_candidates  # noqa: E402


def test_support_canvas_uses_raw_image_shape_and_patch_footprints() -> None:
    records = pd.DataFrame({
        "patch_path": [
            "data/patches/test/example_0_0_image.png",
            "data/patches/test/example_528_528_image.png",
        ]
    })
    support = build_support_canvas(records, shape=(1200, 1300))
    assert support.shape == (1200, 1300)
    assert support[0, 0]
    assert support[527, 527]
    assert support[528, 528]
    assert support[1055, 1055]
    assert not support[1100, 1100]
    assert int(support.sum()) == 2 * 528 * 528


def test_place_patch_max_clips_to_canvas_shape() -> None:
    canvas = np.zeros((4, 4), dtype=np.float32)
    patch = np.full((3, 3), 0.4, dtype=np.float32)
    place_patch_max(canvas, patch, 2, 2)
    assert np.all(canvas[2:4, 2:4] == 0.4)
    patch2 = np.full((2, 2), 0.2, dtype=np.float32)
    place_patch_max(canvas, patch2, 2, 2)
    assert np.all(canvas[2:4, 2:4] == 0.4)


def test_support_restriction_ignores_prediction_outside_support_for_crop_dice() -> None:
    gt = np.zeros((8, 8), dtype=bool)
    gt[4, 4] = True
    support = np.zeros_like(gt)
    support[4, 4] = True
    binary = np.zeros_like(gt)
    binary[4, 4] = True
    binary[0, 0] = True  # outside support: must not count as FP in crop Dice
    hough = np.zeros_like(gt)
    metrics = component_pixel_metrics(gt, binary, hough, support, crop_margin=8)
    assert metrics["gt_pixels_supported"] == 1
    assert metrics["recall_pre_hough"] == 1.0
    assert metrics["dice_crop_pre_hough"] == 1.0


def test_cleaned_gt_components_filters_small_noise() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[10, 3:25] = True
    mask[2, 2] = True  # tiny noisy component
    components, raw_count = cleaned_gt_components(
        mask, close_kernel=1, min_area=5, min_major_axis=10.0,
    )
    assert raw_count == 2
    assert len(components) == 1
    assert components[0]["area"] == 22
    assert components[0]["major_axis_length_px"] >= 20.0


def test_profile_metrics_recovers_clean_relative_contrast_and_fwhm() -> None:
    offsets = np.arange(-20, 21, dtype=float)
    sigma = 3.0
    background = 12.0
    amplitude = 30.0
    profile = background + amplitude * np.exp(-(offsets ** 2) / (2 * sigma ** 2))
    # Add asymmetric sideband scatter so the local contrast denominator is non-zero.
    profile = profile + np.where(offsets > 12, 1.0, 0.0)
    metrics = profile_metrics(offsets, profile, sideband_start=12.0)
    assert metrics["peak_minus_background"] > 25.0
    assert metrics["relative_display_contrast"] is not None
    assert metrics["relative_display_contrast"] > 10.0
    assert metrics["fwhm_px"] is not None
    assert 4.0 <= metrics["fwhm_px"] <= 10.0
    assert metrics["integrated_excess"] > 100.0


def test_hough_gap_ranking_uses_post_minus_pre_pixels() -> None:
    rows = [
        {"source_image": "a.png", "is_positive": True, "pixels_covered_pre": 10, "pixels_covered_post": 30},
        {"source_image": "b.png", "is_positive": True, "pixels_covered_pre": 100, "pixels_covered_post": 130},
        {"source_image": "c.png", "is_positive": False, "pixels_covered_pre": 0, "pixels_covered_post": 999},
    ]
    ranked = rank_hough_candidates(rows)
    assert [r["source_image"] for r in ranked] == ["b.png", "a.png"]
    assert ranked[0]["hough_recovered_gt_px"] == 30
    assert ranked[0]["rank_by_hough_recovered_gt_px"] == 1


def test_provenance_marks_locked_post_hoc_context() -> None:
    p = provenance(checkpoint="ckpt.pth", threshold=LOCKED_THRESHOLD, normalisation=LOCKED_NORMALISATION)
    assert p["checkpoint"] == "ckpt.pth"
    assert p["threshold"] == LOCKED_THRESHOLD
    assert p["normalisation"] == LOCKED_NORMALISATION
    assert p["split"] == "sampled_test"
    assert p["post_hoc_locked_winner_analysis"] is True
    assert "not calibrated" in p["display_intensity_caveat"]


def test_fp_distance_stratification_separates_near_ambiguous_and_far() -> None:
    target = np.zeros((12, 12), dtype=bool)
    support = np.ones_like(target, dtype=bool)
    binary = np.zeros_like(target, dtype=bool)
    target[5, 5] = True
    binary[5, 7] = True   # distance 2: near
    binary[5, 8] = True   # distance 3: ambiguous
    binary[5, 11] = True  # distance 6: far

    strata = fp_distance_strata(target, binary, support, near_px=2.0, far_px=5.0)

    assert bool(strata["near_gt_fp"][5, 7])
    assert bool(strata["ambiguous_fp"][5, 8])
    assert bool(strata["far_fp"][5, 11])
    assert int(strata["fp_mask"].sum()) == 3
    assert int(strata["near_gt_fp"].sum()) == 1
    assert int(strata["ambiguous_fp"].sum()) == 1
    assert int(strata["far_fp"].sum()) == 1


def test_fp_intensity_rider_effect_sizes_match_synthetic_categories() -> None:
    raw = np.full((12, 24), 10, dtype=np.uint8)
    target = np.zeros(raw.shape, dtype=bool)
    support = np.ones(raw.shape, dtype=bool)
    binary = np.zeros(raw.shape, dtype=bool)
    target[6, 2:8] = True
    raw[target] = 60
    binary[5, 2:8] = True   # near-GT FP, same display excess as trail
    raw[5, 2:8] = 60
    binary[0, 18:24] = True  # far FP, same display excess as background

    rng = np.random.default_rng(123)
    samples, record = collect_fp_intensity_samples(
        raw, target, binary, support, near_px=2.0, far_px=5.0,
        per_category_limit=1000, rng=rng,
    )
    pooled = {name: [samples[name]] for name in FP_RIDER_CATEGORIES}
    rider, final_samples = build_fp_intensity_rider(
        pooled, [record], near_px=2.0, far_px=5.0,
        subsample=1000, per_image_quota=1000, rng=np.random.default_rng(456),
    )

    assert rider["categories"]["near_gt_fp"]["n_sampled"] == 6
    assert rider["categories"]["far_fp"]["n_sampled"] == 6
    assert rider["categories"]["gt_trail"]["median"] == 50.0
    assert rider["categories"]["background"]["median"] == 0.0
    assert rider["ks_d_near_fp_vs_gt_trail"] <= 0.01
    assert rider["ks_d_far_fp_vs_background"] <= 0.01
    assert all(len(final_samples[name]) == rider["categories"][name]["n_sampled"] for name in FP_RIDER_CATEGORIES)
    json.dumps(rider)  # summary is JSON-safe and contains no sampled arrays


def test_hough_gap_crop_centres_largest_recovered_component() -> None:
    gap = np.zeros((120, 140), dtype=bool)
    fallback = np.zeros_like(gap)
    gap[10:14, 10:14] = True
    gap[72:82, 64:80] = True

    sl_y, sl_x, local = _local_gap_crop(gap, fallback, gap.shape, half_size=20)

    global_y = sl_y.start + local[0]
    global_x = sl_x.start + local[1]
    assert 74 <= global_y <= 79
    assert 69 <= global_x <= 75
    assert sl_y.start <= 72 < sl_y.stop
    assert sl_x.start <= 64 < sl_x.stop
