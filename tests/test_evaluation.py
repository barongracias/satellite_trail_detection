"""Tests for the evaluation layer: EDA helpers, plots, and segmentation metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from src.data.catalog import SatelliteCatalog
from tests.helpers import write_pair as _write_pair
from src.evaluation.eda import (  # noqa: E402
    compute_image_summary_dataframe,
    compute_mask_component_stats,
    compute_observation_date_dataframe,
    compute_patch_dataframe,
    plot_example_satellite_trail_patches,
    plot_image_level_summary,
    plot_mask_inspection_grid,
    plot_mask_overlay,
    plot_metadata_missing_values,
    plot_observation_date_distribution,
    plot_patch_density_distribution,
    plot_patch_density_heatmap,
    plot_ra_dec_distributions,
    plot_ra_dec_ranges,
    plot_random_image_mask_pairs,
    plot_random_mask_overlays,
    plot_satellite_name_frequency,
    plot_trail_length_distribution,
    summarise_patch_dataframe,
    summarise_patches_by_image,
)

matplotlib.use("Agg")

torch = pytest.importorskip("torch")

from src.evaluation.segmentation import (  # noqa: E402
    SegmentationCounts,
    boundary_tolerant_counts,
    boundary_tolerant_metrics,
    bootstrap_metrics_cluster,
    bootstrap_metrics_patch,
    centerline_dice,
    combine_counts,
    compute_metrics_from_counts,
    compute_segmentation_counts,
    compute_segmentation_metrics,
    false_positive_distances,
)


def test_boundary_tolerant_tolerance_zero_matches_exact() -> None:
    pred = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=np.uint8)
    gt = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.uint8)
    bm = boundary_tolerant_metrics(boundary_tolerant_counts(pred, gt, tolerance=0))
    em = compute_metrics_from_counts(compute_segmentation_counts(pred, gt))
    assert bm["precision"] == pytest.approx(em.precision)
    assert bm["recall"] == pytest.approx(em.recall)


def test_boundary_tolerant_recovers_one_pixel_offset_line() -> None:
    # A prediction shifted 1px from the GT line scores 0 exactly but ~1.0 at
    # tolerance=1 — the core PS-0 effect (thin labels, ±1px disagreement).
    gt = np.zeros((9, 9), dtype=np.uint8)
    gt[:, 4] = 1
    pred = np.zeros((9, 9), dtype=np.uint8)
    pred[:, 5] = 1
    exact = boundary_tolerant_metrics(boundary_tolerant_counts(pred, gt, tolerance=0))
    tol1 = boundary_tolerant_metrics(boundary_tolerant_counts(pred, gt, tolerance=1))
    assert exact["precision"] == pytest.approx(0.0)
    assert exact["recall"] == pytest.approx(0.0)
    assert tol1["precision"] == pytest.approx(1.0)
    assert tol1["recall"] == pytest.approx(1.0)


def test_boundary_tolerant_recovers_diagonal_one_pixel_offset() -> None:
    # A 1px DIAGONAL offset must be tolerated at tolerance=1 — this only holds for
    # the 8-neighbour box (MORPH_RECT); a 4-neighbour cross would leave it as FP/FN.
    gt = np.eye(9, dtype=np.uint8)                       # main diagonal
    pred = np.zeros((9, 9), dtype=np.uint8)
    for i in range(8):
        pred[i + 1, i] = 1                               # diagonal shifted 1px down
    exact = boundary_tolerant_metrics(boundary_tolerant_counts(pred, gt, tolerance=0))
    tol1 = boundary_tolerant_metrics(boundary_tolerant_counts(pred, gt, tolerance=1))
    assert exact["precision"] < 0.2 and exact["recall"] < 0.2
    assert tol1["precision"] == pytest.approx(1.0)
    assert tol1["recall"] == pytest.approx(1.0)


def test_boundary_tolerant_precision_non_decreasing_in_tolerance() -> None:
    rng = np.random.default_rng(0)
    gt = (rng.random((16, 16)) > 0.8).astype(np.uint8)
    pred = (rng.random((16, 16)) > 0.8).astype(np.uint8)
    precs = [boundary_tolerant_metrics(boundary_tolerant_counts(pred, gt, t))["precision"]
             for t in (0, 1, 2, 3)]
    assert all(b >= a - 1e-9 for a, b in zip(precs, precs[1:]))


def test_boundary_tolerant_counts_add_and_shape_guard() -> None:
    a = boundary_tolerant_counts(np.eye(4, dtype=np.uint8), np.eye(4, dtype=np.uint8), 0)
    combined = a + a
    assert combined.tp_precision == 2 * a.tp_precision
    with pytest.raises(ValueError):
        boundary_tolerant_counts(np.zeros((2, 2)), np.zeros((3, 3)), 1)


def test_centerline_dice_identical_and_disjoint() -> None:
    line = np.zeros((11, 11), dtype=np.uint8)
    line[:, 5] = 1
    assert centerline_dice(line, line) == pytest.approx(1.0)
    # both empty -> perfect agreement on absence
    empty = np.zeros((11, 11), dtype=np.uint8)
    assert centerline_dice(empty, empty) == pytest.approx(1.0)
    # one empty -> 0
    assert centerline_dice(line, empty) == pytest.approx(0.0)
    # disjoint thick blocks: neither skeleton overlaps the other mask
    pred = np.zeros((11, 11), dtype=np.uint8); pred[0:3, 0:3] = 1
    gt = np.zeros((11, 11), dtype=np.uint8); gt[8:11, 8:11] = 1
    assert centerline_dice(pred, gt) == pytest.approx(0.0)


def test_centerline_dice_shape_guard() -> None:
    with pytest.raises(ValueError):
        centerline_dice(np.zeros((4, 4)), np.zeros((5, 5)))


def test_false_positive_distances_adjacent_and_diagonal() -> None:
    gt = np.zeros((5, 5), dtype=np.uint8)
    gt[2, 2] = 1
    pred = np.zeros((5, 5), dtype=np.uint8)
    pred[2, 2] = 1   # TP (not an FP)
    pred[2, 3] = 1   # FP 1px to the right -> distance 1.0
    pred[0, 0] = 1   # FP at corner -> distance sqrt(8)
    dists, whole = false_positive_distances(pred, gt)
    assert whole == 0
    assert sorted(round(float(d), 4) for d in dists) == [1.0, pytest.approx(np.sqrt(8), abs=1e-4)]


def test_false_positive_distances_empty_gt_reported_separately() -> None:
    gt = np.zeros((5, 5), dtype=np.uint8)
    pred = np.zeros((5, 5), dtype=np.uint8)
    pred[1, 1] = 1
    pred[3, 3] = 1
    dists, whole = false_positive_distances(pred, gt)
    assert dists.size == 0       # distance undefined when GT is empty
    assert whole == 2            # counted as whole-patch FP instead


def _write_meerlicht_pair(
    root_dir: Path,
    timestamp: str,
    image_array: np.ndarray,
    mask_array: np.ndarray,
) -> None:
    """Write a pair whose filename matches the MeerLICHT date convention."""
    _write_pair(root_dir, f"ML1_{timestamp}", image_array, mask_array)


def test_patch_statistics_capture_empty_and_non_empty_patches(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    _write_pair(
        processed_dir,
        "A",
        np.array(
            [
                [10, 10, 10, 10],
                [10, 20, 20, 10],
                [10, 20, 20, 10],
                [10, 10, 10, 10],
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [255, 255, 0, 0],
                [255, 255, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    _write_pair(
        processed_dir,
        "B",
        np.full((4, 4), 50, dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint8),
    )

    patch_df = compute_patch_dataframe(
        root_dir=processed_dir,
        patch_size=2,
        stride=2,
    )
    summary = summarise_patch_dataframe(patch_df)

    assert summary.total_images == 2
    assert summary.total_patches == 8
    assert summary.empty_patches == 7
    assert summary.non_empty_patches == 1
    assert summary.positive_pixel_fraction == 0.125


def test_eda_plot_helpers_save_meeting_figures(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    image_array = np.array(
        [
            [10, 20, 30, 40],
            [20, 30, 40, 50],
            [30, 40, 50, 60],
            [40, 50, 60, 70],
        ],
        dtype=np.uint8,
    )
    mask_array = np.array(
        [
            [255, 255, 0, 0],
            [255, 255, 0, 0],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )
    _write_pair(processed_dir, "A", image_array, mask_array)

    patch_df = compute_patch_dataframe(
        root_dir=processed_dir,
        patch_size=2,
        stride=2,
    )
    image_df = compute_image_summary_dataframe(processed_dir)
    patch_summary = summarise_patches_by_image(patch_df)

    paths = [
        plot_random_image_mask_pairs(processed_dir, sample_count=1, output_dir=tmp_path, show=False),
        plot_mask_overlay(
            image_path=processed_dir / "A_red.fits_full.png",
            mask_path=processed_dir / "A_red_mask.png",
            output_name="single_overlay.png",
            output_dir=tmp_path,
            show=False,
        ),
        plot_random_mask_overlays(processed_dir, sample_count=1, output_dir=tmp_path, show=False),
        plot_patch_density_distribution(patch_df, output_dir=tmp_path, show=False),
        plot_patch_density_heatmap(patch_df, image_name="A_red.fits_full.png", output_dir=tmp_path, show=False),
        plot_example_satellite_trail_patches(patch_df, sample_count=1, output_dir=tmp_path, show=False),
        plot_image_level_summary(image_df=image_df, patch_df=patch_df, output_dir=tmp_path, show=False),
        plot_mask_inspection_grid(patch_df=patch_df, n_examples=1, output_dir=tmp_path, show=False),
    ]

    assert image_df.loc[0, "mask_positive_fraction"] == 0.5
    assert patch_summary.loc[0, "non_empty_patches"] == 2
    for path in paths:
        assert path.exists()


def test_mask_component_stats_capture_simple_components(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    mask_array = np.zeros((6, 6), dtype=np.uint8)
    mask_array[1:5, 2:4] = 255
    _write_pair(processed_dir, "A", np.full((6, 6), 30, dtype=np.uint8), mask_array)

    component_df = compute_mask_component_stats(processed_dir, min_area=1)
    assert len(component_df) == 1
    row = component_df.iloc[0]
    assert row["area"] == 8
    assert row["major_axis"] > row["minor_axis"]


def test_observation_date_dataframe_parses_meerlicht_filenames(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    image_array = np.zeros((4, 4), dtype=np.uint8)
    mask_array = np.zeros((4, 4), dtype=np.uint8)
    _write_meerlicht_pair(processed_dir, "20190602_181454", image_array, mask_array)
    _write_meerlicht_pair(processed_dir, "20171023_002210", image_array, mask_array)

    date_df = compute_observation_date_dataframe(processed_dir)
    assert set(date_df["observation_year"].tolist()) == {2017, 2019}

    output_path = plot_observation_date_distribution(date_df, output_dir=tmp_path, show=False)
    assert output_path.exists()


def test_catalog_plots_are_saved(tmp_path: Path) -> None:
    csv_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "Length": [10.0, 20.0, 30.0],
            "Start_RA": [1.0, 2.0, 3.0],
            "End_RA": [1.5, 2.5, 3.5],
            "Start_DEC": [-1.0, 0.0, 1.0],
            "End_DEC": [-0.5, 0.5, 1.5],
            "Satellite_Name": ["A", "B", "A"],
        }
    ).to_csv(csv_path, index=False)

    catalog = SatelliteCatalog(csv_path)

    paths = [
        plot_satellite_name_frequency(catalog, output_dir=tmp_path, show=False),
        plot_trail_length_distribution(catalog, output_dir=tmp_path, show=False),
        plot_ra_dec_ranges(catalog, output_dir=tmp_path, show=False),
        plot_metadata_missing_values(catalog, output_dir=tmp_path, show=False),
        plot_ra_dec_distributions(catalog, output_dir=tmp_path, show=False),
    ]
    for path in paths:
        assert path.exists()


def test_segmentation_metrics_work_for_binary_masks_and_logits() -> None:
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    target = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    counts = compute_segmentation_counts(prediction, target)
    metrics = compute_segmentation_metrics(prediction, target)

    assert counts.true_positive == 1
    assert counts.false_positive == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.dice == 2.0 / 3.0
    assert metrics.iou == 0.5

    logits = np.array([[8.0, -8.0], [8.0, -8.0]], dtype=np.float32)
    counts = compute_segmentation_counts(logits, target, from_logits=True)
    combined = combine_counts([counts, counts])

    assert combined.true_positive == 2


def test_segmentation_counts_returns_python_ints() -> None:
    """Locks the public API guarantee: SegmentationCounts fields are Python ints,
    not torch.Tensor / np.integer / numpy scalars. Required for asdict()/json.dump
    serialisation into checkpoint payloads and summary JSONs."""
    import torch as _torch
    prediction = _torch.tensor([[1, 0], [1, 0]], dtype=_torch.float32)
    target = _torch.tensor([[1, 0], [0, 0]], dtype=_torch.float32)
    counts = compute_segmentation_counts(prediction, target)
    assert type(counts.true_positive) is int
    assert type(counts.false_positive) is int
    assert type(counts.true_negative) is int
    assert type(counts.false_negative) is int


def test_evaluate_accumulates_match_direct_counts(tmp_path: Path) -> None:
    """The streaming eval accumulator must yield identical counts to the
    one-shot compute_segmentation_counts on the concatenation of all batches."""
    import torch as _torch
    from src.evaluation.segmentation import evaluate_model_on_dataloader

    _torch.manual_seed(0)
    # 3 batches × 2 samples × 1 channel × 4×4
    images_batches = [_torch.randn(2, 1, 4, 4) for _ in range(3)]
    masks_batches = [(_torch.rand(2, 1, 4, 4) > 0.7).float() for _ in range(3)]
    loader = [
        {"image": img, "mask": msk}
        for img, msk in zip(images_batches, masks_batches)
    ]

    class _Identity(_torch.nn.Module):
        def forward(self, x):
            return x   # logits == input; sigmoid path exercised via threshold

    model = _Identity()
    device = _torch.device("cpu")
    result = evaluate_model_on_dataloader(model, loader, device, threshold=0.5)

    # Direct counts on concatenated batches:
    all_logits = _torch.cat(images_batches, dim=0)
    all_masks = _torch.cat(masks_batches, dim=0)
    direct = compute_segmentation_counts(
        all_logits, all_masks, threshold=0.5, from_logits=True
    )
    assert result.counts.true_positive == direct.true_positive
    assert result.counts.false_positive == direct.false_positive
    assert result.counts.true_negative == direct.true_negative
    assert result.counts.false_negative == direct.false_negative


def test_segmentation_counts_rejects_shape_mismatch() -> None:
    """Strict shape check must fire before flattening so same-numel-different-shape
    mismatches (e.g. (1,4) vs (2,2)) still raise."""
    import torch as _torch
    # Case (a): different number of elements.
    a = _torch.zeros((1, 4))
    b = _torch.zeros((1, 5))
    try:
        compute_segmentation_counts(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError on different-numel shapes")

    # Case (b): same numel, different shape.
    a = _torch.zeros((1, 4))
    b = _torch.zeros((2, 2))
    try:
        compute_segmentation_counts(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError on same-numel different-shape")


def test_segmentation_counts_handles_mixed_numpy_torch_input() -> None:
    """Device-alignment: prediction may be numpy, target torch, or vice versa;
    the public function must align both to the prediction device without raising."""
    import torch as _torch
    pred_np = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    target_torch = _torch.tensor([[1, 0, 0, 0]], dtype=_torch.float32)
    counts_a = compute_segmentation_counts(pred_np, target_torch)

    pred_torch = _torch.tensor([[1, 0, 1, 0]], dtype=_torch.float32)
    target_np = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    counts_b = compute_segmentation_counts(pred_torch, target_np)

    assert counts_a == counts_b
    assert counts_a.true_positive == 1
    assert counts_a.false_positive == 1


def test_bootstrap_metrics_cluster_point_matches_aggregate_and_is_reproducible() -> None:
    """Point estimate equals compute_metrics_from_counts on the aggregated
    counts; fixed seed → identical resample draws → identical CIs."""
    per_image = {
        "img_a": SegmentationCounts(true_positive=10, false_positive=2,
                                     true_negative=88, false_negative=1),
        "img_b": SegmentationCounts(true_positive=5, false_positive=1,
                                     true_negative=44, false_negative=0),
        "img_c": SegmentationCounts(true_positive=0, false_positive=0,
                                     true_negative=50, false_negative=0),
    }
    point = compute_metrics_from_counts(combine_counts(per_image.values()))
    result = bootstrap_metrics_cluster(per_image, n_resamples=200, seed=2804)

    assert set(result.keys()) == {"precision", "recall", "dice", "iou"}
    assert abs(result["precision"][0] - point.precision) < 1e-9
    assert abs(result["recall"][0] - point.recall) < 1e-9
    assert abs(result["dice"][0] - point.dice) < 1e-9
    assert abs(result["iou"][0] - point.iou) < 1e-9
    for name in ("precision", "recall", "dice", "iou"):
        _, lo, hi = result[name]
        assert lo <= hi, f"{name}: lo {lo} > hi {hi}"

    result_again = bootstrap_metrics_cluster(per_image, n_resamples=200, seed=2804)
    assert result == result_again


def test_bootstrap_metrics_cluster_is_insertion_order_independent() -> None:
    """Dict insertion order must not change the seeded bootstrap output —
    callers building the per-image mapping via glob/walk vs. a sorted iterator
    should land identical CIs from the same underlying counts."""
    counts_a = SegmentationCounts(true_positive=10, false_positive=2,
                                   true_negative=88, false_negative=1)
    counts_b = SegmentationCounts(true_positive=5, false_positive=1,
                                   true_negative=44, false_negative=0)
    counts_c = SegmentationCounts(true_positive=0, false_positive=0,
                                   true_negative=50, false_negative=0)

    forward = {"img_a": counts_a, "img_b": counts_b, "img_c": counts_c}
    reverse = {"img_c": counts_c, "img_b": counts_b, "img_a": counts_a}

    result_forward = bootstrap_metrics_cluster(forward, n_resamples=200, seed=2804)
    result_reverse = bootstrap_metrics_cluster(reverse, n_resamples=200, seed=2804)
    assert result_forward == result_reverse


def test_bootstrap_metrics_patch_collapses_with_single_unit() -> None:
    """With one patch, every resample is that patch → CI lo == hi == point."""
    counts = SegmentationCounts(true_positive=4, false_positive=1,
                                 true_negative=10, false_negative=1)
    result = bootstrap_metrics_patch([counts], n_resamples=50, seed=1)
    for name in ("precision", "recall", "dice", "iou"):
        point, lo, hi = result[name]
        assert abs(lo - point) < 1e-12, f"{name}: lo {lo} != point {point}"
        assert abs(hi - point) < 1e-12, f"{name}: hi {hi} != point {point}"


def test_surface_nsd_identical_masks_perfect_at_tau_zero() -> None:
    """tau=0 degenerate behaviour: identical masks give NSD=1; any boundary
    offset breaks tau=0 but not tau>=offset."""
    from src.evaluation.segmentation import surface_distance_counts

    mask = np.zeros((32, 32), dtype=bool)
    mask[10, 5:25] = True
    c = surface_distance_counts(mask, mask, tau=0.0)
    assert c.nsd == 1.0
    assert c.pred_boundary_total == c.gt_boundary_total == 20


def test_surface_nsd_one_px_offset_line_is_one_at_tau_one() -> None:
    """A synthetic 1 px-offset line scores NSD=1 at tau=1 but < 1 at tau=0 —
    the boundary-scale equivalence the metric is meant to capture."""
    from src.evaluation.segmentation import surface_distance_counts_multi

    gt = np.zeros((32, 64), dtype=bool)
    pred = np.zeros((32, 64), dtype=bool)
    gt[10, 5:60] = True
    pred[11, 5:60] = True   # same line, shifted down by exactly 1 px
    counts = surface_distance_counts_multi(pred, gt, taus=(0.0, 1.0, 2.0))
    assert counts[0.0].nsd == 0.0
    assert counts[1.0].nsd == 1.0
    assert counts[2.0].nsd == 1.0


def test_surface_nsd_empty_mask_edges() -> None:
    """Empty/empty has undefined NSD (None — caller excludes); one-sided empty
    scores 0; tau must be non-negative."""
    import pytest as _pytest

    from src.evaluation.segmentation import (
        surface_distance_counts,
        surface_distance_counts_multi,
    )

    empty = np.zeros((16, 16), dtype=bool)
    line = np.zeros((16, 16), dtype=bool)
    line[8, 2:14] = True

    both_empty = surface_distance_counts(empty, empty, tau=1.0)
    assert both_empty.nsd is None
    assert both_empty.pred_boundary_total == both_empty.gt_boundary_total == 0

    pred_only = surface_distance_counts(line, empty, tau=1.0)   # whole-image FP
    assert pred_only.nsd == 0.0
    assert pred_only.pred_boundary_total > 0

    gt_only = surface_distance_counts(empty, line, tau=1.0)     # whole-image FN
    assert gt_only.nsd == 0.0

    with _pytest.raises(ValueError):
        surface_distance_counts_multi(line, line, taus=(-1.0,))


def test_surface_nsd_counts_micro_aggregate_by_addition() -> None:
    """SurfaceDistanceCounts adds field-wise so canvas-level micro aggregation
    equals counting over the union of boundaries."""
    from src.evaluation.segmentation import (
        surface_distance_counts,
    )

    gt_a = np.zeros((16, 16), dtype=bool); gt_a[4, 2:10] = True
    pred_a = np.zeros((16, 16), dtype=bool); pred_a[5, 2:10] = True
    gt_b = np.zeros((16, 16), dtype=bool); gt_b[8, 1:15] = True
    pred_b = np.zeros((16, 16), dtype=bool)   # whole-image FN

    c_a = surface_distance_counts(pred_a, gt_a, tau=1.0)
    c_b = surface_distance_counts(pred_b, gt_b, tau=1.0)
    total = c_a + c_b
    assert total.pred_boundary_total == c_a.pred_boundary_total
    assert total.gt_boundary_total == c_a.gt_boundary_total + c_b.gt_boundary_total
    expected = (c_a.pred_boundary_within_tau + c_a.gt_boundary_within_tau) / (
        c_a.pred_boundary_total + c_a.gt_boundary_total
        + c_b.pred_boundary_total + c_b.gt_boundary_total
    )
    assert total.nsd == _pytest_approx(expected)


def _pytest_approx(x: float):
    import pytest as _pytest

    return _pytest.approx(x, abs=1e-12)


def test_surface_nsd_thick_mask_boundary_is_eroded_interior_excluded() -> None:
    """On a wide structure the interior must not count as boundary: a filled
    8x8 square has a 28-px one-pixel-wide boundary ring."""
    from src.evaluation.segmentation import surface_distance_counts

    sq = np.zeros((16, 16), dtype=bool)
    sq[4:12, 4:12] = True
    c = surface_distance_counts(sq, sq, tau=0.0)
    assert c.gt_boundary_total == 28
    assert c.nsd == 1.0
