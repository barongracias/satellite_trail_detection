"""Tests for the evaluation layer: EDA helpers, plots, and segmentation metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

from src.data.catalog import SatelliteCatalog
from src.evaluation.eda import (
    compute_image_summary_dataframe,
    compute_mask_component_stats,
    compute_observation_date_dataframe,
    compute_patch_dataframe,
    plot_example_satellite_trail_patches,
    plot_image_level_summary,
    plot_mask_inspection_grid,
    plot_mask_overlay,
    plot_mask_thickness_distribution,
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
from src.evaluation.segmentation import (
    combine_counts,
    compute_segmentation_counts,
    compute_segmentation_metrics,
)


matplotlib.use("Agg")


def _write_pair(
    root_dir: Path,
    stem: str,
    image_array: np.ndarray,
    mask_array: np.ndarray,
) -> None:
    Image.fromarray(image_array).save(root_dir / f"{stem}_red.fits_full.png")
    Image.fromarray(mask_array).save(root_dir / f"{stem}_red_mask.png")


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
