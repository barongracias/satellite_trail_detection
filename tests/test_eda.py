from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from src.evaluation.eda import (
    compute_image_summary_dataframe,
    compute_patch_dataframe,
    plot_example_satellite_trail_patches,
    plot_image_level_summary,
    plot_mask_overlay,
    plot_patch_density_distribution,
    plot_patch_density_heatmap,
    plot_random_image_mask_pairs,
    plot_random_mask_overlays,
    summarise_patches_by_image,
    summarise_patch_dataframe,
)


matplotlib.use("Agg")


def _write_test_pair(
    root_dir: Path,
    stem: str,
    image_array: np.ndarray,
    mask_array: np.ndarray,
) -> None:
    Image.fromarray(image_array).save(root_dir / f"{stem}_red.fits_full.png")
    Image.fromarray(mask_array).save(root_dir / f"{stem}_red_mask.png")


def test_patch_statistics_capture_empty_and_non_empty_patches(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    _write_test_pair(
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
    _write_test_pair(
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
    _write_test_pair(processed_dir, "A", image_array, mask_array)

    patch_df = compute_patch_dataframe(
        root_dir=processed_dir,
        patch_size=2,
        stride=2,
    )
    image_df = compute_image_summary_dataframe(processed_dir)
    patch_summary = summarise_patches_by_image(patch_df)

    pair_path = plot_random_image_mask_pairs(
        processed_dir,
        sample_count=1,
        output_dir=tmp_path,
        show=False,
    )
    overlay_path = plot_mask_overlay(
        image_path=processed_dir / "A_red.fits_full.png",
        mask_path=processed_dir / "A_red_mask.png",
        output_name="single_overlay.png",
        output_dir=tmp_path,
        show=False,
    )
    overlays_path = plot_random_mask_overlays(
        processed_dir,
        sample_count=1,
        output_dir=tmp_path,
        show=False,
    )
    density_path = plot_patch_density_distribution(
        patch_df,
        output_dir=tmp_path,
        show=False,
    )
    heatmap_path = plot_patch_density_heatmap(
        patch_df,
        image_name="A_red.fits_full.png",
        output_dir=tmp_path,
        show=False,
    )
    trail_path = plot_example_satellite_trail_patches(
        patch_df,
        sample_count=2,
        output_dir=tmp_path,
        show=False,
    )
    image_summary_path = plot_image_level_summary(
        image_df=image_df,
        patch_df=patch_df,
        output_dir=tmp_path,
        show=False,
    )

    assert image_df.loc[0, "mask_positive_fraction"] == 0.5
    assert patch_summary.loc[0, "non_empty_patches"] == 2
    assert pair_path.exists()
    assert overlay_path.exists()
    assert overlays_path.exists()
    assert density_path.exists()
    assert heatmap_path.exists()
    assert trail_path.exists()
    assert image_summary_path.exists()
