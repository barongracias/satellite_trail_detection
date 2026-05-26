"""Command-line helper for quick inspection of the processed patch dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.constants import PATCH_SIZE
from src.data.indexing import PatchDatasetConfig
from src.evaluation.eda import compute_patch_dataframe, summarise_patch_dataframe
from src.utils.logger import get_logger


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for dataset inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/Processed"),
        help="Directory containing processed image and mask PNG pairs.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help="Square patch size in pixels.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=PATCH_SIZE,
        help="Patch stride in pixels.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-density positive patches to print.",
    )
    return parser


def main() -> None:
    """Compute and print dataset-level patch statistics."""
    args = build_parser().parse_args()
    logger = get_logger("inspect_dataset")
    dataset_config = PatchDatasetConfig(
        root_dir=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    patch_df = compute_patch_dataframe(
        root_dir=dataset_config,
        logger=logger,
    )
    summary = summarise_patch_dataframe(patch_df)
    empty_patch_fraction = summary.empty_patches / summary.total_patches
    non_empty_patch_fraction = summary.non_empty_patches / summary.total_patches

    top_patches = (
        patch_df.loc[~patch_df["is_empty"]]
        .sort_values("positive_pixel_fraction", ascending=False)
        .head(args.top_k)
        .loc[:, ["image_name", "y", "x", "positive_pixel_fraction"]]
    )

    print("\nDataset summary")
    print(
        f"root={dataset_config.root_dir} | patch_size={dataset_config.patch_size} "
        f"| stride={dataset_config.resolved_stride} | image_pairs={summary.total_images}"
    )
    print(f"total_patches={summary.total_patches}")
    print(f"empty_patches={summary.empty_patches}")
    print(f"non_empty_patches={summary.non_empty_patches}")
    print(f"empty_patch_fraction={empty_patch_fraction:.4f}")
    print(f"non_empty_patch_fraction={non_empty_patch_fraction:.4f}")
    print(f"positive_pixel_fraction={summary.positive_pixel_fraction:.8f}")
    print("\nHighest-density non-empty patches")
    if top_patches.empty:
        print("No non-empty patches found.")
    else:
        print(top_patches.to_string(index=False))


if __name__ == "__main__":
    main()
