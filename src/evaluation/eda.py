"""Reusable exploratory analysis helpers for image, mask, and metadata EDA."""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from src.data.catalog import SatelliteCatalog
from src.data.indexing import PatchDatasetConfig, ProcessedImagePair, discover_image_mask_pairs


Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class PatchDatasetStatistics:
    """Summarise patch-level sparsity across a processed dataset."""

    total_images: int
    total_patches: int
    empty_patches: int
    non_empty_patches: int
    positive_pixel_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        """Return the summary in a notebook-friendly dictionary form."""
        return asdict(self)


def compute_image_summary_dataframe(
    root_dir: str | Path | PatchDatasetConfig,
) -> pd.DataFrame:
    """Summarise each processed image and mask pair at full-frame level."""
    pairs = get_image_mask_pairs(root_dir)
    rows: list[dict[str, str | int | float]] = []

    for pair in pairs:
        image_array = _load_grayscale_array(pair.image_path)
        mask_array = _load_grayscale_array(pair.mask_path) > 0
        height, width = image_array.shape
        rows.append(
            {
                "image_name": pair.image_path.name,
                "width": width,
                "height": height,
                "image_mean": float(image_array.mean()),
                "image_std": float(image_array.std()),
                "mask_positive_pixels": int(mask_array.sum()),
                "mask_positive_fraction": float(mask_array.mean()),
            }
        )

    return pd.DataFrame(rows)


def summarise_patches_by_image(patch_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise patch counts and sparsity for each full image."""
    grouped = (
        patch_df.groupby("image_name")
        .agg(
            total_patches=("image_name", "size"),
            empty_patches=("is_empty", "sum"),
            mean_positive_fraction=("positive_pixel_fraction", "mean"),
            max_positive_fraction=("positive_pixel_fraction", "max"),
        )
        .reset_index()
    )
    grouped["empty_patches"] = grouped["empty_patches"].astype(int)
    grouped["non_empty_patches"] = (
        grouped["total_patches"] - grouped["empty_patches"]
    ).astype(int)
    return grouped


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    """Create the figure output directory if it does not yet exist."""
    path = Path(output_dir) if output_dir is not None else Path("results") / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log(logger: logging.Logger | None, message: str, *args: object) -> None:
    """Emit a log message when a logger is available."""
    if logger is not None:
        logger.info(message, *args)


def _load_grayscale_array(path: Path) -> np.ndarray:
    """Load a grayscale PNG as a NumPy array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _calculate_display_size(
    width: int,
    height: int,
    max_size: int,
) -> tuple[int, int]:
    """Calculate a resized image shape that preserves aspect ratio."""
    if max(width, height) <= max_size:
        return width, height

    scale = max_size / float(max(width, height))
    return int(width * scale), int(height * scale)


def _load_resized_grayscale_array(
    path: Path,
    max_size: int = 2048,
    resample: int = Image.Resampling.BILINEAR,
) -> np.ndarray:
    """Load a grayscale PNG resized for efficient full-frame display."""
    with Image.open(path) as image:
        grayscale = image.convert("L")
        target_size = _calculate_display_size(
            width=grayscale.width,
            height=grayscale.height,
            max_size=max_size,
        )
        if target_size != grayscale.size:
            grayscale = grayscale.resize(target_size, resample=resample)
        return np.asarray(grayscale, dtype=np.uint8)


def _load_patch_arrays(
    image_path: Path,
    mask_path: Path,
    x: int,
    y: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load aligned image and mask patches from a processed pair."""
    with Image.open(image_path) as image:
        image_patch = np.asarray(
            image.crop((x, y, x + patch_size, y + patch_size)).convert("L"),
            dtype=np.uint8,
        )

    with Image.open(mask_path) as mask_image:
        mask_patch = np.asarray(
            mask_image.crop((x, y, x + patch_size, y + patch_size)).convert("L"),
            dtype=np.uint8,
        )

    return image_patch, mask_patch


def _normalise_image_for_display(image: np.ndarray) -> np.ndarray:
    """Scale an astronomical image to the unit interval for clearer plotting."""
    lower = float(np.percentile(image, 1.0))
    upper = float(np.percentile(image, 99.5))
    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)
    clipped = np.clip(image.astype(np.float32), lower, upper)
    return (clipped - lower) / (upper - lower)


def _iter_patch_positions(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> Iterable[tuple[int, int, int, int]]:
    """Yield patch grid indices and top-left coordinates."""
    for grid_y, y in enumerate(range(0, height - patch_size + 1, stride)):
        for grid_x, x in enumerate(range(0, width - patch_size + 1, stride)):
            yield grid_y, grid_x, y, x


def _prepare_figure(filename: str, output_dir: str | Path | None) -> Path:
    """Resolve a figure path beneath the default figure directory."""
    return _resolve_output_dir(output_dir) / filename


def get_image_mask_pairs(
    root_dir: str | Path | PatchDatasetConfig,
) -> list[ProcessedImagePair]:
    """
    Discover processed image and mask pairs beneath a directory.

    Parameters
    ----------
    root_dir:
        Directory containing `*_red.fits_full.png` images and matching masks,
        or a validated dataset configuration.
    """
    config = (
        root_dir
        if isinstance(root_dir, PatchDatasetConfig)
        else PatchDatasetConfig(root_dir=Path(root_dir), patch_size=1)
    )
    return discover_image_mask_pairs(config)


def compute_patch_dataframe(
    root_dir: str | Path | PatchDatasetConfig,
    patch_size: int = 512,
    stride: int | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Compute patch-level sparsity and coordinate statistics for a dataset.

    The function scans binary masks only, so it is suitable for exploratory
    analysis without constructing a PyTorch dataset.
    """
    config = (
        root_dir
        if isinstance(root_dir, PatchDatasetConfig)
        else PatchDatasetConfig(
            root_dir=Path(root_dir),
            patch_size=patch_size,
            stride=stride,
        )
    )
    rows: list[dict[str, int | float | str | bool]] = []
    pairs = get_image_mask_pairs(config)

    for pair in pairs:
        _log(logger, "Analysing mask patches for %s", pair.image_path.name)
        mask_array = _load_grayscale_array(pair.mask_path) > 0
        height, width = mask_array.shape
        total_pixels = config.patch_size * config.patch_size

        for grid_y, grid_x, y, x in _iter_patch_positions(
            height=height,
            width=width,
            patch_size=config.patch_size,
            stride=config.resolved_stride,
        ):
            patch_mask = mask_array[
                y : y + config.patch_size,
                x : x + config.patch_size,
            ]
            positive_pixels = int(patch_mask.sum())
            positive_fraction = positive_pixels / total_pixels
            rows.append(
                {
                    "image_name": pair.image_path.name,
                    "image_path": str(pair.image_path),
                    "mask_path": str(pair.mask_path),
                    "grid_y": grid_y,
                    "grid_x": grid_x,
                    "y": y,
                    "x": x,
                    "patch_size": config.patch_size,
                    "stride": config.resolved_stride,
                    "positive_pixels": positive_pixels,
                    "positive_pixel_fraction": positive_fraction,
                    "is_empty": positive_pixels == 0,
                }
            )

    patch_df = pd.DataFrame(rows)
    if patch_df.empty:
        raise RuntimeError(
            f"No patch statistics were generated for {config.root_dir}"
        )

    return patch_df


def summarise_patch_dataframe(patch_df: pd.DataFrame) -> PatchDatasetStatistics:
    """Summarise a patch statistics DataFrame into headline meeting metrics."""
    total_patches = int(len(patch_df))
    empty_patches = int(patch_df["is_empty"].sum())
    non_empty_patches = total_patches - empty_patches
    positive_pixel_fraction = float(patch_df["positive_pixels"].sum()) / float(
        total_patches * int(patch_df["patch_size"].iloc[0]) ** 2
    )
    return PatchDatasetStatistics(
        total_images=int(patch_df["image_name"].nunique()),
        total_patches=total_patches,
        empty_patches=empty_patches,
        non_empty_patches=non_empty_patches,
        positive_pixel_fraction=positive_pixel_fraction,
    )


def plot_random_image_mask_pairs(
    root_dir: str | Path,
    sample_count: int = 4,
    seed: int = 42,
    output_dir: str | Path | None = None,
    display_max_size: int = 2048,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Plot random full image and mask pairs side by side."""
    import matplotlib.pyplot as plt

    pairs = get_image_mask_pairs(root_dir)
    sample_count = min(sample_count, len(pairs))
    sampled_pairs = random.Random(seed).sample(pairs, k=sample_count)

    fig, axes = plt.subplots(sample_count, 2, figsize=(12, 4 * sample_count))
    axes_array = np.atleast_2d(axes)

    for row, pair in enumerate(sampled_pairs):
        image = _normalise_image_for_display(
            _load_resized_grayscale_array(
                pair.image_path,
                max_size=display_max_size,
                resample=Image.Resampling.BILINEAR,
            )
        )
        mask = _load_resized_grayscale_array(
            pair.mask_path,
            max_size=display_max_size,
            resample=Image.Resampling.NEAREST,
        )

        axes_array[row, 0].imshow(image, cmap="gray")
        axes_array[row, 0].set_title(f"Image: {pair.image_path.stem}")
        axes_array[row, 0].axis("off")

        axes_array[row, 1].imshow(mask, cmap="magma")
        axes_array[row, 1].set_title(f"Mask: {pair.mask_path.stem}")
        axes_array[row, 1].axis("off")

    fig.suptitle("Random Image and Mask Pairs", fontsize=14)
    fig.tight_layout()

    output_path = _prepare_figure("eda_random_image_mask_pairs.png", output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved random image and mask pairs to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_mask_overlay(
    image_path: str | Path,
    mask_path: str | Path,
    output_name: str,
    output_dir: str | Path | None = None,
    alpha: float = 0.35,
    display_max_size: int = 2048,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Overlay a binary mask on top of a grayscale astronomical image."""
    import matplotlib.pyplot as plt

    image_array = _normalise_image_for_display(
        _load_resized_grayscale_array(
            Path(image_path),
            max_size=display_max_size,
            resample=Image.Resampling.BILINEAR,
        )
    )
    mask_array = _load_resized_grayscale_array(
        Path(mask_path),
        max_size=display_max_size,
        resample=Image.Resampling.NEAREST,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image_array, cmap="gray")
    ax.imshow(np.ma.masked_where(mask_array == 0, mask_array), cmap="autumn", alpha=alpha)
    ax.set_title(Path(image_path).stem)
    ax.axis("off")
    fig.tight_layout()

    output_path = _prepare_figure(output_name, output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved mask overlay to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_random_mask_overlays(
    root_dir: str | Path,
    sample_count: int = 4,
    seed: int = 42,
    output_dir: str | Path | None = None,
    alpha: float = 0.35,
    display_max_size: int = 2048,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Plot several random mask overlays in a single meeting-ready figure."""
    import matplotlib.pyplot as plt

    pairs = get_image_mask_pairs(root_dir)
    sample_count = min(sample_count, len(pairs))
    sampled_pairs = random.Random(seed).sample(pairs, k=sample_count)

    fig, axes = plt.subplots(1, sample_count, figsize=(5 * sample_count, 5))
    axes_array = np.atleast_1d(axes)

    for axis, pair in zip(axes_array, sampled_pairs, strict=False):
        image = _normalise_image_for_display(
            _load_resized_grayscale_array(
                pair.image_path,
                max_size=display_max_size,
                resample=Image.Resampling.BILINEAR,
            )
        )
        mask = _load_resized_grayscale_array(
            pair.mask_path,
            max_size=display_max_size,
            resample=Image.Resampling.NEAREST,
        )
        axis.imshow(image, cmap="gray")
        axis.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=alpha)
        axis.set_title(pair.image_path.stem)
        axis.axis("off")

    fig.suptitle("Mask Overlays on Astronomical Images", fontsize=14)
    fig.tight_layout()

    output_path = _prepare_figure("eda_random_mask_overlays.png", output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved random mask overlays to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_patch_density_distribution(
    patch_df: pd.DataFrame,
    output_dir: str | Path | None = None,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Plot the distribution of mask density across all indexed patches."""
    import matplotlib.pyplot as plt

    non_empty = patch_df.loc[~patch_df["is_empty"], "positive_pixel_fraction"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(patch_df["positive_pixel_fraction"], bins=40, color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Positive pixel fraction")
    axes[0].set_ylabel("Patch count")
    axes[0].set_title("All patches")

    if non_empty.empty:
        axes[1].text(0.5, 0.5, "No non-empty patches", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        axes[1].hist(non_empty, bins=30, color="#d62728")
        axes[1].set_xlabel("Positive pixel fraction")
        axes[1].set_ylabel("Patch count")
        axes[1].set_title("Non-empty patches only")

    fig.suptitle("Patch Mask Density Distribution", fontsize=14)
    fig.tight_layout()

    output_path = _prepare_figure("eda_patch_density_distribution.png", output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved patch density distribution to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_patch_density_heatmap(
    patch_df: pd.DataFrame,
    image_name: str,
    output_dir: str | Path | None = None,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Plot the per-patch mask density heatmap for one source image."""
    import matplotlib.pyplot as plt

    subset = patch_df.loc[patch_df["image_name"] == image_name].copy()
    if subset.empty:
        raise ValueError(f"No patch rows found for {image_name}")

    heatmap = subset.pivot(index="grid_y", columns="grid_x", values="positive_pixel_fraction")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(heatmap.to_numpy(), cmap="magma", origin="upper")
    ax.set_title(f"Patch Density Heatmap: {image_name}")
    ax.set_xlabel("Patch x index")
    ax.set_ylabel("Patch y index")
    fig.colorbar(image, ax=ax, label="Positive pixel fraction")
    fig.tight_layout()

    safe_stem = Path(image_name).stem.replace(".", "_")
    output_path = _prepare_figure(
        f"eda_patch_density_heatmap_{safe_stem}.png",
        output_dir,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved patch density heatmap to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_image_level_summary(
    image_df: pd.DataFrame,
    patch_df: pd.DataFrame,
    output_dir: str | Path | None = None,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Plot full-image mask coverage and non-empty patch counts by image."""
    import matplotlib.pyplot as plt

    patch_summary = summarise_patches_by_image(patch_df).set_index("image_name")
    image_summary = image_df.set_index("image_name")
    combined = image_summary.join(
        patch_summary.loc[:, ["non_empty_patches"]],
        how="left",
    ).reset_index().sort_values("non_empty_patches")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].barh(
        combined["image_name"],
        combined["mask_positive_fraction"],
        color="#1f77b4",
    )
    axes[0].set_title("Full-image mask coverage")
    axes[0].set_xlabel("Positive pixel fraction")
    axes[0].set_ylabel("Image")

    axes[1].barh(
        combined["image_name"],
        combined["non_empty_patches"],
        color="#d62728",
    )
    axes[1].set_title("Non-empty patches by image")
    axes[1].set_xlabel("Patch count")
    axes[1].set_ylabel("Image")

    fig.tight_layout()
    output_path = _prepare_figure("eda_image_level_summary.png", output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved image-level summary plot to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_example_satellite_trail_patches(
    patch_df: pd.DataFrame,
    sample_count: int = 6,
    sort_by_density: bool = True,
    seed: int = 42,
    output_dir: str | Path | None = None,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Plot example non-empty patches to show clear satellite trail structure.

    Parameters
    ----------
    patch_df:
        Patch statistics DataFrame returned by `compute_patch_dataframe`.
    sample_count:
        Number of non-empty patches to visualise.
    sort_by_density:
        When true, show the densest trail patches first.
    """
    import matplotlib.pyplot as plt

    non_empty = patch_df.loc[~patch_df["is_empty"]].copy()
    if non_empty.empty:
        raise ValueError("No non-empty patches are available for visualisation")

    sample_count = min(sample_count, len(non_empty))
    if sort_by_density:
        selected = non_empty.nlargest(sample_count, "positive_pixel_fraction")
    else:
        selected = non_empty.sample(n=sample_count, random_state=seed)

    fig, axes = plt.subplots(sample_count, 3, figsize=(12, 4 * sample_count))
    axes_array = np.atleast_2d(axes)

    for row, patch in enumerate(selected.itertuples(index=False)):
        image_patch, mask_patch = _load_patch_arrays(
            image_path=Path(patch.image_path),
            mask_path=Path(patch.mask_path),
            x=int(patch.x),
            y=int(patch.y),
            patch_size=int(patch.patch_size),
        )
        display_image = _normalise_image_for_display(image_patch)

        axes_array[row, 0].imshow(display_image, cmap="gray")
        axes_array[row, 0].set_title("Image patch")
        axes_array[row, 0].axis("off")

        axes_array[row, 1].imshow(mask_patch, cmap="magma")
        axes_array[row, 1].set_title("Mask patch")
        axes_array[row, 1].axis("off")

        axes_array[row, 2].imshow(display_image, cmap="gray")
        axes_array[row, 2].imshow(
            np.ma.masked_where(mask_patch == 0, mask_patch),
            cmap="autumn",
            alpha=0.4,
        )
        axes_array[row, 2].set_title(
            f"Overlay | y={patch.y}, x={patch.x} | density={patch.positive_pixel_fraction:.4f}"
        )
        axes_array[row, 2].axis("off")

    fig.suptitle("Example Satellite Trail Patches", fontsize=14)
    fig.tight_layout()

    output_path = _prepare_figure("eda_example_satellite_trail_patches.png", output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved example satellite trail patches to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_satellite_name_frequency(
    catalog: SatelliteCatalog,
    top_n: int = 15,
    output_dir: str | Path | None = None,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Plot the most frequent satellite names in the metadata catalogue."""
    import matplotlib.pyplot as plt

    frequency = catalog.get_satellite_frequency(top_n=top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    frequency.plot.barh(ax=ax, color="#2ca02c")
    ax.set_xlabel("Trail count")
    ax.set_ylabel("Satellite name")
    ax.set_title(f"Top {len(frequency)} Satellite Labels in the Catalogue")
    fig.tight_layout()

    output_path = _prepare_figure("eda_satellite_name_frequency.png", output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved satellite frequency plot to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_trail_length_distribution(
    catalog: SatelliteCatalog,
    output_dir: str | Path | None = None,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Plot the metadata trail length distribution."""
    import matplotlib.pyplot as plt

    lengths = catalog.get_trail_lengths()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=30, color="#9467bd", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Trail length")
    ax.set_ylabel("Count")
    ax.set_title("Satellite Trail Length Distribution")
    fig.tight_layout()

    output_path = _prepare_figure("eda_trail_length_distribution.png", output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved trail length distribution to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_metadata_missing_values(
    catalog: SatelliteCatalog,
    output_dir: str | Path | None = None,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Plot missing-value counts for the metadata catalogue."""
    import matplotlib.pyplot as plt

    missing = catalog.get_missing_value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    missing.plot.bar(ax=ax, color="#8c564b")
    ax.set_xlabel("Metadata column")
    ax.set_ylabel("Missing value count")
    ax.set_title("Metadata Missing Values")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    output_path = _prepare_figure("eda_metadata_missing_values.png", output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved metadata missing-value plot to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_ra_dec_distributions(
    catalog: SatelliteCatalog,
    output_dir: str | Path | None = None,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Plot start and end distributions for right ascension and declination."""
    import matplotlib.pyplot as plt

    coordinates = catalog.get_coordinate_dataframe()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(coordinates["Start_RA"], bins=40, alpha=0.6, label="Start RA")
    axes[0].hist(coordinates["End_RA"], bins=40, alpha=0.6, label="End RA")
    axes[0].set_title("Right Ascension Distribution")
    axes[0].set_xlabel("Right ascension")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(coordinates["Start_DEC"], bins=40, alpha=0.6, label="Start DEC")
    axes[1].hist(coordinates["End_DEC"], bins=40, alpha=0.6, label="End DEC")
    axes[1].set_title("Declination Distribution")
    axes[1].set_xlabel("Declination")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    output_path = _prepare_figure("eda_ra_dec_distributions.png", output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved RA and DEC distribution plot to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_ra_dec_ranges(
    catalog: SatelliteCatalog,
    output_dir: str | Path | None = None,
    show: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Plot catalogue trail segments in right ascension and declination space."""
    import matplotlib.pyplot as plt

    coordinates = catalog.get_coordinate_dataframe()
    ranges = catalog.get_ra_dec_ranges()

    fig, ax = plt.subplots(figsize=(8, 6))
    for row in coordinates.itertuples(index=False):
        ax.plot(
            [row.Start_RA, row.End_RA],
            [row.Start_DEC, row.End_DEC],
            color="#1f77b4",
            alpha=0.3,
            linewidth=0.8,
        )

    ax.scatter(
        coordinates["Start_RA"],
        coordinates["Start_DEC"],
        s=10,
        color="#ff7f0e",
        label="Start",
    )
    ax.scatter(
        coordinates["End_RA"],
        coordinates["End_DEC"],
        s=10,
        color="#2ca02c",
        label="End",
    )
    ax.set_xlabel("Right ascension")
    ax.set_ylabel("Declination")
    ax.set_title(
        "Catalogue Trail Coordinates\n"
        f"RA range: {ranges['ra'][0]:.2f} to {ranges['ra'][1]:.2f} | "
        f"DEC range: {ranges['dec'][0]:.2f} to {ranges['dec'][1]:.2f}"
    )
    ax.legend()
    fig.tight_layout()

    output_path = _prepare_figure("eda_ra_dec_ranges.png", output_dir)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    _log(logger, "Saved RA and DEC range plot to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)
    return output_path
