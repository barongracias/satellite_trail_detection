"""Compute per-image mean/std from processed images and add to manifest."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def compute_image_stats(image_path: Path) -> tuple[float, float]:
    """Return (mean, std) of a grayscale image on the [0, 1] scale.

    Stats are computed on pixel values divided by 255 to match the scale
    of patches after ``JointTransform`` (which converts PIL→float and /255).
    """
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return float(arr.mean()), float(arr.std())


def main() -> None:
    manifest_path = Path("data/patches/manifest.csv")
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    manifest = pd.read_csv(manifest_path)
    source_images = manifest["source_image"].unique()
    image_stats = {}

    for src in source_images:
        src_path = Path(src)
        if src_path.exists():
            mean, std = compute_image_stats(src_path)
            image_stats[str(src)] = (mean, std)
            print(f"{src_path.name}: mean={mean:.4f}, std={std:.4f}")
        else:
            print(f"Warning: source image not found: {src} (writing NaN stats)")

    # NaN for missing sources so PatchDirectoryDataset's pd.isna fallback engages
    # rather than silently using bogus (0, 1) stats that look valid downstream.
    nan_pair = (float("nan"), float("nan"))
    manifest["image_mean"] = manifest["source_image"].map(
        lambda x: image_stats.get(x, nan_pair)[0]
    )
    manifest["image_std"] = manifest["source_image"].map(
        lambda x: image_stats.get(x, nan_pair)[1]
    )

    manifest.to_csv(manifest_path, index=False)
    print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
