"""Build a pre-split, pre-sampled patch dataset on disk.

Reads all image/mask pairs from data/Processed/, performs an image-level
stratified split, extracts 528×528 patches at stride 528, applies 1:3 pos:neg
sampling (all positive patches + 3× random negatives), writes canonical patch
orientations for every split, and saves the resulting PNG pairs under
data/patches/{train,val,test}/. A manifest CSV is written to data/patches/.

Usage (CSD3 — do NOT run locally):
    python scripts/data/build_patch_dataset.py \
        --data-root data/Processed \
        --out-dir   data/patches \
        --patch-size 528 \
        --stride    528 \
        --seed      2804
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from PIL import Image

from src.config.constants import GLOBAL_SEED, PATCH_SIZE
from src.data.indexing import PatchDatasetConfig, discover_image_mask_pairs
from src.data.splits import create_image_level_splits
from src.data.transforms import get_eval_transforms
from src.utils.seed import seed_everything


Image.MAX_IMAGE_PIXELS = None

_MANIFEST_COLUMNS = [
    "split",
    "source_image",
    "patch_path",
    "mask_path",
    "positive_pixel_fraction",
]

PatchEntry = tuple[Path, Path, int, int, int]


def _count_trail_pixels(mask_path: Path) -> int:
    with Image.open(mask_path) as m:
        arr = np.asarray(m.convert("L"), dtype=np.uint8)
    return int((arr > 0).sum())


def _extract_split_patches(
    pairs: list,
    patch_size: int,
    stride: int,
) -> tuple[list[PatchEntry], list[PatchEntry]]:
    """Return lightweight positive/negative patch metadata for one split."""
    pos_patches: list[PatchEntry] = []
    neg_patches: list[PatchEntry] = []

    for pair in pairs:
        with Image.open(pair.mask_path) as msk_pil:
            msk_full = msk_pil.convert("L")

        for y in range(0, pair.height - patch_size + 1, stride):
            for x in range(0, pair.width - patch_size + 1, stride):
                msk_patch = msk_full.crop((x, y, x + patch_size, y + patch_size))
                pos_pixels = int((np.asarray(msk_patch, dtype=np.uint8) > 0).sum())
                entry = (pair.image_path, pair.mask_path, y, x, pos_pixels)
                if pos_pixels > 0:
                    pos_patches.append(entry)
                else:
                    neg_patches.append(entry)

    return pos_patches, neg_patches


def _write_patches(
    pos_patches: list[PatchEntry],
    neg_patches: list[PatchEntry],
    out_dir: Path,
    split: str,
    transform,
    patch_size: int,
    rng: random.Random,
    manifest_rows: list,
    no_neg_sampling: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_pixels = patch_size * patch_size

    # Default: 1:3 sampling — all positives, random sample of negatives.
    # no_neg_sampling=True: keep every negative patch (matches the paper's
    # natural class distribution; used when building a parity test set).
    if no_neg_sampling:
        sampled_neg = neg_patches
    else:
        n_neg = min(len(neg_patches), len(pos_patches) * 3)
        sampled_neg = rng.sample(neg_patches, n_neg) if n_neg > 0 else []
    all_patches = pos_patches + sampled_neg

    entries_by_source: dict[tuple[Path, Path], list[PatchEntry]] = defaultdict(list)
    for entry in all_patches:
        src_img, src_msk, *_ = entry
        entries_by_source[(src_img, src_msk)].append(entry)

    for (src_img, src_msk), entries in entries_by_source.items():
        with Image.open(src_img) as img_pil:
            img_full = img_pil.convert("L")
        with Image.open(src_msk) as msk_pil:
            msk_full = msk_pil.convert("L")

        for _, _, y, x, pos_pixels in entries:
            stem = f"{src_img.stem}_{y}_{x}"
            img_out = out_dir / f"{stem}_image.png"
            msk_out = out_dir / f"{stem}_mask.png"

            crop_box = (x, y, x + patch_size, y + patch_size)
            img_patch = img_full.crop(crop_box)
            msk_patch = msk_full.crop(crop_box)
            img_t, msk_t = transform(img_patch, msk_patch)

            # img_t: [1, H, W] float32 in [0, 1]; convert back to uint8 for PNG
            img_arr = (img_t.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)
            msk_arr = (msk_t.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_arr, mode="L").save(img_out)
            Image.fromarray(msk_arr, mode="L").save(msk_out)

            pos_frac = pos_pixels / total_pixels

            manifest_rows.append(
                {
                    "split": split,
                    "source_image": str(src_img),
                    "patch_path": str(img_out),
                    "mask_path": str(msk_out),
                    "positive_pixel_fraction": round(pos_frac, 8),
                }
            )


def build(
    data_root: Path,
    out_dir: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = GLOBAL_SEED,
    splits: tuple[str, ...] = ("train", "val", "test"),
    no_neg_sampling: bool = False,
) -> Path:
    """Run the full patch-building pipeline and return the manifest path."""
    seed_everything(seed)
    config = PatchDatasetConfig(root_dir=data_root, patch_size=patch_size, stride=stride)
    pairs = discover_image_mask_pairs(config)

    trail_counts = [_count_trail_pixels(p.mask_path) for p in pairs]
    train_pairs, val_pairs, test_pairs = create_image_level_splits(
        pairs, trail_counts, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    eval_transform = get_eval_transforms()
    # Shared across splits: train negatives are sampled first, then val, then
    # test. A standalone test-only build (uncommon: parity-test uses
    # --no-neg-sampling and is unaffected) would consume the RNG in a
    # different order and therefore pick a different test-negative subset.
    # The preregistered workflow builds all three together so this is stable.
    rng = random.Random(seed)
    manifest_rows: list = []

    for split_name, split_pairs, transform in [
        ("train", train_pairs, eval_transform),
        ("val", val_pairs, eval_transform),
        ("test", test_pairs, eval_transform),
    ]:
        if split_name not in splits or not split_pairs:
            continue
        pos_patches, neg_patches = _extract_split_patches(
            split_pairs, patch_size, stride
        )
        _write_patches(
            pos_patches,
            neg_patches,
            out_dir=out_dir / split_name,
            split=split_name,
            transform=transform,
            patch_size=patch_size,
            rng=rng,
            manifest_rows=manifest_rows,
            no_neg_sampling=no_neg_sampling,
        )

    manifest_path = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/Processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/patches"))
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_SIZE)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to build. Use 'test' alone to build only "
             "the parity test set (see --no-neg-sampling).",
    )
    parser.add_argument(
        "--no-neg-sampling",
        action="store_true",
        help="Disable 1:3 pos:neg sampling and keep all negative patches. "
             "Use with --splits test --out-dir data/patches_test_full to build "
             "a paper-comparable test set with the natural class distribution.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    manifest = build(
        data_root=args.data_root,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        splits=splits,
        no_neg_sampling=args.no_neg_sampling,
    )
    print(f"Manifest written to {manifest}")
