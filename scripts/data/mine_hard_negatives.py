#!/usr/bin/env python
"""Mine train-split empty-mask patches that the segmentation model marks positive."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config.constants import PATCH_SIZE
from src.data.dataset import PatchDirectoryDataset
from src.models.loading import load_segmentation_model
from src.utils.seed import seed_everything


class IndexedSubset(Dataset[dict[str, Any]]):
    """Wrap PatchDirectoryDataset while preserving source row indices."""

    def __init__(self, base: PatchDirectoryDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        source_idx = self.indices[idx]
        sample = self.base[source_idx]
        return {"image": sample["image"], "mask": sample["mask"], "source_index": source_idx}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patch_dir", default="data/patches")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--out",
        default="results/classical/hard_negative_train_t44_s2804.json",
    )
    return parser.parse_args()


def _row_payload(row: pd.Series, fp_pixels: int, mean_probability: float) -> dict[str, Any]:
    payload = {
        "patch_path": str(row["patch_path"]),
        "mask_path": str(row["mask_path"]),
        "source_image": str(row["source_image"]),
        "positive_pixel_fraction": float(row.get("positive_pixel_fraction", 0.0)),
        "fp_pixels": int(fp_pixels),
        "pred_positive_fraction": float(fp_pixels / (PATCH_SIZE * PATCH_SIZE)),
        "mean_probability": float(mean_probability),
    }
    if "image_mean" in row and pd.notna(row["image_mean"]):
        payload["image_mean"] = float(row["image_mean"])
    if "image_std" in row and pd.notna(row["image_std"]):
        payload["image_std"] = float(row["image_std"])
    return payload


def main() -> None:
    args = parse_args()
    if not (0.0 < args.threshold < 1.0):
        raise SystemExit("--threshold must lie in (0, 1)")
    if args.top_k <= 0:
        raise SystemExit("--top_k must be positive")

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalisation = load_segmentation_model(args.checkpoint, device)

    dataset = PatchDirectoryDataset(
        Path(args.patch_dir) / "train",
        normalisation=normalisation,
        augment_train=False,
        noise_augment=False,
    )
    fractions = pd.to_numeric(dataset.records["positive_pixel_fraction"], errors="coerce").fillna(0.0)
    empty_indices = fractions[fractions <= 0.0].index.astype(int).tolist()
    subset = IndexedSubset(dataset, empty_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    hard_negatives: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            probs = torch.sigmoid(model(images)).squeeze(1)
            pred = probs >= args.threshold
            fp_pixels = pred.flatten(1).sum(dim=1).cpu().tolist()
            mean_probs = probs.flatten(1).mean(dim=1).cpu().tolist()
            source_indices = batch["source_index"].cpu().tolist()
            for source_idx, fp_count, mean_prob in zip(source_indices, fp_pixels, mean_probs):
                if fp_count <= 0:
                    continue
                row = dataset.records.iloc[int(source_idx)]
                hard_negatives.append(_row_payload(row, int(fp_count), float(mean_prob)))

    hard_negatives.sort(
        key=lambda item: (item["fp_pixels"], item["mean_probability"]),
        reverse=True,
    )
    selected = hard_negatives[: args.top_k]
    out = {
        "checkpoint": str(args.checkpoint),
        "patch_dir": str(args.patch_dir),
        "split": "train",
        "normalisation": normalisation,
        "threshold": float(args.threshold),
        "top_k": int(args.top_k),
        "candidate_empty_patches": len(empty_indices),
        "false_positive_empty_patches": len(hard_negatives),
        "selected_count": len(selected),
        "hard_negatives": selected,
        "generated": date.today().isoformat(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"Mined {len(selected)}/{len(hard_negatives)} hard negatives "
        f"from {len(empty_indices)} empty train patches -> {out_path}"
    )


if __name__ == "__main__":
    main()
