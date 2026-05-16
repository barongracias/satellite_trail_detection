#!/usr/bin/env python
"""Hough post-processing on U-Net predicted masks.

For each test image, reconstructs the full predicted binary mask from sampled
test patches, applies probabilistic Hough transform to detect line-like structures,
then reports FNR before and after post-processing.

Deviations from Stoppa et al. 2024:
- Hough is applied to the reconstructed patch-subset canvas rather than a
  dense full-image prediction, because the pre-built test patches use 1:3
  pos:neg sampling (all positive patches are retained).
- Image-level FNR: a test image is trail-positive if any of its test patches
  have positive_pixel_fraction > 0 in the manifest.

Usage (CSD3):
    CHECKPOINT=results/checkpoints/unet_sweep_best_best.pth \
    THRESHOLD=0.63 \
      sbatch slurm/hough_postprocess.sbatch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.config.constants import PAPER_FNR_POST_HOUGH, PAPER_FNR_PRE_HOUGH
from src.models.unet import UNet
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

Image.MAX_IMAGE_PIXELS = None

_PATCH_SIZE = 512
_YX_RE = re.compile(r"_(\d+)_(\d+)_image$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch_dir", default="data/patches")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hough_threshold", type=int, default=50)
    p.add_argument("--min_line_length", type=int, default=100)
    p.add_argument("--max_line_gap", type=int, default=50)
    p.add_argument("--line_thickness", type=int, default=3)
    p.add_argument("--out", default="results/classical/hough_postprocess.json")
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[UNet, str]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = UNet(
        base_channels=cfg.get("base_channels", 8),
        dropout_rate=cfg.get("dropout_rate", 0.5),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg.get("normalisation", "fixed")


def _normalise(t: torch.Tensor, normalisation: str) -> torch.Tensor:
    if normalisation == "per_image":
        return (t - t.mean()) / (t.std() + 1e-6)
    return (t - 0.5) / 0.5


def _parse_yx(patch_path: str) -> tuple[int, int]:
    m = _YX_RE.search(Path(patch_path).stem)
    if m is None:
        raise ValueError(f"Cannot parse y,x from patch filename: {patch_path}")
    return int(m.group(1)), int(m.group(2))


def _infer_patch(
    patch_path: str,
    model: UNet,
    normalisation: str,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    with Image.open(patch_path) as img:
        t = TF.pil_to_tensor(img.convert("L")).float() / 255.0
    t = _normalise(t, normalisation).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    return (prob >= threshold).astype(np.uint8) * 255


def _apply_hough(
    canvas: np.ndarray,
    hough_threshold: int,
    min_line_length: int,
    max_line_gap: int,
    line_thickness: int,
) -> np.ndarray:
    lines = cv2.HoughLinesP(
        canvas,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    result = np.zeros_like(canvas)
    if lines is None:
        return result
    for line in lines[:, 0]:
        x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        cv2.line(result, (x1, y1), (x2, y2), color=255, thickness=line_thickness)
    return result


def main() -> None:
    args = parse_args()
    logger = get_logger("hough_postprocess")
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, normalisation = load_model(args.checkpoint, device)
    logger.info("Loaded %s (normalisation=%s, threshold=%.3f)", args.checkpoint, normalisation, args.threshold)

    manifest = pd.read_csv(Path(args.patch_dir) / "manifest.csv")
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    logger.info("Test patches in manifest: %d", len(test_df))

    groups = test_df.groupby("source_image")
    logger.info("Unique test images: %d", len(groups))

    rows = []
    n_positive = 0
    n_fn_pre = 0
    n_fn_post = 0

    for source_image, group in groups:
        is_positive = bool((group["positive_pixel_fraction"] > 0).any())

        yx_list = [_parse_yx(p) for p in group["patch_path"]]
        canvas_h = max(y for y, _ in yx_list) + _PATCH_SIZE
        canvas_w = max(x for _, x in yx_list) + _PATCH_SIZE
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        for patch_path in group["patch_path"]:
            pred = _infer_patch(patch_path, model, normalisation, args.threshold, device)
            y, x = _parse_yx(patch_path)
            canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE] = np.maximum(
                canvas[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE], pred
            )

        detected_pre = bool(canvas.any())
        hough_canvas = _apply_hough(
            canvas,
            hough_threshold=args.hough_threshold,
            min_line_length=args.min_line_length,
            max_line_gap=args.max_line_gap,
            line_thickness=args.line_thickness,
        )
        detected_post = bool(canvas.any() or hough_canvas.any())

        if is_positive:
            n_positive += 1
            if not detected_pre:
                n_fn_pre += 1
            if not detected_post:
                n_fn_post += 1

        logger.info(
            "%s | positive=%s | pre=%s | post=%s",
            Path(source_image).name, is_positive, detected_pre, detected_post,
        )
        rows.append({
            "source_image": str(source_image),
            "is_positive": is_positive,
            "detected_pre_hough": detected_pre,
            "detected_post_hough": detected_post,
        })

    fnr_pre = n_fn_pre / n_positive if n_positive > 0 else float("nan")
    fnr_post = n_fn_post / n_positive if n_positive > 0 else float("nan")

    result = {
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "hough_threshold": args.hough_threshold,
        "min_line_length": args.min_line_length,
        "max_line_gap": args.max_line_gap,
        "n_test_images": len(rows),
        "n_positive_images": n_positive,
        "fnr_pre_hough": round(fnr_pre, 6),
        "fnr_post_hough": round(fnr_post, 6),
        "paper_fnr_pre_hough": PAPER_FNR_PRE_HOUGH,
        "paper_fnr_post_hough": PAPER_FNR_POST_HOUGH,
        "per_image": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "FNR  pre-Hough: %.4f  post-Hough: %.4f  (paper: %.4f → %.4f)",
        fnr_pre, fnr_post, PAPER_FNR_PRE_HOUGH, PAPER_FNR_POST_HOUGH,
    )
    logger.info("Saved to %s", out_path)
    print(
        f"\nFNR pre-Hough:  {fnr_pre:.4f}  (paper: {PAPER_FNR_PRE_HOUGH})"
        f"\nFNR post-Hough: {fnr_post:.4f}  (paper: {PAPER_FNR_POST_HOUGH})"
    )


if __name__ == "__main__":
    main()
