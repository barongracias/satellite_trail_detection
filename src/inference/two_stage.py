"""Two-stage satellite trail detector: CNN patch-classifier gate → U-Net segmentor.

CLI usage::

    python -m src.inference.two_stage \\
        --classifier  results/checkpoints/classifier_base_latest.pth \\
        --clf-threshold 0.18 \\
        --unet        results/checkpoints/<winner>_best.pth \\
        --unet-threshold <val-optimal> \\
        --manifest    data/patches/manifest.csv \\
        --split       test \\
        --out         results/classical/two_stage_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn

from src.classical.hough_runner import run_hough_on_canvas
from src.config.constants import PATCH_SIZE
from src.data.transforms import normalise_tensor
from src.models.classifier import PatchClassifier
from src.models.loading import load_segmentation_model


_YX_RE = re.compile(r"_(\d+)_(\d+)_image$")

# Preregistered M5.6 Hough parameters; must match hough_postprocess.py defaults.
_HOUGH_KWARGS: dict[str, Any] = dict(
    hough_input_threshold=0.1,
    hough_threshold=50,
    min_line_length=100,
    max_line_gap=250,
    line_thickness=3,
)


@dataclass
class _PatchResult:
    clf_prob: float
    unet_probs: np.ndarray  # (H, W) float32
    is_positive: bool
    gt: np.ndarray          # (H, W) bool
    source_image: str
    patch_path: str


def _parse_yx(patch_path: str) -> tuple[int, int]:
    m = _YX_RE.search(Path(patch_path).stem)
    if m is None:
        raise ValueError(f"Cannot parse y,x from patch filename: {patch_path}")
    return int(m.group(1)), int(m.group(2))


def _load_unet(path: Path, device: torch.device) -> tuple[nn.Module, str]:
    return load_segmentation_model(path, device)


def _load_classifier(path: Path, device: torch.device) -> tuple[nn.Module, str]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = PatchClassifier()
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(device), cfg.get("normalisation", "full_image")


def _load_patch(row: dict[str, Any], normalisation: str) -> torch.Tensor:
    with Image.open(row["patch_path"]) as img:
        t = TF.pil_to_tensor(img.convert("L")).float() / 255.0
    return normalise_tensor(
        t,
        mode=normalisation,
        full_image_mean=row.get("image_mean"),
        full_image_std=row.get("image_std"),
    )


def _infer_all(
    rows: list[dict[str, Any]],
    classifier: nn.Module,
    unet: nn.Module,
    device: torch.device,
    normalisation: str,
) -> list[_PatchResult]:
    """Run both models on every patch and cache results for post-hoc threshold sweeps."""
    results: list[_PatchResult] = []
    with torch.no_grad():
        for row in rows:
            x = _load_patch(row, normalisation).unsqueeze(0).to(device)
            clf_prob = float(torch.sigmoid(classifier(x).squeeze()))
            unet_probs = torch.sigmoid(unet(x).squeeze()).cpu().numpy().astype(np.float32)
            with Image.open(row["mask_path"]) as m:
                gt = np.asarray(m.convert("L"), dtype=bool)
            results.append(_PatchResult(
                clf_prob=clf_prob,
                unet_probs=unet_probs,
                is_positive=float(row["positive_pixel_fraction"]) > 0,
                gt=gt,
                source_image=row["source_image"],
                patch_path=row["patch_path"],
            ))
    return results


def _prd(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    d = 2 * tp / max(2 * tp + fp + fn, 1)
    return round(p, 6), round(r, 6), round(d, 6)


def _compute_metrics(
    results: list[_PatchResult],
    clf_threshold: float,
    unet_threshold: float,
) -> dict[str, Any]:
    routed = np.array([r.clf_prob >= clf_threshold for r in results])
    labels = np.array([r.is_positive for r in results], dtype=bool)

    clf_tp = int((routed & labels).sum())
    clf_fp = int((routed & ~labels).sum())
    clf_fn = int((~routed & labels).sum())
    clf_prec = clf_tp / max(clf_tp + clf_fp, 1)
    clf_rec = clf_tp / max(clf_tp + clf_fn, 1)
    clf_f1 = 2 * clf_prec * clf_rec / max(clf_prec + clf_rec, 1e-9)

    r_tp = r_fp = r_fn = 0
    e_tp = e_fp = e_fn = 0
    for r, is_routed in zip(results, routed):
        if is_routed:
            pred = r.unet_probs >= unet_threshold
            tp = int((pred & r.gt).sum())
            fp = int((pred & ~r.gt).sum())
            fn = int((~pred & r.gt).sum())
            r_tp += tp; r_fp += fp; r_fn += fn
            e_tp += tp; e_fp += fp; e_fn += fn
        else:
            # Rejected patches: all GT pixels count as false negatives end-to-end.
            e_fn += int(r.gt.sum())

    r_p, r_r, r_d = _prd(r_tp, r_fp, r_fn)
    e_p, e_r, e_d = _prd(e_tp, e_fp, e_fn)

    return {
        "classifier_threshold": clf_threshold,
        "classifier_precision": round(clf_prec, 6),
        "classifier_recall": round(clf_rec, 6),
        "classifier_f1": round(clf_f1, 6),
        "classifier_fnr": round(clf_fn / max(clf_tp + clf_fn, 1), 6),
        "unet_precision_on_routed": r_p,
        "unet_recall_on_routed": r_r,
        "unet_dice_on_routed": r_d,
        "end_to_end_precision": e_p,
        "end_to_end_recall": e_r,
        "end_to_end_dice": e_d,
        "patches_routed_to_unet": int(routed.sum()),
        "patches_routed_fraction": round(float(routed.mean()), 6),
    }


def _hough_pass(
    results: list[_PatchResult],
    clf_threshold: float,
    unet_threshold: float,
) -> dict[str, Any]:
    by_image: dict[str, list[_PatchResult]] = {}
    for r in results:
        by_image.setdefault(r.source_image, []).append(r)

    detected_pre = detected_post = n_positive = 0
    for patches in by_image.values():
        yx = [_parse_yx(r.patch_path) for r in patches]
        h = max(y for y, _ in yx) + PATCH_SIZE
        w = max(x for _, x in yx) + PATCH_SIZE
        prob_canvas = np.zeros((h, w), dtype=np.float32)
        gt_canvas = np.zeros((h, w), dtype=np.uint8)
        for r, (y, x) in zip(patches, yx):
            if r.clf_prob >= clf_threshold:
                prob_canvas[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = np.maximum(
                    prob_canvas[y:y + PATCH_SIZE, x:x + PATCH_SIZE], r.unet_probs
                )
            gt_canvas[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = np.maximum(
                gt_canvas[y:y + PATCH_SIZE, x:x + PATCH_SIZE],
                r.gt.astype(np.uint8) * 255,
            )
        hr = run_hough_on_canvas(
            prob_canvas, gt_canvas, threshold=unet_threshold, **_HOUGH_KWARGS
        )
        if hr.gt_pixels > 0:
            n_positive += 1
            detected_pre += int(hr.detected_pre)
            detected_post += int(hr.detected_post)

    return {
        "hough_end_to_end_fnr_pre": round(1 - detected_pre / max(n_positive, 1), 6),
        "hough_end_to_end_fnr_post": round(1 - detected_post / max(n_positive, 1), 6),
        "hough_n_positive_images": n_positive,
    }


def _pareto_front(sweep: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pts = sorted(sweep, key=lambda d: d["end_to_end_recall"], reverse=True)
    front: list[dict[str, Any]] = []
    best_prec = -1.0
    for pt in pts:
        if pt["end_to_end_precision"] >= best_prec:
            front.append(pt)
            best_prec = pt["end_to_end_precision"]
    return front


def _joint_sweep(
    results: list[_PatchResult],
    clf_thresholds: np.ndarray,
    unet_thresholds: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Step 7: joint threshold sweep with cached U-Net inference."""
    sweep: list[dict[str, Any]] = []
    for t_clf in clf_thresholds:
        for t_unet in unet_thresholds:
            m = _compute_metrics(results, float(t_clf), float(t_unet))
            sweep.append({
                "classifier_threshold": float(t_clf),
                "unet_threshold": float(t_unet),
                "end_to_end_precision": m["end_to_end_precision"],
                "end_to_end_recall": m["end_to_end_recall"],
                "patches_routed_fraction": m["patches_routed_fraction"],
            })
    return sweep, _pareto_front(sweep)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classifier", required=True)
    p.add_argument("--clf-threshold", type=float, required=True)
    p.add_argument("--unet", required=True)
    p.add_argument("--unet-threshold", type=float, required=True)
    p.add_argument("--manifest", default="data/patches/manifest.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--baseline",
        default=None,
        help="Path to single-stage threshold sweep JSON for baseline_unet_* comparison fields.",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run joint (clf × unet) threshold sweep and compute Pareto front (Step 7).",
    )
    p.add_argument("--clf-thresholds-n", type=int, default=50)
    p.add_argument("--unet-thresholds-n", type=int, default=81)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(args.manifest)
    rows = manifest[manifest["split"] == args.split].to_dict("records")

    classifier, classifier_normalisation = _load_classifier(Path(args.classifier), device)
    unet, unet_normalisation = _load_unet(Path(args.unet), device)
    if classifier_normalisation != unet_normalisation:
        raise ValueError(
            "classifier and U-Net normalisation mismatch: "
            f"{classifier_normalisation!r} != {unet_normalisation!r}"
        )

    results = _infer_all(rows, classifier, unet, device, classifier_normalisation)

    output: dict[str, Any] = {
        "classifier_checkpoint": str(args.classifier),
        "unet_checkpoint": str(args.unet),
        "clf_threshold": args.clf_threshold,
        "unet_threshold": args.unet_threshold,
        "split": args.split,
        "normalisation": classifier_normalisation,
        "classifier_normalisation": classifier_normalisation,
        "unet_normalisation": unet_normalisation,
        **_compute_metrics(results, args.clf_threshold, args.unet_threshold),
        **_hough_pass(results, args.clf_threshold, args.unet_threshold),
    }

    if args.baseline is not None:
        with open(args.baseline) as f:
            bl = json.load(f)
        output["baseline_unet_precision"] = bl.get("test_precision")
        output["baseline_unet_recall"] = bl.get("test_recall")
        output["baseline_unet_dice"] = bl.get("test_dice")

    if args.sweep:
        clf_ts = np.linspace(0.01, 0.5, args.clf_thresholds_n)
        unet_ts = np.linspace(0.1, 0.9, args.unet_thresholds_n)
        sweep, pareto = _joint_sweep(results, clf_ts, unet_ts)
        output["sweep"] = sweep
        output["pareto_front"] = pareto

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
