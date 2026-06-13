#!/usr/bin/env python
"""Score the M9.4 gold-standard re-annotation audit (SCAFFOLD).

Runs for real only after annotation is complete. Ingests the annotator's
binary PNG masks (drawn on the blinded crops exported by
export_audit_crops.py), maps them back to frame coordinates via the sealed
manifest, and computes the preregistered analyses:

  (a) original-vs-gold strict and +/-1 px P/R/F1   — the label-noise floor;
  (b) model-vs-gold (same tolerances) on the same crops, next to
      model-vs-original — does the detector agree with careful annotation
      better than the original labels do;
  (c) per-component width medians (original mask, gold mask, prediction) —
      the thin-labels vs over-paint discriminator.

Annotation contract: one PNG per crop, same name (c001.png ...), same
528x528 grid, containing at most two distinct pixel values (background and
trail; an all-background PNG has one value, which must be 0). FP/decoy
verdicts ("trail" / "no_trail" / "uncertain") go in a separate CSV
(--verdicts_csv, columns crop_name,verdict); "uncertain" crops are reported
separately and never scored.

Gold masks are evaluation-only: no retraining, no threshold re-selection,
no replacement of original masks. Everything here is post-hoc measurement
of the locked pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from PIL import Image

from src.evaluation.segmentation import (
    BoundaryTolerantCounts,
    boundary_tolerant_counts,
    boundary_tolerant_metrics,
)
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

CROP_SIZE = 528
TOLERANCES = (0, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotation_dir", required=True,
                   help="Directory of the annotator's binary PNGs (c001.png ...).")
    p.add_argument("--sealed_manifest", default="data/gold/sealed_crop_manifest.json")
    p.add_argument("--verdicts_csv", default=None,
                   help="CSV with crop_name,verdict for FP/decoy crops "
                        "(trail / no_trail / uncertain).")
    p.add_argument("--checkpoint", default="results/checkpoints/model-best.pth")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--patch_dir", default="data/patches",
                   help="Manifest source for per-image normalisation stats.")
    p.add_argument("--out", default="results/classical/gold_audit_eval.json")
    return p.parse_args()


def load_annotation_mask(path: str | Path) -> np.ndarray:
    """Load one annotation PNG as a boolean mask.

    Enforces the annotation contract: at most two distinct pixel values; the
    higher value is trail. A single-valued PNG must be all-zero (empty
    annotation). Raises ValueError otherwise (e.g. anti-aliased brushes or
    accidental greyscale exports).
    """
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
    values = np.unique(arr)
    if len(values) > 2:
        raise ValueError(
            f"{path}: annotation must be binary, found {len(values)} distinct "
            f"pixel values {values[:10].tolist()} — re-export without anti-aliasing"
        )
    if len(values) == 1:
        if int(values[0]) != 0:
            raise ValueError(
                f"{path}: single-valued annotation must be all-background (0), "
                f"found constant value {int(values[0])}"
            )
        return np.zeros_like(arr, dtype=bool)
    return arr == values.max()


def score_mask_pair(
    reference: np.ndarray,
    other: np.ndarray,
    tolerances: tuple[int, ...] = TOLERANCES,
) -> dict[int, BoundaryTolerantCounts]:
    """Boundary-tolerant counts of `other` scored against `reference` at each
    tolerance (0 = strict). Counts are micro-aggregatable by addition."""
    return {t: boundary_tolerant_counts(other, reference, tolerance=t)
            for t in tolerances}


def width_median(mask: np.ndarray) -> float | None:
    """Median perpendicular width of a thin structure: 2x the Euclidean
    distance-to-background sampled on the skeleton. None for empty masks."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    if not mask.any():
        return None
    skel = skeletonize(mask)
    if not skel.any():
        return None
    return float(np.median(2.0 * distance_transform_edt(mask)[skel]))


def load_verdicts(path: str | Path) -> dict[str, str]:
    """crop_name -> verdict (trail / no_trail / uncertain)."""
    allowed = {"trail", "no_trail", "uncertain"}
    out: dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            verdict = row["verdict"].strip()
            if verdict not in allowed:
                raise ValueError(f"Unknown verdict {verdict!r} for {row['crop_name']}")
            out[row["crop_name"].strip()] = verdict
    return out


def _metrics_block(totals: dict[int, BoundaryTolerantCounts]) -> dict:
    return {f"tolerance_{t}px": boundary_tolerant_metrics(c)
            for t, c in totals.items()}


def main() -> None:
    import pandas as pd

    from scripts.evaluation.hough_postprocess import _infer_batch, _load_normalised_patch
    from scripts.figures._locked_winner_canvases import load_raw_mask

    args = parse_args()
    logger = get_logger("gold_audit_eval")
    seed_everything()

    sealed = json.loads(Path(args.sealed_manifest).read_text())
    annotation_dir = Path(args.annotation_dir)
    verdicts = load_verdicts(args.verdicts_csv) if args.verdicts_csv else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.models.loading import load_segmentation_model
    model, normalisation = load_segmentation_model(args.checkpoint, device)
    logger.info("Loaded %s (normalisation=%s) on %s", args.checkpoint, normalisation, device)

    manifest = pd.read_csv(Path(args.patch_dir) / "manifest.csv")
    stats_by_image: dict[str, tuple[float, float]] = {}
    if {"image_mean", "image_std"}.issubset(manifest.columns):
        for src, grp in manifest.groupby("source_image"):
            stats_by_image[str(src)] = (
                float(grp["image_mean"].iloc[0]), float(grp["image_std"].iloc[0])
            )

    # Boundary aggregates are restricted to interior/endpoint crops (the
    # label-noise / width analysis). FP/decoy crops are existence controls and
    # are summarised separately, never mixed into these totals.
    totals_orig_vs_gold = {t: BoundaryTolerantCounts(0, 0, 0, 0) for t in TOLERANCES}
    totals_model_vs_gold = {t: BoundaryTolerantCounts(0, 0, 0, 0) for t in TOLERANCES}
    totals_model_vs_orig = {t: BoundaryTolerantCounts(0, 0, 0, 0) for t in TOLERANCES}
    BOUNDARY_STRATA = ("interior", "endpoint")
    existence: list[dict] = []  # fp/decoy: verdict vs model-fired vs gold-has-trail
    per_crop: list[dict] = []
    n_uncertain = 0
    n_not_visible = 0
    mask_cache: dict[str, np.ndarray] = {}

    for rec in sealed["crops"]:
        name = rec["crop_name"]
        stratum = rec["stratum"]
        ann_path = annotation_dir / name
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation for {name} in {annotation_dir}")
        gold = load_annotation_mask(ann_path)
        if gold.shape != (CROP_SIZE, CROP_SIZE):
            raise ValueError(f"{name}: annotation shape {gold.shape} != crop grid")

        if stratum in ("fp", "decoy") and verdicts.get(name) == "uncertain":
            n_uncertain += 1
            per_crop.append({"crop_name": name, "stratum": stratum,
                             "verdict": "uncertain", "scored": False})
            continue
        # Interior/endpoint crops where the gold annotator saw nothing are a
        # label-error category, excluded from boundary statistics.
        if stratum in ("interior", "endpoint") and not gold.any():
            n_not_visible += 1
            per_crop.append({"crop_name": name, "stratum": stratum,
                             "verdict": "not_visible", "scored": False})
            continue

        src = rec["source_image"]
        y0, x0 = rec["y0"], rec["x0"]
        if src not in mask_cache:
            mask_cache[src] = load_raw_mask(src)
        original = mask_cache[src][y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]

        # Locked-model prediction on exactly this crop window. Crop windows are
        # patch-aligned for fp/decoy strata but arbitrary for interior/endpoint;
        # full_image normalisation stats come from the manifest per source image.
        mean, std = stats_by_image.get(src, (None, None))
        crop_png = Path(args.annotation_dir).parent / "audit_crops" / name
        crop_source = crop_png if crop_png.exists() else None
        if crop_source is None:
            raise FileNotFoundError(
                f"Raw crop {name} not found next to annotations; pass the "
                f"audit_crops directory exported by export_audit_crops.py"
            )
        patch = _load_normalised_patch(str(crop_source), normalisation, mean, std)
        prob = _infer_batch([patch], model, device)[0]
        pred = prob >= args.threshold

        row = {
            "crop_name": name, "stratum": stratum, "scored": True,
            "original_vs_gold": {}, "model_vs_gold": {}, "model_vs_original": {},
            "width_median_px": {
                "original": width_median(original),
                "gold": width_median(gold),
                "prediction": width_median(pred),
            },
        }
        is_boundary = stratum in BOUNDARY_STRATA
        for label, ref, other, totals in (
            ("original_vs_gold", gold, original, totals_orig_vs_gold),
            ("model_vs_gold", gold, pred, totals_model_vs_gold),
            ("model_vs_original", original, pred, totals_model_vs_orig),
        ):
            counts = score_mask_pair(ref, other)
            # Only interior/endpoint crops feed the boundary/label-noise
            # aggregate; fp/decoy per-crop blocks are still recorded below.
            if is_boundary:
                for t in TOLERANCES:
                    totals[t] = totals[t] + counts[t]
            row[label] = _metrics_block(counts)
        if not is_boundary:
            existence.append({
                "crop_name": name, "stratum": stratum,
                "verdict": verdicts.get(name),
                "gold_has_trail": bool(gold.any()),
                "model_fired": bool(pred.any()),
            })
        per_crop.append(row)

    def _existence_summary(stratum: str) -> dict:
        rows = [e for e in existence if e["stratum"] == stratum]
        return {
            "n": len(rows),
            "verdict_trail": sum(1 for e in rows if e["verdict"] == "trail"),
            "verdict_no_trail": sum(1 for e in rows if e["verdict"] == "no_trail"),
            "gold_has_trail": sum(1 for e in rows if e["gold_has_trail"]),
            "model_fired": sum(1 for e in rows if e["model_fired"]),
        }

    out = {
        "sealed_manifest": str(args.sealed_manifest),
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "tolerances_px": list(TOLERANCES),
        "n_crops": len(sealed["crops"]),
        "n_scored": sum(1 for r in per_crop if r.get("scored")),
        "n_uncertain": n_uncertain,
        "n_not_visible": n_not_visible,
        "aggregate_scope": "interior+endpoint crops only (boundary/label-noise)",
        "aggregate": {
            "original_vs_gold": _metrics_block(totals_orig_vs_gold),
            "model_vs_gold": _metrics_block(totals_model_vs_gold),
            "model_vs_original": _metrics_block(totals_model_vs_orig),
        },
        "existence_controls": {
            "fp": _existence_summary("fp"),
            "decoy": _existence_summary("decoy"),
            "note": (
                "FP/decoy crops are existence controls, excluded from the "
                "boundary aggregate. decoy verdict_trail > 0 flags a "
                "trigger-happy annotator; fp gold_has_trail counts originals "
                "that missed a real trail the gold annotator confirmed."
            ),
        },
        "evaluation_only_note": (
            "Gold masks are evaluation-only derivatives of MeerLICHT data: "
            "no retraining, no threshold re-selection, no mask replacement. "
            "Stored under data/gold/, never committed."
        ),
        "scaffold_status": (
            "INCOMPLETE: reports point aggregates and existence controls only. "
            "Still to implement before the real run (protocol analyses b/d): "
            "component-level bootstrap intervals, and inter-/intra-annotator "
            "agreement for NSD tau calibration (requires the second annotator's "
            "masks). Point estimates here are not the full preregistered output."
        ),
        "per_crop": per_crop,
        "generated": str(date.today()),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False))
    logger.info("Saved to %s", out_path)
    print(f"Gold audit eval -> {out_path}")
    for label in ("original_vs_gold", "model_vs_gold", "model_vs_original"):
        print(f"  {label}: {out['aggregate'][label]}")


if __name__ == "__main__":
    main()
