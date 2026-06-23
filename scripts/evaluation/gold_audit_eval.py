#!/usr/bin/env python
"""Score the locked model against a blinded single-author reannotation.

The 64 new masks are an evaluation-only reference, not an independent gold
standard. Boundary crops are compared at strict and +/-1-pixel tolerance;
false-positive and decoy crops are reported separately as existence controls.
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
BOUNDARY_STRATA = ("interior", "endpoint")
CONTROL_STRATA = ("fp", "decoy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--verdicts_csv", required=True)
    parser.add_argument("--crop_dir", default="data/gold/audit_crops")
    parser.add_argument(
        "--sealed_manifest", default="data/gold/sealed_crop_manifest.json"
    )
    parser.add_argument("--checkpoint", default="results/checkpoints/model-best.pth")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--patch_dir", default="data/patches")
    parser.add_argument("--out", default="results/classical/gold_audit_eval.json")
    return parser.parse_args()


def load_annotation_mask(path: str | Path) -> np.ndarray:
    """Load a 0/255 PNG as a boolean mask and reject soft-edged exports."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    values = set(np.unique(array).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(
            f"{path}: annotation must be binary 0/255; found {sorted(values)}"
        )
    if values == {255}:
        raise ValueError(f"{path}: single-valued annotation must be all-background (0)")
    return array == 255


def score_mask_pair(
    reference: np.ndarray,
    other: np.ndarray,
    tolerances: tuple[int, ...] = TOLERANCES,
) -> dict[int, BoundaryTolerantCounts]:
    """Score ``other`` against ``reference`` at each pixel tolerance."""
    return {
        tolerance: boundary_tolerant_counts(other, reference, tolerance=tolerance)
        for tolerance in tolerances
    }


def width_median(mask: np.ndarray) -> float | None:
    """Estimate median structure width as twice skeleton-to-edge distance."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    if not mask.any():
        return None
    skeleton = skeletonize(mask)
    if not skeleton.any():
        return None
    return float(np.median(2.0 * distance_transform_edt(mask)[skeleton]))


def load_verdicts(path: str | Path) -> dict[str, str]:
    """Load one trail/no_trail/uncertain verdict for each crop."""
    allowed = {"trail", "no_trail", "uncertain"}
    verdicts: dict[str, str] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"crop_name", "verdict"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"{path}: expected crop_name,verdict columns")
        for row in reader:
            name = row["crop_name"].strip()
            verdict = row["verdict"].strip()
            if name in verdicts:
                raise ValueError(f"{path}: duplicate crop name {name}")
            if verdict not in allowed:
                raise ValueError(f"Unknown verdict {verdict!r} for {name}")
            verdicts[name] = verdict
    return verdicts


def _empty_totals() -> dict[int, BoundaryTolerantCounts]:
    return {tolerance: BoundaryTolerantCounts(0, 0, 0, 0) for tolerance in TOLERANCES}


def _metrics_block(totals: dict[int, BoundaryTolerantCounts]) -> dict[str, dict]:
    return {
        f"tolerance_{tolerance}px": boundary_tolerant_metrics(counts)
        for tolerance, counts in totals.items()
    }


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def classify_mechanism(
    original_reference_f1: float,
    model_original_f1: float,
    original_width: float | None,
    reference_width: float | None,
    model_width: float | None,
) -> str:
    """Apply the protocol's fixed diagnostic interpretation rules."""
    if any(value is None for value in (original_width, reference_width, model_width)):
        return "mixed/inconclusive"
    assert original_width is not None
    assert reference_width is not None
    assert model_width is not None
    if (
        original_reference_f1 <= model_original_f1 + 0.02
        and reference_width - original_width >= 1.0
        and abs(model_width - reference_width) <= 1.0
    ):
        return "thin-original support"
    if abs(reference_width - original_width) < 1.0 and model_width - reference_width >= 1.0:
        return "model-overpaint support"
    return "mixed/inconclusive"


def main() -> None:
    import pandas as pd

    from scripts.evaluation.hough_postprocess import _infer_batch, _load_normalised_patch
    from scripts.figures._locked_winner_canvases import load_raw_mask
    from src.models.loading import load_segmentation_model

    args = parse_args()
    logger = get_logger("gold_audit_eval")
    seed_everything()

    sealed = json.loads(Path(args.sealed_manifest).read_text())
    records = sealed["crops"]
    expected_names = [record["crop_name"] for record in records]
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("Sealed manifest contains duplicate crop names")

    annotation_dir = Path(args.annotation_dir)
    actual_names = sorted(path.name for path in annotation_dir.glob("*.png"))
    if actual_names != sorted(expected_names):
        raise ValueError("Annotation filenames do not exactly match the sealed manifest")
    verdicts = load_verdicts(args.verdicts_csv)
    if sorted(verdicts) != sorted(expected_names):
        raise ValueError("Verdict crop names do not exactly match the sealed manifest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalisation = load_segmentation_model(args.checkpoint, device)
    logger.info("Loaded %s with %s normalisation on %s", args.checkpoint, normalisation, device)

    patch_manifest = pd.read_csv(Path(args.patch_dir) / "manifest.csv")
    stats_by_image: dict[str, tuple[float, float]] = {}
    if {"image_mean", "image_std"}.issubset(patch_manifest.columns):
        for source, group in patch_manifest.groupby("source_image"):
            stats_by_image[str(source)] = (
                float(group["image_mean"].iloc[0]),
                float(group["image_std"].iloc[0]),
            )

    totals = {
        "original_vs_reference": _empty_totals(),
        "model_vs_reference": _empty_totals(),
        "model_vs_original": _empty_totals(),
    }
    widths: dict[str, list[float]] = {"original": [], "reference": [], "model": []}
    per_crop: list[dict] = []
    controls: list[dict] = []
    mask_cache: dict[str, np.ndarray] = {}
    n_uncertain = 0
    n_not_visible = 0

    for record in records:
        name = record["crop_name"]
        stratum = record["stratum"]
        verdict = verdicts[name]
        reference = load_annotation_mask(annotation_dir / name)
        if reference.shape != (CROP_SIZE, CROP_SIZE):
            raise ValueError(f"{name}: shape {reference.shape} != {(CROP_SIZE, CROP_SIZE)}")

        if verdict == "uncertain":
            n_uncertain += 1
            per_crop.append(
                {"crop_name": name, "stratum": stratum, "verdict": verdict, "scored": False}
            )
            continue
        if stratum in BOUNDARY_STRATA and not reference.any():
            n_not_visible += 1
            per_crop.append(
                {"crop_name": name, "stratum": stratum, "verdict": "not_visible", "scored": False}
            )
            continue

        source = record["source_image"]
        if source not in mask_cache:
            mask_cache[source] = load_raw_mask(source)
        y0, x0 = int(record["y0"]), int(record["x0"])
        original = mask_cache[source][y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]

        crop_path = Path(args.crop_dir) / name
        if not crop_path.exists():
            raise FileNotFoundError(f"Missing audit crop {crop_path}")
        mean, std = stats_by_image.get(source, (None, None))
        patch = _load_normalised_patch(str(crop_path), normalisation, mean, std)
        probability = _infer_batch([patch], model, device)[0]
        prediction = probability >= args.threshold

        row = {
            "crop_name": name,
            "stratum": stratum,
            "verdict": verdict,
            "scored": True,
            "width_median_px": {
                "original": width_median(original),
                "reference": width_median(reference),
                "model": width_median(prediction),
            },
        }
        comparisons = (
            ("original_vs_reference", reference, original),
            ("model_vs_reference", reference, prediction),
            ("model_vs_original", original, prediction),
        )
        for label, target, candidate in comparisons:
            counts = score_mask_pair(target, candidate)
            row[label] = _metrics_block(counts)
            if stratum in BOUNDARY_STRATA:
                for tolerance in TOLERANCES:
                    totals[label][tolerance] = totals[label][tolerance] + counts[tolerance]

        if stratum in BOUNDARY_STRATA:
            for label, value in row["width_median_px"].items():
                if value is not None:
                    widths[label].append(value)
        elif stratum in CONTROL_STRATA:
            controls.append(
                {
                    "crop_name": name,
                    "stratum": stratum,
                    "verdict": verdict,
                    "reference_has_trail": bool(reference.any()),
                    "model_fired": bool(prediction.any()),
                }
            )
        per_crop.append(row)

    aggregate = {label: _metrics_block(counts) for label, counts in totals.items()}
    width_summary = {label: _median_or_none(values) for label, values in widths.items()}
    mechanism = classify_mechanism(
        aggregate["original_vs_reference"]["tolerance_0px"]["f1"],
        aggregate["model_vs_original"]["tolerance_0px"]["f1"],
        width_summary["original"],
        width_summary["reference"],
        width_summary["model"],
    )

    def control_summary(stratum: str) -> dict[str, int]:
        rows = [row for row in controls if row["stratum"] == stratum]
        return {
            "n": len(rows),
            "verdict_trail": sum(row["verdict"] == "trail" for row in rows),
            "verdict_no_trail": sum(row["verdict"] == "no_trail" for row in rows),
            "reference_has_trail": sum(row["reference_has_trail"] for row in rows),
            "model_fired": sum(row["model_fired"] for row in rows),
        }

    output = {
        "schema_version": 2,
        "generated": str(date.today()),
        "annotation_design": "blinded single-author self-reannotation",
        "limitation": (
            "The reference is not an independent gold standard; no second annotator "
            "or inter-annotator agreement is available."
        ),
        "checkpoint": str(args.checkpoint),
        "sealed_manifest": str(args.sealed_manifest),
        "threshold": args.threshold,
        "tolerances_px": list(TOLERANCES),
        "n_crops": len(records),
        "n_scored_boundary": sum(
            row["scored"] and row["stratum"] in BOUNDARY_STRATA for row in per_crop
        ),
        "n_uncertain": n_uncertain,
        "n_not_visible": n_not_visible,
        "aggregate_scope": "interior and endpoint crops only",
        "aggregate": aggregate,
        "width_median_px": width_summary,
        "diagnostic_interpretation": mechanism,
        "existence_controls": {
            "fp": control_summary("fp"),
            "decoy": control_summary("decoy"),
        },
        "evaluation_only": (
            "No retraining, threshold reselection, or replacement of original masks."
        ),
        "per_crop": per_crop,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False))
    logger.info("Saved %s", output_path)
    print(f"Reannotation audit -> {output_path}")
    for label, block in aggregate.items():
        print(f"  {label}: {block}")
    print(f"  width_median_px: {width_summary}")
    print(f"  diagnostic_interpretation: {mechanism}")


if __name__ == "__main__":
    main()
