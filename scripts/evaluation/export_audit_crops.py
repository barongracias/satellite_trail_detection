#!/usr/bin/env python
"""Export blinded audit crops for the M9.4 gold-standard re-annotation audit.

Cuts ~64 crops of 528x528 raw-image pixels in four strata (sample spec
finalised 2026-06-12 in agents/PLAN.md):

  - INTERIOR  (~30): one crop per sampled cleaned GT component at a random
    along-trail position, ~equal counts per display-contrast tertile, from
    >= --min_distinct_images distinct test images.
  - ENDPOINT  (~15): centred on a randomly chosen endpoint of additional
    components (disjoint from the interior sample where possible).
  - FP        (~12): whole-patch false-positive patches — model fired
    (>=1 pixel at the locked threshold) on a patch whose mask is empty.
  - DECOY     (~7): genuinely empty patches — mask empty AND model quiet
    (no pixel at or above --quiet_threshold), shuffled in so the annotator
    cannot assume every crop contains something.

Blinding contract: the exported crops contain RAW image pixels only — no
masks, no predictions, ever — and are saved under neutral shuffled names
(c001.png, c002.png, ...). The crop -> frame-coordinate mapping is written to
a SEALED manifest the annotator must not open; a committable summary with
stratum counts (but no mapping) is written separately.

The locked pipeline is untouched: the checkpoint/threshold are used only to
classify negative patches as fired/quiet for the FP and decoy strata.

Usage (CSD3):
    CHECKPOINT=results/checkpoints/model-best.pth \
      sbatch slurm/export_audit_crops.sbatch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from PIL import Image

from src.models.loading import load_segmentation_model
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

Image.MAX_IMAGE_PIXELS = None

CROP_SIZE = 528


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--components_json",
                   default="results/classical/faint_streak_t44_s2804.json",
                   help="Locked faint-streak component JSON providing the cleaned "
                        "components and their contrast tertiles.")
    p.add_argument("--patch_dir", default="data/patches",
                   help="Sampled patch dir; its manifest supplies the negative "
                        "patches for the FP/decoy strata.")
    p.add_argument("--checkpoint", default="results/checkpoints/model-best.pth")
    p.add_argument("--threshold", type=float, default=0.45,
                   help="Locked segmentation threshold; a negative patch 'fires' "
                        "if any pixel reaches it.")
    p.add_argument("--quiet_threshold", type=float, default=0.1,
                   help="A negative patch is 'quiet' (decoy-eligible) if no pixel "
                        "reaches this; matches the locked Hough input threshold so "
                        "decoys are empty for both pipeline stages.")
    p.add_argument("--out_dir", default="data/gold/audit_crops")
    p.add_argument("--sealed_manifest", default="data/gold/sealed_crop_manifest.json",
                   help="Crop -> frame-coordinate mapping. The annotator must not "
                        "open this file. Lives under data/ (never committed).")
    p.add_argument("--public_summary",
                   default="results/classical/audit_crops_summary.json",
                   help="Committable summary: stratum counts, parameters, sealed-"
                        "manifest sha256 — no mapping.")
    p.add_argument("--seed", type=int, default=2804)
    p.add_argument("--n_interior", type=int, default=30)
    p.add_argument("--n_endpoint", type=int, default=15)
    p.add_argument("--n_fp", type=int, default=12)
    p.add_argument("--n_decoy", type=int, default=7)
    p.add_argument("--min_distinct_images", type=int, default=10)
    return p.parse_args()


def crop_window(centre_y: float, centre_x: float, shape: tuple[int, int],
                size: int = CROP_SIZE) -> tuple[int, int]:
    """Top-left (y0, x0) of a size x size crop centred on (centre_y, centre_x),
    clamped so the window lies fully inside the frame."""
    h, w = shape
    if h < size or w < size:
        raise ValueError(f"frame {shape} smaller than crop size {size}")
    y0 = int(round(centre_y)) - size // 2
    x0 = int(round(centre_x)) - size // 2
    return max(0, min(y0, h - size)), max(0, min(x0, w - size))


def sample_point_on_component(comp_mask: np.ndarray,
                              rng: np.random.Generator) -> tuple[int, int]:
    """A component pixel at a uniformly random along-trail (major-axis) position."""
    from scripts.figures.faint_streak_analysis import component_axis

    coords = np.argwhere(comp_mask)
    centre, axis, _ = component_axis(coords)
    proj = (coords - centre) @ axis
    t = rng.uniform(proj.min(), proj.max())
    return tuple(int(v) for v in coords[int(np.argmin(np.abs(proj - t)))])


def component_endpoints(comp_mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """The two extreme component pixels along the major axis."""
    from scripts.figures.faint_streak_analysis import component_axis

    coords = np.argwhere(comp_mask)
    centre, axis, _ = component_axis(coords)
    proj = (coords - centre) @ axis
    lo = coords[int(np.argmin(proj))]
    hi = coords[int(np.argmax(proj))]
    return (int(lo[0]), int(lo[1])), (int(hi[0]), int(hi[1]))


def stratified_component_sample(
    components: list[dict],
    n_interior: int,
    n_endpoint: int,
    min_distinct_images: int,
    rng: np.random.Generator,
    max_attempts: int = 200,
) -> tuple[list[dict], list[dict]]:
    """Sample interior components ~evenly over contrast tertiles plus a disjoint
    endpoint sample, deterministically retrying until the interior sample spans
    >= min_distinct_images distinct source images."""
    by_tier: dict[str, list[dict]] = {}
    for c in components:
        by_tier.setdefault(c["contrast_tier"], []).append(c)
    tiers = sorted(by_tier)
    base = n_interior // len(tiers)
    quota = {t: base for t in tiers}
    for t in tiers[: n_interior - base * len(tiers)]:
        quota[t] += 1

    for _ in range(max_attempts):
        interior: list[dict] = []
        shortfall = 0
        for t in tiers:
            pool = by_tier[t]
            take = min(quota[t], len(pool))
            shortfall += quota[t] - take
            idx = rng.choice(len(pool), size=take, replace=False)
            interior.extend(pool[i] for i in idx)
        if shortfall:
            # Tertiles are near-equal by construction; top up from the rest.
            chosen = {id(c) for c in interior}
            rest = [c for c in components if id(c) not in chosen]
            idx = rng.choice(len(rest), size=min(shortfall, len(rest)), replace=False)
            interior.extend(rest[i] for i in idx)
        n_images = len({c["source_image"] for c in interior})
        if n_images >= min_distinct_images:
            break
    else:
        raise RuntimeError(
            f"Could not reach {min_distinct_images} distinct images in "
            f"{max_attempts} attempts (got {n_images})"
        )

    chosen = {(c["source_image"], c["component_index"]) for c in interior}
    remaining = [c for c in components
                 if (c["source_image"], c["component_index"]) not in chosen]
    n_end = min(n_endpoint, len(remaining))
    idx = rng.choice(len(remaining), size=n_end, replace=False)
    endpoint = [remaining[i] for i in idx]
    return interior, endpoint


def classify_negative_patches(
    neg_df,
    model: torch.nn.Module,
    normalisation: str,
    device: torch.device,
    threshold: float,
    quiet_threshold: float,
) -> tuple[list, list]:
    """Split manifest rows of mask-empty patches into (fired, quiet) by the
    locked model's per-patch max probability."""
    from scripts.evaluation.hough_postprocess import (
        _HOUGH_MAX_BATCH,
        _chunks,
        _infer_batch,
        _load_normalised_patch,
    )

    has_stats = {"image_mean", "image_std"}.issubset(neg_df.columns)
    fired, quiet = [], []
    rows = list(neg_df.itertuples(index=False))
    for chunk in _chunks(rows, _HOUGH_MAX_BATCH):
        patches = []
        for row in chunk:
            mean = getattr(row, "image_mean", None) if has_stats else None
            std = getattr(row, "image_std", None) if has_stats else None
            patches.append(_load_normalised_patch(row.patch_path, normalisation, mean, std))
        probs = _infer_batch(patches, model, device)
        for row, prob in zip(chunk, probs):
            peak = float(prob.max())
            if peak >= threshold:
                fired.append(row)
            elif peak < quiet_threshold:
                quiet.append(row)
    return fired, quiet


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import pandas as pd

    from scripts.evaluation.hough_postprocess import _parse_yx
    from scripts.figures._locked_winner_canvases import load_raw_image, load_raw_mask
    from scripts.figures.faint_streak_analysis import cleaned_gt_components

    args = parse_args()
    logger = get_logger("export_audit_crops")
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    comp_doc = json.loads(Path(args.components_json).read_text())
    params = comp_doc["parameters"]
    components = [c for c in comp_doc["components"] if not c.get("invalid_contrast")]
    logger.info("Components available: %d (from %s)", len(components), args.components_json)

    interior, endpoint = stratified_component_sample(
        components, args.n_interior, args.n_endpoint,
        args.min_distinct_images, rng,
    )
    logger.info("Sampled %d interior + %d endpoint components over %d images",
                len(interior), len(endpoint),
                len({c["source_image"] for c in interior}))

    # Re-derive component pixel masks with the exact extraction used by the
    # locked faint-streak analysis, and sanity-check the component indexing.
    needed_images = sorted({c["source_image"] for c in interior + endpoint})
    comp_masks: dict[tuple[str, int], np.ndarray] = {}
    raw_cache: dict[str, np.ndarray] = {}
    for src in needed_images:
        gt = load_raw_mask(src)
        comps, _ = cleaned_gt_components(
            gt,
            close_kernel=int(params["close_kernel"]),
            min_area=int(params["min_component_area"]),
            min_major_axis=float(params["min_major_axis"]),
        )
        expected = [c["component_index"] for c in comp_doc["components"]
                    if c["source_image"] == src]
        if max(expected) > len(comps):
            raise RuntimeError(
                f"Component indexing mismatch for {src}: JSON has index "
                f"{max(expected)} but re-derivation found {len(comps)} components"
            )
        for idx, comp in enumerate(comps, start=1):
            comp_masks[(src, idx)] = comp["mask"]
        raw_cache[src] = load_raw_image(src)

    # FP / decoy strata need the locked model's verdict on mask-empty patches.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalisation = load_segmentation_model(args.checkpoint, device)
    logger.info("Loaded %s (normalisation=%s) on %s", args.checkpoint, normalisation, device)
    manifest = pd.read_csv(Path(args.patch_dir) / "manifest.csv")
    neg_df = manifest[(manifest["split"] == "test")
                      & (manifest["positive_pixel_fraction"] <= 0)].reset_index(drop=True)
    logger.info("Mask-empty test patches: %d", len(neg_df))
    fired, quiet = classify_negative_patches(
        neg_df, model, normalisation, device, args.threshold, args.quiet_threshold,
    )
    logger.info("Negative patches fired/quiet: %d/%d", len(fired), len(quiet))
    if len(fired) < args.n_fp or len(quiet) < args.n_decoy:
        raise RuntimeError(
            f"Not enough FP ({len(fired)}>={args.n_fp}?) or decoy "
            f"({len(quiet)}>={args.n_decoy}?) candidates"
        )
    fp_rows = [fired[i] for i in rng.choice(len(fired), size=args.n_fp, replace=False)]
    decoy_rows = [quiet[i] for i in rng.choice(len(quiet), size=args.n_decoy, replace=False)]

    # Assemble crop records (frame coordinates + provenance), then shuffle.
    records: list[dict] = []
    for c in interior:
        mask = comp_masks[(c["source_image"], c["component_index"])]
        cy, cx = sample_point_on_component(mask, rng)
        y0, x0 = crop_window(cy, cx, mask.shape)
        records.append({
            "stratum": "interior",
            "source_image": c["source_image"],
            "component_index": c["component_index"],
            "contrast_tier": c["contrast_tier"],
            "centre_yx": [cy, cx], "y0": y0, "x0": x0,
        })
    for c in endpoint:
        mask = comp_masks[(c["source_image"], c["component_index"])]
        ends = component_endpoints(mask)
        cy, cx = ends[int(rng.integers(0, 2))]
        y0, x0 = crop_window(cy, cx, mask.shape)
        records.append({
            "stratum": "endpoint",
            "source_image": c["source_image"],
            "component_index": c["component_index"],
            "contrast_tier": c["contrast_tier"],
            "centre_yx": [cy, cx], "y0": y0, "x0": x0,
        })
    for stratum, rows in (("fp", fp_rows), ("decoy", decoy_rows)):
        for row in rows:
            y, x = _parse_yx(row.patch_path)
            records.append({
                "stratum": stratum,
                "source_image": str(row.source_image),
                "patch_path": str(row.patch_path),
                "y0": int(y), "x0": int(x),
            })

    order = rng.permutation(len(records))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for new_idx, rec_idx in enumerate(order, start=1):
        rec = records[int(rec_idx)]
        src = rec["source_image"]
        if src not in raw_cache:
            raw_cache[src] = load_raw_image(src)
        raw = raw_cache[src]
        y0, x0 = rec["y0"], rec["x0"]
        crop = raw[y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]
        if crop.shape != (CROP_SIZE, CROP_SIZE):
            raise RuntimeError(f"Bad crop shape {crop.shape} for {rec}")
        name = f"c{new_idx:03d}.png"
        Image.fromarray(crop, mode="L").save(out_dir / name)
        rec["crop_name"] = name

    sealed = {
        "blinding_note": (
            "SEALED: maps neutral crop names to frame coordinates and strata. "
            "The annotator must not open this file before annotation is complete."
        ),
        "crop_size": CROP_SIZE,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "quiet_threshold": args.quiet_threshold,
        "components_json": str(args.components_json),
        "patch_dir": str(args.patch_dir),
        "crops": sorted(records, key=lambda r: r["crop_name"]),
        "generated": str(date.today()),
    }
    sealed_path = Path(args.sealed_manifest)
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_path.write_text(json.dumps(sealed, indent=2, allow_nan=False))

    counts = {s: sum(1 for r in records if r["stratum"] == s)
              for s in ("interior", "endpoint", "fp", "decoy")}
    summary = {
        "n_crops": len(records),
        "stratum_counts": counts,
        "n_distinct_images_interior": len({c["source_image"] for c in interior}),
        "crop_size": CROP_SIZE,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "quiet_threshold": args.quiet_threshold,
        "fired_candidates": len(fired),
        "quiet_candidates": len(quiet),
        "components_json": str(args.components_json),
        "sealed_manifest_sha256": _sha256(sealed_path),
        "blinding_note": (
            "Crops contain raw 8-bit display pixels only; masks and predictions "
            "are never exported. The crop-to-frame mapping is sealed under "
            "data/gold/ (uncommitted); this summary intentionally has no mapping."
        ),
        "generated": str(date.today()),
    }
    summary_path = Path(args.public_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    logger.info("Wrote %d crops to %s", len(records), out_dir)
    logger.info("Sealed manifest: %s | public summary: %s", sealed_path, summary_path)
    print(f"Exported {len(records)} crops: {counts}")
    print(f"Sealed manifest (DO NOT OPEN as annotator): {sealed_path}")
    print(f"Public summary: {summary_path}")


if __name__ == "__main__":
    main()
