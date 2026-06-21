#!/usr/bin/env python
"""Aggregate the three load-bearing locked-model diagnostics into 5-seed bands.

Converts the single-seed (s2804) diagnostics into mean +/- sample SD across the
five validation-locked seeds {7, 13, 42, 1234, 2804}, matching how strict
precision/recall are already reported across seeds. Inference-only: reads the
already-written per-seed diagnostic JSONs, computes no model output, and does
not retrain, re-tune, or reselect anything. Each seed contributes at its OWN
validation-optimal threshold (from threshold_sweep_winner_t44_s<seed>.json).

Diagnostics aggregated:
  1. fp_decomposition  -> inter_patch_fp_fraction              (target ~0.175)
  2. boundary_tolerant -> per_tolerance['1'].precision (+-1 px) (target ~0.946)
  3. geometry_eval     -> fp_distance.fraction_within_1px_all_fp(target ~0.57)

Writes results/classical/diagnostics_multiseed_summary.json.
"""

from __future__ import annotations

import json
import statistics
from datetime import date
from pathlib import Path

CLASS = Path("results/classical")
SEEDS = [7, 13, 42, 1234, 2804]  # validation-locked five-seed set
ANCHOR_SEED = 2804  # == model-best.pth; its single-seed values are the published ones


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _seed_threshold(seed: int) -> float | None:
    d = _load(CLASS / f"threshold_sweep_winner_t44_s{seed}.json")
    return None if d is None else d.get("optimal_threshold")


def _fp_decomp_value(seed: int):
    p = CLASS / f"fp_decomposition_unet_paper_arch_noise_topk_t44_s{seed}.json"
    d = _load(p)
    return (None, p) if d is None else (d.get("inter_patch_fp_fraction"), p)


def _boundary_value(seed: int):
    p = CLASS / f"boundary_tolerant_unet_paper_arch_noise_topk_t44_s{seed}.json"
    d = _load(p)
    if d is None:
        return None, p
    return d["per_tolerance"]["1"]["precision"], p


def _fp_distance_value(seed: int):
    p = CLASS / f"geometry_eval_t44_s{seed}.json"
    d = _load(p)
    if d is None:
        return None, p
    return d["fp_distance"]["fraction_within_1px_all_fp"], p


def _band(per_seed: list[dict], anchor: float | None) -> dict:
    vals = [r["value"] for r in per_seed if r["value"] is not None]
    n = len(vals)
    mean = statistics.fmean(vals) if n else None
    sd = statistics.stdev(vals) if n >= 2 else (0.0 if n == 1 else None)
    out = {
        "per_seed": per_seed,
        "n_seeds_used": n,
        "n_seeds_missing": len(per_seed) - n,
        "mean": None if mean is None else round(mean, 6),
        "sample_sd": None if sd is None else round(sd, 6),
        "anchor_seed": ANCHOR_SEED,
        "anchor_value": anchor,
    }
    if mean is not None and sd is not None and anchor is not None:
        out["anchor_within_band"] = bool(mean - sd <= anchor <= mean + sd)
    return out


def _collect(name: str, key_desc: str, value_fn, target: float) -> dict:
    per_seed = []
    anchor = None
    for s in SEEDS:
        val, src = value_fn(s)
        per_seed.append(
            {
                "seed": s,
                "threshold": _seed_threshold(s),
                "value": None if val is None else round(val, 6),
                "source_json": str(src),
                "available": val is not None,
            }
        )
        if s == ANCHOR_SEED and val is not None:
            anchor = round(val, 6)
    band = _band(per_seed, anchor)
    band["metric"] = key_desc
    band["target_single_seed"] = target
    return {name: band}


def main() -> None:
    summary = {
        "description": (
            "5-seed mean +/- sample SD bands for the three load-bearing "
            "locked-model diagnostics. Inference-only aggregation over already-"
            "written per-seed JSONs; no retraining/reselection. Each seed uses "
            "its own validation-optimal threshold."
        ),
        "seeds": SEEDS,
        "anchor_seed": ANCHOR_SEED,
        "anchor_note": (
            "anchor_seed (2804) == results/checkpoints/model-best.pth; its "
            "single-seed value is the one reported in the thesis. "
            "anchor_within_band = whether that value lies in [mean-SD, mean+SD]."
        ),
        "sd_convention": "sample standard deviation (n-1), matching restudy_topk_summary.json",
        "generated": str(date.today()),
        "diagnostics": {},
    }
    summary["diagnostics"].update(
        _collect(
            "fp_decomposition_inter_patch_fp_fraction",
            "fp_decomposition.inter_patch_fp_fraction",
            _fp_decomp_value,
            0.175,
        )
    )
    summary["diagnostics"].update(
        _collect(
            "boundary_tolerant_1px_precision",
            "boundary_tolerant.per_tolerance['1'].precision (+-1 px)",
            _boundary_value,
            0.946,
        )
    )
    summary["diagnostics"].update(
        _collect(
            "fp_distance_fraction_within_1px_all_fp",
            "geometry_eval.fp_distance.fraction_within_1px_all_fp",
            _fp_distance_value,
            0.57,
        )
    )

    out_path = CLASS / "diagnostics_multiseed_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    print(f"Wrote {out_path}")
    for name, band in summary["diagnostics"].items():
        print(
            f"  {name}: mean={band['mean']} sd={band['sample_sd']} "
            f"n={band['n_seeds_used']}/{len(SEEDS)} "
            f"anchor(s{ANCHOR_SEED})={band['anchor_value']} "
            f"within_band={band.get('anchor_within_band')}"
        )


if __name__ == "__main__":
    main()
