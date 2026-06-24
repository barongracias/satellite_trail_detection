#!/usr/bin/env python
"""Bootstrap CIs and interior/endpoint breakdown for the gold-audit boundary crops.

Pure re-analysis of results/classical/gold_audit_eval.json (no model, no GPU): an
equal-weight-per-crop bootstrap (seed 2804, B=10000, percentile 95% CIs) over the
SCORED BOUNDARY crops (stratum in {interior, endpoint}, scored=True; n=44). This
complements the pixel-weighted micro-aggregate already reported in the scored JSON
-- report both so they can be cross-checked.

Computes:
  1. parity_ci: per-crop Delta = f1(model_vs_reference) - f1(original_vs_reference)
     at 0/1 px (mean + 95% CI; CI spanning 0 => no clear evidence that the model
     agrees with the human reference worse than the original consortium labels do),
     plus the per-crop mean F1 for model_vs_reference and original_vs_reference separately.
  2. width_diff_ci: per-crop (reference - original) and (model - reference) median-
     width differences over boundary crops with all three widths present
     (CIs spanning 0 rule out systematic thinning / over-paint).
  3. stratum_breakdown: per stratum n, mean per-crop F1 (all three comparisons at
     0/1 px), median widths (original / reference / model).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IN_PATH = Path("results/classical/gold_audit_eval.json")
OUT_PATH = Path("results/classical/gold_audit_bootstrap.json")
SEED = 2804
B = 10000
BOUNDARY_STRATA = ("interior", "endpoint")
COMPARISONS = ("original_vs_reference", "model_vs_reference", "model_vs_original")
TOLERANCES = (0, 1)


def _f1(crop: dict, comparison: str, tol: int) -> float:
    return crop[comparison][f"tolerance_{tol}px"]["f1"]


def _ci95(samples: np.ndarray) -> list[float]:
    return [round(float(np.percentile(samples, 2.5)), 6), round(float(np.percentile(samples, 97.5)), 6)]


def _spans_zero(samples: np.ndarray) -> bool:
    return bool(np.percentile(samples, 2.5) <= 0.0 <= np.percentile(samples, 97.5))


def main() -> None:
    data = json.loads(IN_PATH.read_text())
    crops = [c for c in data["per_crop"] if c.get("scored") and c["stratum"] in BOUNDARY_STRATA]
    n = len(crops)
    rng = np.random.default_rng(SEED)

    # ---- 1. paired parity CI -------------------------------------------------
    parity: dict = {"n": n}
    for tol in TOLERANCES:
        mvr = np.array([_f1(c, "model_vs_reference", tol) for c in crops])
        ovr = np.array([_f1(c, "original_vs_reference", tol) for c in crops])
        delta = mvr - ovr
        idx = rng.integers(0, n, size=(B, n))  # shared resample keeps the three coherent
        boot_delta = delta[idx].mean(axis=1)
        boot_mvr = mvr[idx].mean(axis=1)
        boot_ovr = ovr[idx].mean(axis=1)
        parity[f"tolerance_{tol}px"] = {
            "mean_delta_model_minus_original": round(float(delta.mean()), 6),
            "delta_ci95": _ci95(boot_delta),
            "delta_ci_spans_zero": _spans_zero(boot_delta),
            "mean_f1_model_vs_reference": round(float(mvr.mean()), 6),
            "model_vs_reference_ci95": _ci95(boot_mvr),
            "mean_f1_original_vs_reference": round(float(ovr.mean()), 6),
            "original_vs_reference_ci95": _ci95(boot_ovr),
        }

    # ---- 2. width-difference CI ---------------------------------------------
    wcrops = [
        c for c in crops
        if all(c["width_median_px"][k] is not None for k in ("original", "reference", "model"))
    ]
    nw = len(wcrops)
    ref_minus_orig = np.array([c["width_median_px"]["reference"] - c["width_median_px"]["original"] for c in wcrops])
    model_minus_ref = np.array([c["width_median_px"]["model"] - c["width_median_px"]["reference"] for c in wcrops])
    idxw = rng.integers(0, nw, size=(B, nw))
    width_diff = {
        "n": nw,
        "reference_minus_original_px": {
            "mean": round(float(ref_minus_orig.mean()), 6),
            "ci95": _ci95(ref_minus_orig[idxw].mean(axis=1)),
            "ci_spans_zero": _spans_zero(ref_minus_orig[idxw].mean(axis=1)),
        },
        "model_minus_reference_px": {
            "mean": round(float(model_minus_ref.mean()), 6),
            "ci95": _ci95(model_minus_ref[idxw].mean(axis=1)),
            "ci_spans_zero": _spans_zero(model_minus_ref[idxw].mean(axis=1)),
        },
    }

    # ---- 3. interior vs endpoint breakdown ----------------------------------
    breakdown: dict = {}
    for stratum in BOUNDARY_STRATA:
        sc = [c for c in crops if c["stratum"] == stratum]
        block: dict = {"n": len(sc)}
        for comparison in COMPARISONS:
            for tol in TOLERANCES:
                block[f"mean_f1_{comparison}_{tol}px"] = round(
                    float(np.mean([_f1(c, comparison, tol) for c in sc])), 6
                )
        for key in ("original", "reference", "model"):
            vals = [c["width_median_px"][key] for c in sc if c["width_median_px"][key] is not None]
            block[f"median_width_{key}_px"] = round(float(np.median(vals)), 6) if vals else None
        breakdown[stratum] = block

    out = {
        "source": str(IN_PATH),
        "scope": "scored boundary crops (interior + endpoint), equal-weight per crop",
        "n_boundary_crops": n,
        "seed": SEED,
        "n_bootstrap": B,
        "ci": "95% percentile",
        "note": (
            "Equal-weight-per-crop bootstrap; complements (does not replace) the pixel-weighted "
            "micro-aggregate in gold_audit_eval.json (model_vs_reference 0.893 / "
            "original_vs_reference 0.890 strict F1)."
        ),
        "parity_ci": parity,
        "width_diff_ci": width_diff,
        "stratum_breakdown": breakdown,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, allow_nan=False))

    # ---- readable summary ----------------------------------------------------
    print(f"Wrote {OUT_PATH}  (n={n} boundary crops, B={B}, seed={SEED})\n")
    print("PARITY  Delta = f1(model_vs_ref) - f1(orig_vs_ref):")
    for tol in TOLERANCES:
        p = parity[f"tolerance_{tol}px"]
        print(f"  {tol}px: mean Delta {p['mean_delta_model_minus_original']:+.4f}  CI95 {p['delta_ci95']}  "
              f"spans0={p['delta_ci_spans_zero']}")
        print(f"        mean F1  model_vs_ref {p['mean_f1_model_vs_reference']:.4f} {p['model_vs_reference_ci95']} | "
              f"orig_vs_ref {p['mean_f1_original_vs_reference']:.4f} {p['original_vs_reference_ci95']}")
    print("\nWIDTH DIFF (px):")
    w = width_diff
    print(f"  reference - original: mean {w['reference_minus_original_px']['mean']:+.3f}  "
          f"CI95 {w['reference_minus_original_px']['ci95']}  spans0={w['reference_minus_original_px']['ci_spans_zero']}")
    print(f"  model - reference:    mean {w['model_minus_reference_px']['mean']:+.3f}  "
          f"CI95 {w['model_minus_reference_px']['ci95']}  spans0={w['model_minus_reference_px']['ci_spans_zero']}")
    print("\nSTRATUM BREAKDOWN:")
    for stratum, block in breakdown.items():
        print(f"  {stratum} (n={block['n']}): "
              f"mvr0 {block['mean_f1_model_vs_reference_0px']:.3f} ovr0 {block['mean_f1_original_vs_reference_0px']:.3f} "
              f"mvo0 {block['mean_f1_model_vs_original_0px']:.3f} | "
              f"width o/r/m {block['median_width_original_px']:.2f}/{block['median_width_reference_px']:.2f}/{block['median_width_model_px']:.2f}")


if __name__ == "__main__":
    main()
