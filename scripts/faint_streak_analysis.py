#!/usr/bin/env python
"""Relative display-contrast diagnostic for the locked M5.6 winner.

Post-hoc analysis only: re-scores the locked U-Net + Hough prediction on the
sampled test split. Raw intensities are 8-bit display PNG values, so reported
contrast is a relative display-space proxy, not calibrated flux or SNR.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.stats import ks_2samp

from scripts._locked_winner_canvases import (
    LOCKED_CHECKPOINT,
    LOCKED_THRESHOLD,
    iter_positive_source_groups,
    load_locked_model,
    provenance,
    read_test_manifest,
    reconstruct_locked_canvases,
    write_json,
)
from scripts.make_thesis_figures import COLORS, configure_style, save_vector
from src.config.constants import GLOBAL_SEED
from src.evaluation.segmentation import SegmentationCounts, compute_metrics_from_counts

CLASSICAL = Path("results/classical")
FIGURES = Path("results/figures")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=LOCKED_CHECKPOINT)
    p.add_argument("--patch-dir", default="data/patches")
    p.add_argument("--threshold", type=float, default=LOCKED_THRESHOLD)
    p.add_argument("--out", default=str(CLASSICAL / "faint_streak_t44_s2804.json"))
    p.add_argument("--tag", default="t44_s2804")
    p.add_argument("--max-images", type=int, default=None,
                   help="Optional smoke-test cap on positive source images.")
    p.add_argument("--min-component-area", type=int, default=25)
    p.add_argument("--min-major-axis", type=float, default=25.0)
    p.add_argument("--close-kernel", type=int, default=5)
    p.add_argument("--crop-margin", type=int, default=32)
    p.add_argument("--profile-half-width", type=int, default=24)
    p.add_argument("--profile-step", type=float, default=16.0)
    p.add_argument("--sideband-start", type=float, default=12.0)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--fp-rider-near-px", type=float, default=2.0)
    p.add_argument("--fp-rider-far-px", type=float, default=5.0)
    p.add_argument("--fp-rider-subsample", type=int, default=30000)
    p.add_argument("--skip-fp-rider", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()


def _safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _bbox_from_mask(mask: np.ndarray, margin: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return slice(0, 0), slice(0, 0)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, shape[1])
    return slice(y0, y1), slice(x0, x1)


def component_axis(coords_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return centre, unit major-axis vector in (y, x), and projected length."""
    pts = coords_yx.astype(float)
    centre = pts.mean(axis=0)
    centred = pts - centre
    if len(pts) < 2:
        return centre, np.array([1.0, 0.0]), 0.0
    cov = np.cov(centred, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm else np.array([1.0, 0.0])
    proj = centred @ axis
    return centre, axis, float(proj.max() - proj.min()) if proj.size else 0.0


def cleaned_gt_components(
    gt_mask: np.ndarray,
    *,
    close_kernel: int,
    min_area: int,
    min_major_axis: float,
) -> tuple[list[dict[str, Any]], int]:
    """Group GT components with light morphology, but return original-GT pixels."""
    gt_u8 = gt_mask.astype(np.uint8)
    n_raw, _ = cv2.connectedComponents(gt_u8, connectivity=8)
    grouped = gt_u8
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        grouped = cv2.morphologyEx(gt_u8, cv2.MORPH_CLOSE, kernel)
    n_labels, labels = cv2.connectedComponents(grouped.astype(np.uint8), connectivity=8)
    components: list[dict[str, Any]] = []
    for label in range(1, n_labels):
        grouping_mask = labels == label
        original = np.logical_and(gt_mask, grouping_mask)
        area = int(original.sum())
        if area < min_area:
            continue
        coords = np.argwhere(original)
        centre, axis, major_length = component_axis(coords)
        if major_length < min_major_axis:
            continue
        components.append({
            "label": int(label),
            "mask": original,
            "area": area,
            "centre_yx": centre,
            "axis_yx": axis,
            "major_axis_length_px": major_length,
        })
    return components, int(n_raw - 1)


def sample_component_profile(
    raw_image: np.ndarray,
    component_mask: np.ndarray,
    centre_yx: np.ndarray,
    axis_yx: np.ndarray,
    *,
    half_width: int,
    step: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    coords = np.argwhere(component_mask)
    if coords.size == 0:
        return np.arange(-half_width, half_width + 1, dtype=float), np.array([]), 0
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    normal = np.array([-axis_yx[1], axis_yx[0]], dtype=float)
    proj = (coords.astype(float) - centre_yx) @ axis_yx
    lo, hi = np.percentile(proj, [5, 95]) if len(proj) > 4 else (proj.min(), proj.max())
    if hi <= lo:
        sample_ts = np.array([(lo + hi) / 2.0])
    else:
        sample_ts = np.arange(lo, hi + 1e-6, step)
        if sample_ts.size == 0:
            sample_ts = np.array([(lo + hi) / 2.0])
    profiles = []
    raw_float = raw_image.astype(float)
    for t in sample_ts:
        point = centre_yx + axis_yx * t
        ys = point[0] + normal[0] * offsets
        xs = point[1] + normal[1] * offsets
        prof = map_coordinates(raw_float, [ys, xs], order=1, mode="nearest")
        profiles.append(prof)
    if not profiles:
        return offsets, np.array([]), 0
    return offsets, np.median(np.vstack(profiles), axis=0), len(profiles)


def profile_metrics(
    offsets: np.ndarray,
    profile: np.ndarray,
    *,
    sideband_start: float,
) -> dict[str, Any]:
    if profile.size == 0:
        return {
            "fit_status": "no_profile",
            "relative_display_contrast": None,
            "peak_minus_background": None,
            "fwhm_px": None,
            "integrated_excess": None,
            "sideband_scatter": None,
            "saturated": False,
        }
    side = np.abs(offsets) >= sideband_start
    if not np.any(side):
        side = np.ones_like(offsets, dtype=bool)
    side_values = profile[side]
    background = float(np.median(side_values))
    mad = float(np.median(np.abs(side_values - background)))
    scatter = 1.4826 * mad
    excess = profile - background
    peak_idx = int(np.argmax(excess))
    peak_excess = float(max(excess[peak_idx], 0.0))
    peak_raw = float(profile[peak_idx])
    eps = 1e-6
    contrast = None if scatter <= eps else peak_excess / scatter
    invalid = scatter <= eps
    positive = np.clip(excess, 0.0, None)
    integrated = float(np.trapezoid(positive, offsets))
    fwhm = None
    if peak_excess > 0:
        above = offsets[excess >= peak_excess / 2.0]
        if above.size >= 2:
            fwhm = float(above.max() - above.min())
        elif above.size == 1:
            fwhm = 0.0
    return {
        "fit_status": "nonparametric",
        "relative_display_contrast": _safe_float(contrast),
        "peak_minus_background": _safe_float(peak_excess),
        "fwhm_px": _safe_float(fwhm),
        "integrated_excess": _safe_float(integrated),
        "background": _safe_float(background),
        "sideband_scatter": _safe_float(scatter),
        "invalid_contrast": bool(invalid),
        "saturated": bool(peak_raw >= 250.0),
        "peak_raw_display_value": _safe_float(peak_raw),
    }


def component_pixel_metrics(
    component_mask: np.ndarray,
    binary_canvas: np.ndarray,
    hough_canvas: np.ndarray,
    support_canvas: np.ndarray,
    *,
    crop_margin: int,
) -> dict[str, Any]:
    supported_gt = np.logical_and(component_mask, support_canvas)
    gt_pixels = int(supported_gt.sum())
    if gt_pixels == 0:
        return {
            "support_fraction": 0.0,
            "gt_pixels_supported": 0,
            "recall_pre_hough": None,
            "recall_post_hough": None,
            "hough_added_gt_px": 0,
            "hough_added_gt_fraction": None,
            "dice_crop_pre_hough": None,
            "dice_crop_post_hough": None,
        }
    combined = np.logical_or(binary_canvas, hough_canvas)
    tp_pre = int(np.logical_and(binary_canvas, supported_gt).sum())
    tp_post = int(np.logical_and(combined, supported_gt).sum())
    hough_added = int(np.logical_and.reduce((hough_canvas, ~binary_canvas, supported_gt)).sum())
    sl_y, sl_x = _bbox_from_mask(supported_gt, crop_margin, component_mask.shape)
    crop_support = support_canvas[sl_y, sl_x]
    crop_gt = np.logical_and(component_mask[sl_y, sl_x], crop_support)

    def _dice(pred: np.ndarray) -> float | None:
        pred_crop = np.logical_and(pred[sl_y, sl_x], crop_support)
        counts = SegmentationCounts(
            true_positive=int(np.logical_and(pred_crop, crop_gt).sum()),
            false_positive=int(np.logical_and(pred_crop, ~crop_gt).sum()),
            true_negative=int(np.logical_and(~pred_crop, ~crop_gt).sum()),
            false_negative=int(np.logical_and(~pred_crop, crop_gt).sum()),
        )
        return compute_metrics_from_counts(counts).dice

    return {
        "support_fraction": _safe_float(gt_pixels / int(component_mask.sum())),
        "gt_pixels_supported": gt_pixels,
        "recall_pre_hough": _safe_float(tp_pre / gt_pixels),
        "recall_post_hough": _safe_float(tp_post / gt_pixels),
        "hough_added_gt_px": hough_added,
        "hough_added_gt_fraction": _safe_float(hough_added / gt_pixels),
        "dice_crop_pre_hough": _safe_float(_dice(binary_canvas)),
        "dice_crop_post_hough": _safe_float(_dice(combined)),
    }


def _tier_label(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "low"
    if value <= q2:
        return "medium"
    return "high"


def cluster_bootstrap_mean(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    seed: int,
    n_resamples: int,
) -> tuple[float | None, float | None, float | None]:
    values = [r[metric] for r in rows if r.get(metric) is not None]
    if not values:
        return None, None, None
    mean = float(np.mean(values))
    sources = sorted({r["source_image"] for r in rows if r.get(metric) is not None})
    if len(sources) < 2:
        return mean, None, None
    by_source = {s: [r for r in rows if r["source_image"] == s and r.get(metric) is not None]
                 for s in sources}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        drawn_sources = rng.choice(sources, size=len(sources), replace=True)
        drawn_values = []
        for src in drawn_sources:
            drawn_values.extend(r[metric] for r in by_source[src])
        samples.append(float(np.mean(drawn_values)))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return mean, float(lo), float(hi)


def build_tier_summary(
    component_rows: list[dict[str, Any]],
    *,
    n_resamples: int,
) -> tuple[list[dict[str, Any]], str]:
    valid = [r for r in component_rows if r.get("relative_display_contrast") is not None]
    if len(valid) < 15 or len({r["source_image"] for r in valid}) < 3:
        return [], "Too few valid components/sources for stable contrast tiers; use scatter/descriptive summaries."
    vals = np.asarray([r["relative_display_contrast"] for r in valid], dtype=float)
    q1, q2 = np.percentile(vals, [33.333, 66.667])
    for r in valid:
        r["contrast_tier"] = _tier_label(float(r["relative_display_contrast"]), float(q1), float(q2))
    summary = []
    for idx, tier in enumerate(("low", "medium", "high")):
        rows = [r for r in valid if r["contrast_tier"] == tier]
        out = {
            "tier": tier,
            "n_components": len(rows),
            "n_source_images": len({r["source_image"] for r in rows}),
            "contrast_min": _safe_float(min(r["relative_display_contrast"] for r in rows)),
            "contrast_max": _safe_float(max(r["relative_display_contrast"] for r in rows)),
        }
        for metric in ("recall_pre_hough", "recall_post_hough", "hough_added_gt_fraction"):
            mean, lo, hi = cluster_bootstrap_mean(
                rows, metric, seed=GLOBAL_SEED + idx, n_resamples=n_resamples,
            )
            out[f"{metric}_mean"] = _safe_float(mean)
            out[f"{metric}_ci_lo"] = _safe_float(lo)
            out[f"{metric}_ci_hi"] = _safe_float(hi)
        summary.append(out)
    return summary, "Tertiles are descriptive post-hoc bins over relative display contrast."


FP_RIDER_CATEGORIES = ("near_gt_fp", "far_fp", "gt_trail", "background")


def fp_distance_strata(
    target_canvas: np.ndarray,
    binary_canvas: np.ndarray,
    support_canvas: np.ndarray,
    *,
    near_px: float,
    far_px: float,
) -> dict[str, np.ndarray]:
    """Return support-restricted U-Net FP masks stratified by distance to supported GT."""
    target_supported = np.logical_and(target_canvas, support_canvas)
    fp_mask = np.logical_and.reduce((binary_canvas, ~target_supported, support_canvas))
    distances = distance_transform_edt(~target_supported)
    near = np.logical_and(fp_mask, distances <= near_px)
    far = np.logical_and(fp_mask, distances > far_px)
    ambiguous = np.logical_and.reduce((fp_mask, distances > near_px, distances <= far_px))
    return {
        "target_supported": target_supported,
        "fp_mask": fp_mask,
        "near_gt_fp": near,
        "far_fp": far,
        "ambiguous_fp": ambiguous,
        "distances": distances,
    }


def _sample_1d(values: np.ndarray, *, limit: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or limit <= 0:
        return np.array([], dtype=float)
    if values.size <= limit:
        return values.astype(float, copy=True)
    idx = rng.choice(values.size, size=limit, replace=False)
    return values[idx].astype(float, copy=True)


def collect_fp_intensity_samples(
    raw_image: np.ndarray,
    target_canvas: np.ndarray,
    binary_canvas: np.ndarray,
    support_canvas: np.ndarray,
    *,
    near_px: float,
    far_px: float,
    per_category_limit: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect per-image display-excess samples for the support-restricted FP rider."""
    strata = fp_distance_strata(
        target_canvas, binary_canvas, support_canvas, near_px=near_px, far_px=far_px,
    )
    target_supported = strata["target_supported"]
    background_mask = np.logical_and.reduce((support_canvas, ~target_supported, ~binary_canvas))
    background_fallback = False
    background_source = background_mask
    if not np.any(background_source):
        background_source = np.logical_and(support_canvas, ~target_supported)
        background_fallback = True
    if not np.any(background_source):
        background_source = support_canvas
        background_fallback = True
    raw_float = raw_image.astype(float)
    background_median = float(np.median(raw_float[background_source])) if np.any(background_source) else 0.0
    display_excess = raw_float - background_median
    masks = {
        "near_gt_fp": strata["near_gt_fp"],
        "far_fp": strata["far_fp"],
        "gt_trail": target_supported,
        "background": background_mask,
    }
    samples = {
        name: _sample_1d(display_excess[mask], limit=per_category_limit, rng=rng)
        for name, mask in masks.items()
    }
    record = {
        "background_median_display": _safe_float(background_median),
        "background_fallback": bool(background_fallback),
        "ambiguous_fp_px_available": int(strata["ambiguous_fp"].sum()),
    }
    for name, mask in masks.items():
        record[f"{name}_px_available"] = int(mask.sum())
        record[f"{name}_px_sampled"] = int(samples[name].size)
    return samples, record


def _category_summary(values: np.ndarray, n_available: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    out = {
        "n_available": int(n_available),
        "n_sampled": int(values.size),
        "median": None,
        "iqr_lo": None,
        "iqr_hi": None,
    }
    if values.size:
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        out.update({
            "median": _safe_float(q50),
            "iqr_lo": _safe_float(q25),
            "iqr_hi": _safe_float(q75),
        })
    return out


def _ks_d(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) == 0 or len(right) == 0:
        return None
    return _safe_float(ks_2samp(left, right).statistic)


def build_fp_intensity_rider(
    pooled_samples: dict[str, list[np.ndarray]],
    per_image_records: list[dict[str, Any]],
    *,
    near_px: float,
    far_px: float,
    subsample: int,
    per_image_quota: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Summarise the FP-intensity rider without storing sampled arrays in JSON."""
    final_samples: dict[str, np.ndarray] = {}
    categories: dict[str, Any] = {}
    for name in FP_RIDER_CATEGORIES:
        values = np.concatenate(pooled_samples[name]) if pooled_samples[name] else np.array([], dtype=float)
        values = _sample_1d(values, limit=subsample, rng=rng)
        final_samples[name] = values
        n_available = sum(int(r.get(f"{name}_px_available", 0)) for r in per_image_records)
        categories[name] = _category_summary(values, n_available)
    summary = {
        "analysis": "support_restricted_fp_intensity_rider",
        "method_note": (
            "Display excess is raw 8-bit PNG intensity minus a per-image sampled-support "
            "background median. Values are relative display-space proxies, not calibrated flux/SNR."
        ),
        "interpretation_note": (
            "Small KS-D for near-GT FP vs GT trail is consistent with boundary under-capture; "
            "small KS-D for far FP vs background is consistent with background confusion. "
            "These are descriptive post-hoc diagnostics, not proof of label error or hallucination."
        ),
        "parameters": {
            "near_px": _safe_float(near_px),
            "far_px": _safe_float(far_px),
            "subsample_per_category": int(subsample),
            "per_image_category_quota": int(per_image_quota),
            "ambiguous_distance_band_dropped": f"{near_px:g} < d <= {far_px:g}",
        },
        "categories": categories,
        "ks_d_near_fp_vs_gt_trail": _ks_d(final_samples["near_gt_fp"], final_samples["gt_trail"]),
        "ks_d_far_fp_vs_background": _ks_d(final_samples["far_fp"], final_samples["background"]),
        "n_images": len(per_image_records),
        "n_background_fallback_images": sum(bool(r.get("background_fallback")) for r in per_image_records),
        "per_image": per_image_records,
    }
    return summary, final_samples


def make_contrast_figure(component_rows: list[dict[str, Any]], tier_summary: list[dict[str, Any]]) -> None:
    rows = [r for r in component_rows if r.get("relative_display_contrast") is not None]
    if not rows:
        return
    x = np.asarray([r["relative_display_contrast"] for r in rows], dtype=float)
    pre = np.asarray([r["recall_pre_hough"] for r in rows], dtype=float)
    post = np.asarray([r["recall_post_hough"] for r in rows], dtype=float)
    added = np.asarray([r["hough_added_gt_fraction"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    ax.scatter(x, pre, s=14, alpha=0.48, color=COLORS["blue"], label="U-Net")
    ax.scatter(x, post, s=14, alpha=0.58, color=COLORS["orange"], label="U-Net + Hough")
    ax.set_xlabel("Relative display contrast proxy")
    ax.set_ylabel("Component GT-pixel recall")
    ax.set_title("Recall vs relative\ndisplay contrast")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=True, framealpha=0.86, edgecolor="none", loc="lower right")

    ax = axes[1]
    ax.scatter(x, added, s=16, alpha=0.55, color=COLORS["green"])
    ax.set_xlabel("Relative display contrast proxy")
    ax.set_ylabel("Hough-added GT fraction")
    ax.set_title("Post-hoc Hough\ncompletion")
    ax.set_ylim(-0.02, max(0.05, float(np.nanmax(added)) + 0.02))

    fig.subplots_adjust(bottom=0.16, top=0.84, wspace=0.48)
    save_vector(fig, "faint_streak_contrast_vs_recall")

def make_profile_figure(component_rows: list[dict[str, Any]]) -> None:
    candidates = [r for r in component_rows if r.get("relative_display_contrast") is not None
                  and r.get("profile_offsets") and r.get("median_profile")]
    if len(candidates) < 2:
        return
    candidates = sorted(candidates, key=lambda r: r["relative_display_contrast"])
    chosen = [candidates[0], candidates[-1]]
    labels = ["low contrast\nexample", "high contrast\nexample"]
    colours = [COLORS["blue"], COLORS["orange"]]
    fig, ax = plt.subplots(figsize=(4.7, 3.1))
    for row, label, color in zip(chosen, labels, colours):
        offsets = np.asarray(row["profile_offsets"], dtype=float)
        profile = np.asarray(row["median_profile"], dtype=float)
        bg = row.get("background") or 0.0
        ax.plot(
            offsets, profile - bg, color=color, lw=1.2,
            label=f"{label}\nC={row['relative_display_contrast']:.2f}",
        )
    ax.axhline(0.0, color="#555555", lw=0.7)
    ax.set_xlabel("Perpendicular offset from GT component axis (px)")
    ax.set_ylabel("Display value above local background")
    ax.set_title("Example median cross-sections")
    ax.legend(frameon=True, framealpha=0.86, edgecolor="none", loc="upper right", fontsize=6.5)
    fig.subplots_adjust(bottom=0.16, top=0.88)
    save_vector(fig, "faint_streak_profiles")

def make_fp_intensity_figure(
    samples: dict[str, np.ndarray] | None,
    rider: dict[str, Any] | None,
) -> None:
    if not samples or not rider:
        return
    nonempty = [np.asarray(v, dtype=float) for v in samples.values() if len(v)]
    if not nonempty:
        return
    pooled = np.concatenate(nonempty)
    if pooled.size < 2:
        return
    lo, hi = np.percentile(pooled, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(pooled)), float(np.max(pooled))
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 36)
    labels = {
        "near_gt_fp": "Near-GT FP",
        "far_fp": "Far FP",
        "gt_trail": "GT trail",
        "background": "Background",
    }
    colours = {
        "near_gt_fp": COLORS["orange"],
        "far_fp": COLORS["purple"],
        "gt_trail": COLORS["green"],
        "background": COLORS["blue"],
    }
    fig, ax = plt.subplots(figsize=(5.4, 3.25))
    for name in ("gt_trail", "near_gt_fp", "background", "far_fp"):
        vals = np.asarray(samples.get(name, np.array([])), dtype=float)
        if vals.size == 0:
            continue
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.2,
                color=colours[name], label=f"{labels[name]} (n={vals.size})")
    ax.set_xlabel("Display value above per-image background median")
    ax.set_ylabel("Normalised density")
    ax.set_title("False-positive pixel intensity vs background")
    ks_near = rider.get("ks_d_near_fp_vs_gt_trail")
    ks_far = rider.get("ks_d_far_fp_vs_background")
    note = "Display-space proxy from 8-bit PNGs; descriptive locked-winner diagnostic, no p-values."
    if ks_near is not None or ks_far is not None:
        bits = []
        if ks_near is not None:
            bits.append(f"KS-D near FP vs GT = {ks_near:.2f}")
        if ks_far is not None:
            bits.append(f"KS-D far FP vs background = {ks_far:.2f}")
        note = " ; ".join(bits) + "\n" + note
    ax.legend(frameon=True, framealpha=0.86, edgecolor="none", fontsize=6.7, loc="upper left")
    fig.subplots_adjust(bottom=0.16, top=0.88)
    save_vector(fig, "fp_intensity")

def analyse(args: argparse.Namespace) -> dict[str, Any]:
    model, normalisation, device = load_locked_model(args.checkpoint)
    test_df = read_test_manifest(args.patch_dir)
    component_rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    fp_rider_records: list[dict[str, Any]] = []
    fp_sample_pool: dict[str, list[np.ndarray]] = {name: [] for name in FP_RIDER_CATEGORIES}
    rng = np.random.default_rng(GLOBAL_SEED)
    groups = list(iter_positive_source_groups(test_df))
    if args.max_images is not None:
        groups = groups[: args.max_images]
    per_image_fp_quota = int(math.ceil(args.fp_rider_subsample / max(len(groups), 1)))
    for source_idx, (source_image, group) in enumerate(groups, start=1):
        print(f"[{source_idx}/{len(groups)}] {Path(source_image).name}", flush=True)
        canvases = reconstruct_locked_canvases(
            source_image, group, model, device, normalisation, threshold=args.threshold,
        )
        components, raw_component_count = cleaned_gt_components(
            canvases.target_canvas,
            close_kernel=args.close_kernel,
            min_area=args.min_component_area,
            min_major_axis=args.min_major_axis,
        )
        image_record = {
            "source_image": source_image,
            "raw_component_count": raw_component_count,
            "cleaned_component_count": len(components),
            "support_fraction_gt": _safe_float(
                np.logical_and(canvases.target_canvas, canvases.support_canvas).sum()
                / max(int(canvases.target_canvas.sum()), 1)
            ),
        }
        per_image.append(image_record)
        if not args.skip_fp_rider:
            samples, rider_record = collect_fp_intensity_samples(
                canvases.raw_image,
                canvases.target_canvas,
                canvases.binary_canvas,
                canvases.support_canvas,
                near_px=args.fp_rider_near_px,
                far_px=args.fp_rider_far_px,
                per_category_limit=per_image_fp_quota,
                rng=rng,
            )
            rider_record["source_image"] = source_image
            fp_rider_records.append(rider_record)
            for name, values in samples.items():
                fp_sample_pool[name].append(values)
        for comp_idx, comp in enumerate(components, start=1):
            offsets, profile, n_profiles = sample_component_profile(
                canvases.raw_image,
                np.logical_and(comp["mask"], canvases.support_canvas),
                comp["centre_yx"],
                comp["axis_yx"],
                half_width=args.profile_half_width,
                step=args.profile_step,
            )
            prof = profile_metrics(offsets, profile, sideband_start=args.sideband_start)
            pix = component_pixel_metrics(
                comp["mask"], canvases.binary_canvas, canvases.hough_canvas,
                canvases.support_canvas, crop_margin=args.crop_margin,
            )
            row = {
                "source_image": source_image,
                "component_index": comp_idx,
                "area_px": comp["area"],
                "major_axis_length_px": _safe_float(comp["major_axis_length_px"]),
                "n_profiles": n_profiles,
                **prof,
                **pix,
            }
            if profile.size:
                row["profile_offsets"] = [float(v) for v in offsets]
                row["median_profile"] = [float(v) for v in profile]
            component_rows.append(row)
    tier_summary, tier_note = build_tier_summary(component_rows, n_resamples=args.n_bootstrap)
    fp_rider = None
    fp_rider_samples = None
    if not args.skip_fp_rider:
        fp_rider, fp_rider_samples = build_fp_intensity_rider(
            fp_sample_pool,
            fp_rider_records,
            near_px=args.fp_rider_near_px,
            far_px=args.fp_rider_far_px,
            subsample=args.fp_rider_subsample,
            per_image_quota=per_image_fp_quota,
            rng=rng,
        )
    payload = {
        "analysis": "relative_display_contrast_faint_streak_mvp",
        "provenance": provenance(
            checkpoint=args.checkpoint,
            threshold=args.threshold,
            normalisation=normalisation,
            patch_dir=args.patch_dir,
        ),
        "method_note": (
            "Post-hoc locked-winner diagnostic. Relative display contrast is computed from "
            "8-bit display PNG cross-sections and is not calibrated flux/SNR."
        ),
        "support_note": (
            "Prediction/FP/background denominators are restricted to sampled-test support; "
            "component trail recall is reported over supported GT pixels."
        ),
        "parameters": {
            "min_component_area": args.min_component_area,
            "min_major_axis": args.min_major_axis,
            "close_kernel": args.close_kernel,
            "crop_margin": args.crop_margin,
            "profile_half_width": args.profile_half_width,
            "profile_step": args.profile_step,
            "sideband_start": args.sideband_start,
            "fp_rider_near_px": args.fp_rider_near_px,
            "fp_rider_far_px": args.fp_rider_far_px,
            "fp_rider_subsample": args.fp_rider_subsample,
            "skip_fp_rider": args.skip_fp_rider,
        },
        "n_positive_images_processed": len(groups),
        "n_components": len(component_rows),
        "n_components_with_valid_contrast": sum(r.get("relative_display_contrast") is not None for r in component_rows),
        "per_image": per_image,
        "tier_summary": tier_summary,
        "tier_note": tier_note,
        "components": component_rows,
    }
    if fp_rider is not None:
        payload["fp_intensity_rider"] = fp_rider
        payload["_fp_intensity_rider_samples"] = fp_rider_samples
    return payload


def main() -> None:
    args = parse_args()
    configure_style()
    payload = analyse(args)
    fp_rider_samples = payload.pop("_fp_intensity_rider_samples", None)
    write_json(args.out, payload)
    if not args.skip_figures:
        make_contrast_figure(payload["components"], payload["tier_summary"])
        make_profile_figure(payload["components"])
        make_fp_intensity_figure(fp_rider_samples, payload.get("fp_intensity_rider"))
    print(f"Saved {args.out}")
    print(f"Components: {payload['n_components']} | valid contrast: {payload['n_components_with_valid_contrast']}")


if __name__ == "__main__":
    main()
