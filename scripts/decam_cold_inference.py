#!/usr/bin/env python
"""Qualitative cold-DECam inference demo for the locked MeerLICHT winner.

This script predeclares the nine RECA/NOIRLab DECam measured-streak detector
images, retrieves detector-HDU InstCal data, converts each image to the
MeerLICHT display-PNG domain, and applies the locked sampled-test winner. It is
an appendix-style visual illustration only: there are no masks, no benchmark
metrics, no threshold tuning, and no image replacement after inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config.constants import PATCH_SIZE

LOCKED_CHECKPOINT = Path("results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth")
LOCKED_THRESHOLD = 0.45
LOCKED_NORMALISATION = "full_image"
LOCKED_HOUGH_INPUT_THRESHOLD = 0.10
LOCKED_HOUGH_THRESHOLD = 50
LOCKED_MIN_LINE_LENGTH = 100
LOCKED_MAX_LINE_GAP = 250
LOCKED_LINE_THICKNESS = 3

DECAM_PIXEL_SCALE_ARCSEC = 0.2634
MEERLICHT_PIXEL_SCALE_ARCSEC = 0.56
RESAMPLE_FACTOR = DECAM_PIXEL_SCALE_ARCSEC / MEERLICHT_PIXEL_SCALE_ARCSEC
NOIRLAB_ROOT = "https://astroarchive.noirlab.edu"
RECA_SOURCE = "RECA notebooks/paper_notebooks/DECam_streaks_photometry-V2.ipynb cell 2 table"
NOIRLAB_ACKNOWLEDGEMENT = (
    "Based on observations at Cerro Tololo Inter-American Observatory, NSF's "
    "NOIRLab, managed by AURA under a cooperative agreement with the U.S. "
    "National Science Foundation."
)

VISUAL_FIELDS = (
    "visual_hit_unet_binary",
    "visual_hit_unet_prob",
    "visual_hit_hough",
    "visual_fp_clean",
    "human_review_status",
    "human_review_notes",
)


@dataclass(frozen=True)
class ManifestEntry:
    object_name: str
    norad_id: int
    expnum: int
    detector: int
    proc_type: str
    source_notebook_or_list: str
    expected_visual_class: str
    in_reca_measured_streak_sample: bool
    notes: str


@dataclass(frozen=True)
class ProcessedImage:
    record: dict[str, Any]
    image_u8: np.ndarray
    prob_canvas: np.ndarray
    binary_canvas: np.ndarray
    hough_canvas: np.ndarray


def build_manifest_entries() -> list[ManifestEntry]:
    """Return the predeclared nine-row RECA measured-streak detector manifest."""
    rows = [
        ("NAVSTAR-70", 39741, 1134933, 5, "table SNR 4509.6"),
        ("STARLINK-2600", 48387, 1138498, 23, "table SNR 301.9"),
        ("STARLINK-2559", 48298, 1033925, 17, "table SNR 39.5"),
        ("STARLINK-3758", 52297, 1103448, 4, "table SNR 128.8"),
        ("STARLINK-3772", 52290, 1103448, 20, "table SNR 295.4"),
        ("STARLINK-3771", 52293, 1103448, 45, "table SNR 1326.5"),
        ("STARLINK-3765", 52295, 1103448, 34, "table SNR 482.0"),
        ("DELTA-2 R/B", 20763, 1072590, 5, "table SNR 1906.6"),
        ("SORCE", 27651, 1125268, 21, "table SNR 35.3"),
    ]
    return [
        ManifestEntry(
            object_name=name,
            norad_id=norad,
            expnum=expnum,
            detector=detector,
            proc_type="instcal",
            source_notebook_or_list=RECA_SOURCE,
            expected_visual_class="streak",
            in_reca_measured_streak_sample=True,
            notes=notes,
        )
        for name, norad, expnum, detector, notes in rows
    ]


def visual_review_defaults() -> dict[str, None]:
    return {field: None for field in VISUAL_FIELDS}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def write_json(payload: dict[str, Any] | list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def write_manifest(entries: list[ManifestEntry], path: str | Path) -> None:
    payload = {
        "schema_version": "1.0",
        "analysis": "m61b_qualitative_cold_decam_manifest",
        "predeclared_before_inference": True,
        "source": RECA_SOURCE,
        "qualitative_only": True,
        "clean_controls_included": False,
        "entries": [asdict(entry) for entry in entries],
    }
    write_json(payload, path)


def noirlab_search(expnum: int, proc_type: str, timeout: float = 180.0) -> list[dict[str, Any]]:
    qspec = {
        "outfields": [
            "md5sum", "url", "archive_filename", "original_filename",
            "instrument", "proc_type", "EXPNUM", "prod_type",
        ],
        "search": [
            ["instrument", "decam"],
            ["proc_type", proc_type],
            ["EXPNUM", int(expnum), int(expnum)],
            ["prod_type", "image"],
        ],
    }
    url = f"{NOIRLAB_ROOT}/api/adv_search/find/?limit=20"
    import requests

    response = requests.post(url, json=qspec, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if len(payload) >= 2 and isinstance(payload[0], dict) and "HEADER" in payload[0]:
        return [row for row in payload[1:] if isinstance(row, dict)]
    if len(payload) >= 2 and isinstance(payload[0], list):
        columns = payload[0]
        return [dict(zip(columns, row)) for row in payload[1:]]
    raise RuntimeError(f"NOIRLab search returned no image rows for expnum={expnum}")


def retrieve_detector_hdu(
    entry: ManifestEntry,
    raw_dir: str | Path,
    timeout: float = 180.0,
) -> Path:
    """Retrieve a detector-HDU FITS file, caching raw data under an ignored path."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"decam_exp{entry.expnum}_det{entry.detector}_{entry.proc_type}.fits"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    import requests

    rows = noirlab_search(entry.expnum, entry.proc_type, timeout=timeout)
    if "md5sum" not in rows[0]:
        raise RuntimeError(f"NOIRLab search row lacks md5sum for expnum={entry.expnum}")
    base_url = rows[0].get("url") or f"{NOIRLAB_ROOT}/api/retrieve/{rows[0]['md5sum']}/"
    separator = "&" if "?" in base_url else "?"
    retrieve_url = f"{base_url}{separator}hdus={entry.detector}"
    response = requests.get(retrieve_url, timeout=timeout)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return out_path


def select_image_hdu(path: str | Path) -> tuple[np.ndarray, Any, int]:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        for idx, hdu in enumerate(hdul):
            data = hdu.data
            if data is not None and getattr(data, "ndim", 0) == 2:
                return np.asarray(data, dtype=np.float32), hdu.header.copy(), idx
    raise ValueError(f"No 2D image HDU found in {path}")


def header_pixel_scale_arcsec(header: Any) -> float | None:
    if "PIXSCAL" in header:
        try:
            return float(header["PIXSCAL"])
        except Exception:
            return None
    if all(key in header for key in ("CD1_1", "CD2_1", "CD1_2", "CD2_2")):
        x_scale = math.hypot(float(header["CD1_1"]), float(header["CD2_1"])) * 3600.0
        y_scale = math.hypot(float(header["CD1_2"]), float(header["CD2_2"])) * 3600.0
        return float((x_scale + y_scale) / 2.0)
    if "CDELT1" in header and "CDELT2" in header:
        return float((abs(float(header["CDELT1"])) + abs(float(header["CDELT2"]))) * 1800.0)
    return None


def clean_fits_image(image: np.ndarray) -> tuple[np.ndarray, int, float]:
    arr = np.asarray(image, dtype=np.float32).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("FITS image has no finite pixels")
    fill_value = float(np.median(arr[finite]))
    replaced = int((~finite).sum())
    arr[~finite] = fill_value
    return arr, replaced, fill_value


def downsample_linear_area(image: np.ndarray, factor: float = RESAMPLE_FACTOR) -> np.ndarray:
    import cv2

    if factor <= 0:
        raise ValueError("resample factor must be positive")
    h, w = image.shape
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def zscale_to_uint8(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from astropy.visualization import ZScaleInterval

    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("Cannot stretch an image with no finite pixels")
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image[finite])
    fallback = False
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.percentile(image[finite], [0.5, 99.5])
        fallback = True
    if vmax <= vmin:
        vmax = vmin + 1.0
        fallback = True
    scaled = np.clip((image - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    return (scaled * 255.0 + 0.5).astype(np.uint8), {
        "name": "zscale",
        "implementation": "astropy.visualization.ZScaleInterval",
        "vmin": float(vmin),
        "vmax": float(vmax),
        "fallback_percentile_used": bool(fallback),
    }


def reflect_pad_to_multiple(
    image: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> tuple[np.ndarray, dict[str, int], tuple[int, int]]:
    h, w = image.shape
    pad_h = (math.ceil(h / patch_size) * patch_size) - h
    pad_w = (math.ceil(w / patch_size) * patch_size) - w
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    return padded, {"bottom": int(pad_h), "right": int(pad_w)}, (int(h), int(w))


def iter_stride_tiles(image: np.ndarray, patch_size: int = PATCH_SIZE):
    h, w = image.shape
    if h % patch_size or w % patch_size:
        raise ValueError("iter_stride_tiles expects a padded image with patch-size multiple shape")
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            yield y, x, image[y : y + patch_size, x : x + patch_size]


def full_image_stats(image_u8: np.ndarray) -> dict[str, float]:
    image01 = image_u8.astype(np.float32) / 255.0
    return {"mean": float(image01.mean()), "std": float(image01.std())}


def normalise_uint8_patch_array(
    patch_u8: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Apply the locked full-image z-score to a uint8 patch as [1, H, W]."""
    patch = patch_u8.astype(np.float32) / 255.0
    return ((patch - float(mean)) / (float(std) + 1e-6))[None, :, :]


def infer_probability_canvas(
    image_u8: np.ndarray,
    model: Any,
    device: Any,
    batch_size: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    stats = full_image_stats(image_u8)
    padded, pad_shape, original_shape = reflect_pad_to_multiple(image_u8)
    prob_padded = np.zeros(padded.shape, dtype=np.float32)
    tiles = list(iter_stride_tiles(padded))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            batch_tiles = tiles[start : start + batch_size]
            batch_np = np.stack([
                normalise_uint8_patch_array(tile, stats["mean"], stats["std"])
                for _, _, tile in batch_tiles
            ])
            batch = torch.from_numpy(batch_np).to(device)
            probs = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()
            for (y, x, _), prob in zip(batch_tiles, probs):
                prob_padded[y : y + PATCH_SIZE, x : x + PATCH_SIZE] = prob.astype(np.float32)
    h, w = original_shape
    return prob_padded[:h, :w], {
        "normalisation": LOCKED_NORMALISATION,
        "full_image_mean": stats["mean"],
        "full_image_std": stats["std"],
        "pad_shape": pad_shape,
        "padded_shape": [int(v) for v in padded.shape],
        "n_patches": int(len(tiles)),
    }


def hough_overlay(prob_canvas: np.ndarray) -> np.ndarray:
    from src.classical.hough_runner import _apply_hough

    hough_input = (prob_canvas >= LOCKED_HOUGH_INPUT_THRESHOLD).astype(np.uint8) * 255
    return _apply_hough(
        hough_input,
        hough_threshold=LOCKED_HOUGH_THRESHOLD,
        min_line_length=LOCKED_MIN_LINE_LENGTH,
        max_line_gap=LOCKED_MAX_LINE_GAP,
        line_thickness=LOCKED_LINE_THICKNESS,
    ) > 0


def load_locked_model(device_arg: str | None = None) -> tuple[Any, Any]:
    import torch

    from src.models.loading import load_segmentation_model

    if device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalisation = load_segmentation_model(LOCKED_CHECKPOINT, device)
    if normalisation != LOCKED_NORMALISATION:
        raise ValueError(
            f"Locked checkpoint must use normalisation={LOCKED_NORMALISATION!r}; got {normalisation!r}"
        )
    return model, device


def process_entry(
    entry: ManifestEntry,
    model: Any,
    device: Any,
    raw_dir: str | Path,
    batch_size: int,
    save_input_pngs: bool = False,
    input_png_dir: str | Path = "results/figures/decam_cold_inputs",
) -> ProcessedImage:
    fits_path = retrieve_detector_hdu(entry, raw_dir)
    raw, header, hdu_index = select_image_hdu(fits_path)
    cleaned, nonfinite_replaced, fill_value = clean_fits_image(raw)
    downsampled = downsample_linear_area(cleaned)
    image_u8, stretch = zscale_to_uint8(downsampled)
    prob_canvas, inference = infer_probability_canvas(image_u8, model, device, batch_size=batch_size)
    binary_canvas = prob_canvas >= LOCKED_THRESHOLD
    hough_canvas = hough_overlay(prob_canvas)

    if save_input_pngs:
        png_dir = Path(input_png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        out_png = png_dir / f"decam_exp{entry.expnum}_det{entry.detector}_zscale.png"
        cv2.imwrite(str(out_png), image_u8)

    record = {
        **asdict(entry),
        "raw_fits_path": str(fits_path),
        "raw_fits_bytes": int(Path(fits_path).stat().st_size),
        "image_hdu_index": int(hdu_index),
        "raw_shape": [int(v) for v in raw.shape],
        "cleaned_nonfinite_pixels": int(nonfinite_replaced),
        "nonfinite_fill_value": float(fill_value),
        "pixscale_header_arcsec": header_pixel_scale_arcsec(header),
        "resample_factor": float(RESAMPLE_FACTOR),
        "resampling_note": "linear area downsampling with cv2.INTER_AREA; not integrated-flux preserving",
        "resampled_shape": [int(v) for v in image_u8.shape],
        "stretch_recipe": stretch,
        "threshold": float(LOCKED_THRESHOLD),
        "predicted_pixel_count": int(binary_canvas.sum()),
        "predicted_pixel_fraction": float(binary_canvas.mean()),
        "probability_mean": float(prob_canvas.mean()),
        "probability_max": float(prob_canvas.max()),
        "hough_pixel_count": int(hough_canvas.sum()),
        "hough_params": {
            "hough_input_threshold": LOCKED_HOUGH_INPUT_THRESHOLD,
            "hough_threshold": LOCKED_HOUGH_THRESHOLD,
            "min_line_length": LOCKED_MIN_LINE_LENGTH,
            "max_line_gap": LOCKED_MAX_LINE_GAP,
            "line_thickness": LOCKED_LINE_THICKNESS,
        },
        **inference,
        **visual_review_defaults(),
    }
    return ProcessedImage(
        record=record,
        image_u8=image_u8,
        prob_canvas=prob_canvas,
        binary_canvas=binary_canvas,
        hough_canvas=hough_canvas,
    )


def resize_for_figure(arr: np.ndarray, max_dim: int = 1400, nearest: bool = False) -> np.ndarray:
    import cv2

    h, w = arr.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return arr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(arr.astype(np.float32), (new_w, new_h), interpolation=interpolation)


def rotate_panel_for_display(arr: np.ndarray) -> np.ndarray:
    """Rotate portrait detector panels to landscape for a space-efficient montage."""
    return np.rot90(arr, k=1) if arr.shape[0] > arr.shape[1] else arr


def make_montage(processed: list[ProcessedImage], pdf_path: str | Path, svg_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    n = len(processed)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.0, 2.85 * nrows), constrained_layout=False)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, item in zip(axes_arr, processed):
        image = rotate_panel_for_display(resize_for_figure(item.image_u8, nearest=False))
        prob = rotate_panel_for_display(resize_for_figure(item.prob_canvas, nearest=False))
        binary = rotate_panel_for_display(
            resize_for_figure(item.binary_canvas.astype(np.uint8), nearest=True)
        ) > 0
        hough = rotate_panel_for_display(
            resize_for_figure(item.hough_canvas.astype(np.uint8), nearest=True)
        ) > 0
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        low_prob = prob >= LOCKED_HOUGH_INPUT_THRESHOLD
        if low_prob.any():
            ax.contour(
                low_prob.astype(float), levels=[0.5], colors=["#ffb000"],
                linewidths=0.45, alpha=0.9,
            )
        if binary.any():
            ax.contour(
                binary.astype(float), levels=[0.5], colors=["#ff2f92"],
                linewidths=0.75, alpha=0.95,
            )
        if hough.any():
            ax.contour(
                hough.astype(float), levels=[0.5], colors=["#00c8ff"],
                linewidths=0.55, linestyles="dashed", alpha=0.9,
            )
        rec = item.record
        ax.set_title(
            f"{rec['object_name']} | exp {rec['expnum']} det {rec['detector']}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[n:]:
        ax.axis("off")
    fig.subplots_adjust(left=0.006, right=0.996, top=0.875, bottom=0.083, hspace=0.26, wspace=0.012)
    fig.suptitle(
        "Locked MeerLICHT winner on predeclared RECA/NOIRLab DECam measured-streak images\n"
        "Qualitative cold-domain illustration only: probability contours, U-Net binary, dashed Hough overlay",
        fontsize=11,
        y=0.985,
    )
    fig.text(
        0.5, 0.025,
        "Panels rotated 90 deg for page layout. Contours: orange prob >= 0.10, "
        "magenta U-Net >= 0.45, cyan dashed Hough. No masks or benchmark metrics.",
        ha="center",
        fontsize=8,
    )
    for out in (Path(pdf_path), Path(svg_path)):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def build_result_payload(
    entries: list[ManifestEntry],
    processed: list[ProcessedImage],
    manifest_path: str | Path,
    partial_run: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "analysis": "m61b_qualitative_cold_decam_locked_winner",
        "qualitative_only": True,
        "benchmark": False,
        "selection_or_threshold_tuning": False,
        "manual_review_summary": None,
        "reporting_rule": (
            "After human review, report only X/N visible streak-containing images visually flagged; "
            "clean controls would be reported separately, but none are included here."
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "provenance": {
            "git_commit": git_commit(),
            "checkpoint": str(LOCKED_CHECKPOINT),
            "threshold": LOCKED_THRESHOLD,
            "normalisation": LOCKED_NORMALISATION,
            "predeclared_manifest_entries": len(entries),
            "processed_entries": len(processed),
            "partial_run": partial_run,
            "source": RECA_SOURCE,
            "reca_repository": "https://github.com/iausathub/reca-streaks",
            "reca_paper": "https://arxiv.org/abs/2603.10790",
            "noirlab_archive": NOIRLAB_ROOT,
            "noirlab_acknowledgement": NOIRLAB_ACKNOWLEDGEMENT,
            "montage_display_rotation_degrees": 90,
            "display_domain_caveat": (
                "DECam FITS data are linearly resampled and ZScale-stretched into an 8-bit "
                "display-like domain before applying the MeerLICHT PNG-trained model; this is "
                "a qualitative visual stress test, not calibrated cross-dataset performance."
            ),
        },
        "manifest": [asdict(entry) for entry in entries],
        "images": [item.record for item in processed],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-out", default="results/classical/decam_cold_manifest.json")
    parser.add_argument("--output-json", default="results/classical/decam_cold_inference.json")
    parser.add_argument("--figure-prefix", default="results/figures/decam_cold_inference_montage")
    parser.add_argument("--raw-dir", default="data/external/decam/raw")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-input-pngs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = build_manifest_entries()
    write_manifest(entries, args.manifest_out)
    if args.manifest_only:
        print(f"Wrote predeclared manifest: {args.manifest_out}")
        return

    selected = entries[: args.limit] if args.limit is not None else entries
    model, device = load_locked_model(args.device)
    processed = [
        process_entry(
            entry,
            model=model,
            device=device,
            raw_dir=args.raw_dir,
            batch_size=args.batch_size,
            save_input_pngs=args.save_input_pngs,
        )
        for entry in selected
    ]
    payload = build_result_payload(
        entries=entries,
        processed=processed,
        manifest_path=args.manifest_out,
        partial_run=len(processed) != len(entries),
    )
    write_json(payload, args.output_json)
    fig_prefix = Path(args.figure_prefix)
    make_montage(processed, fig_prefix.with_suffix(".pdf"), fig_prefix.with_suffix(".svg"))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {fig_prefix.with_suffix('.pdf')} and {fig_prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
