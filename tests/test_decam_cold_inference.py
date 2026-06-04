"""Tests for the qualitative cold-DECam inference helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.decam_cold_inference import (  # noqa: E402
    DECAM_PIXEL_SCALE_ARCSEC,
    MEERLICHT_PIXEL_SCALE_ARCSEC,
    RESAMPLE_FACTOR,
    MANUAL_REVIEW_FIELDS,
    VISUAL_FIELDS,
    apply_manual_review,
    build_manifest_entries,
    full_image_stats,
    iter_stride_tiles,
    normalise_uint8_patch_array,
    reflect_pad_to_multiple,
    review_template_payload,
    visual_review_defaults,
)


def test_resample_factor_matches_decam_to_meerlicht_pixel_scale() -> None:
    assert RESAMPLE_FACTOR == pytest.approx(DECAM_PIXEL_SCALE_ARCSEC / MEERLICHT_PIXEL_SCALE_ARCSEC)
    assert RESAMPLE_FACTOR == pytest.approx(0.47035714285714286)


def test_reflect_pad_stride_tiles_crop_back_to_original_shape() -> None:
    image = np.arange(700 * 900, dtype=np.uint8).reshape(700, 900)
    padded, pad_shape, original_shape = reflect_pad_to_multiple(image, patch_size=528)
    assert original_shape == image.shape
    assert padded.shape == (1056, 1056)
    assert pad_shape == {"bottom": 356, "right": 156}

    reconstructed = np.zeros_like(padded)
    for y, x, tile in iter_stride_tiles(padded, patch_size=528):
        reconstructed[y : y + 528, x : x + 528] = tile
    h, w = original_shape
    np.testing.assert_array_equal(reconstructed[:h, :w], image)


def test_full_image_normalisation_matches_training_transform_on_uint8() -> None:
    patch = np.arange(528 * 528, dtype=np.uint32).reshape(528, 528) % 256
    patch_u8 = patch.astype(np.uint8)
    stats = full_image_stats(patch_u8)
    got = normalise_uint8_patch_array(patch_u8, stats["mean"], stats["std"])
    expected = (patch_u8.astype(np.float32) / 255.0 - stats["mean"]) / (stats["std"] + 1e-6)
    np.testing.assert_allclose(got, expected[None, :, :])


def test_visual_review_schema_defaults_to_null_and_is_json_safe() -> None:
    defaults = visual_review_defaults()
    assert set(defaults) == set(VISUAL_FIELDS)
    assert all(value is None for value in defaults.values())
    json.dumps(defaults)


def test_manual_review_template_has_null_author_fields() -> None:
    payload = review_template_payload(build_manifest_entries()[:2])
    assert payload["author_review_required"] is True
    assert payload["auto_metric"] is False
    assert len(payload["entries"]) == 2
    for row in payload["entries"]:
        assert "agent_suggestion" not in row
        for field in MANUAL_REVIEW_FIELDS:
            assert field in row
            assert row[field] is None


def test_apply_manual_review_rejects_incomplete_template(tmp_path: Path) -> None:
    entry = build_manifest_entries()[0]
    review = review_template_payload([entry])
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    inference_path = tmp_path / "inference.json"
    inference_path.write_text(json.dumps({"images": [{"expnum": entry.expnum, "detector": entry.detector}]}))

    with pytest.raises(ValueError, match="still null"):
        apply_manual_review(review_path, inference_path)


def test_apply_manual_review_merges_completed_author_fields(tmp_path: Path) -> None:
    entry = build_manifest_entries()[0]
    review = review_template_payload([entry])
    row = review["entries"][0]
    row.update({
        "visual_hit_unet_binary": True,
        "visual_hit_unet_prob": True,
        "visual_hit_hough": False,
        "visual_fp_clean": False,
        "notes": "author checked final figures",
        "reviewer": "B. Gracias",
        "review_date": "2026-06-03",
    })
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    inference_path = tmp_path / "inference.json"
    inference_path.write_text(json.dumps({
        "manual_review_summary": None,
        "images": [{"expnum": entry.expnum, "detector": entry.detector}],
    }))

    merged = apply_manual_review(review_path, inference_path)

    image = merged["images"][0]
    assert image["visual_hit_unet_binary"] is True
    assert image["visual_hit_hough"] is False
    assert image["human_review_status"] == "reviewed"
    assert image["human_review_notes"] == "author checked final figures"
    assert merged["manual_review_summary"]["unet_binary_visual_hits"] == 1
    assert merged["manual_review_summary"]["hough_visual_hits"] == 0
    assert merged["manual_review_summary"]["auto_metric"] is False
