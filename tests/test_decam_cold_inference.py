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
    VISUAL_FIELDS,
    full_image_stats,
    iter_stride_tiles,
    normalise_uint8_patch_array,
    reflect_pad_to_multiple,
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
