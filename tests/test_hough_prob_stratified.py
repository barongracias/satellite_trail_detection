"""Tests for the probability-stratified Hough approximation helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

# Add the repo root to sys.path so `scripts/` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hough_prob_stratified import _DEFAULT_STRATA, _hough_union  # noqa: E402


def _line_prob_canvas() -> np.ndarray:
    """A 200x200 probability canvas with a long, mostly-confident vertical line."""
    canvas = np.zeros((200, 200), dtype=np.float32)
    canvas[:, 100] = 0.9          # confident core, length 200 (>= min_line_length)
    canvas[:, 99] = 0.2           # faint shoulder, only visible at the lowest stratum
    return canvas


def test_stratified_union_superset_of_binary_baseline() -> None:
    # The stratified strata include the binary baseline threshold (0.1), so the
    # union must cover at least as many line pixels as the single-stratum binary.
    canvas = _line_prob_canvas()
    binary = _hough_union(canvas, [0.1]) > 0
    stratified = _hough_union(canvas, list(_DEFAULT_STRATA)) > 0
    assert binary.sum() > 0                      # the long line is detected
    assert stratified.sum() >= binary.sum()      # union never loses coverage
    assert np.all(stratified | ~binary)          # binary pixels ⊆ stratified


def test_stratified_detects_confident_line_alone() -> None:
    # A high-confidence line is recovered even by the strict (0.9) stratum only.
    canvas = _line_prob_canvas()
    strict = _hough_union(canvas, [0.9]) > 0
    assert strict.sum() > 0
