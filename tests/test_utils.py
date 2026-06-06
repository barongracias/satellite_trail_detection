"""Tests for shared utilities: logger, seed harness, and image helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.utils.imaging import largest_component_mask, resize_for_display

torch = pytest.importorskip("torch")

from src.utils.logger import get_logger
from src.utils.seed import seed_everything


def test_logger_creates_a_new_log_file_for_each_run(tmp_path: Path) -> None:
    get_logger("test_logger", run_dir=tmp_path).info("First run")
    get_logger("test_logger", run_dir=tmp_path).info("Second run")

    log_files = sorted(tmp_path.glob("*.log"))
    assert len(log_files) == 2


def test_seed_everything_runs_without_error() -> None:
    seed_everything(42)


def test_largest_component_mask_keeps_only_the_biggest_blob() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2, 2:5] = True          # 3-pixel sliver
    mask[10:14, 10:14] = True    # 16-pixel block
    kept = largest_component_mask(mask)
    assert int(kept.sum()) == 16
    assert kept[11, 11]
    assert not kept[2, 3]


def test_largest_component_mask_passes_through_empty_mask() -> None:
    empty = np.zeros((8, 8), dtype=bool)
    assert not largest_component_mask(empty).any()


def test_resize_for_display_caps_largest_side_and_passes_small_arrays() -> None:
    big = np.zeros((3000, 2000), dtype=np.float32)
    resized = resize_for_display(big, max_dim=1500)
    assert max(resized.shape) == 1500

    small = np.zeros((100, 80), dtype=np.float32)
    assert resize_for_display(small, max_dim=1500) is small
