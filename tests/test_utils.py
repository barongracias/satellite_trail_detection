"""Tests for shared utilities: logger and seed harness."""

from __future__ import annotations

from pathlib import Path

import pytest

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
