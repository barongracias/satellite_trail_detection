"""Tests for shared utilities: logger and decorators."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.decorators import log_call, timer
from src.utils.logger import get_logger


class _Dummy:
    def __init__(self) -> None:
        self.logger = logging.getLogger("dummy")
        self.logger.addHandler(logging.NullHandler())

    @log_call
    @timer
    def compute(self, x: int) -> int:
        return x * 2


def test_decorators_execute_without_error() -> None:
    obj = _Dummy()
    assert obj.compute(5) == 10


def test_logger_creates_a_new_log_file_for_each_run(tmp_path: Path) -> None:
    get_logger("test_logger", run_dir=tmp_path).info("First run")
    get_logger("test_logger", run_dir=tmp_path).info("Second run")

    log_files = sorted(tmp_path.glob("*.log"))
    assert len(log_files) == 2
