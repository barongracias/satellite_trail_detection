from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def _build_log_path(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return run_dir / f"run_{timestamp}.log"


def _reset_handlers(logger: logging.Logger) -> None:
    """Close and remove all existing handlers from a logger."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def get_logger(
    name: str,
    level: int = logging.INFO,
    run_dir: Path | None = None,
) -> logging.Logger:
    """Create a timestamped file logger for a single run."""
    logger: logging.Logger = logging.getLogger(name)

    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        _reset_handlers(logger)

    base_dir: Path = run_dir or Path("results") / "logs"
    log_path: Path = _build_log_path(base_dir)

    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
