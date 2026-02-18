from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def _build_log_path(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / f"run_{timestamp}.log"


def get_logger(
    name: str,
    level: int = logging.INFO,
    run_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Create or retrieve a configured file logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.
    run_dir : Optional[Path]
        Directory where log file will be written.
        Defaults to results/logs/.

    Returns
    -------
    logging.Logger
    """
    logger: logging.Logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    base_dir: Path = run_dir or Path("results") / "logs"
    log_path: Path = _build_log_path(base_dir)

    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger