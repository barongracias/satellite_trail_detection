from pathlib import Path
from src.utils.logger import get_logger

def test_logger_creates_run_directory(tmp_path: Path) -> None:
    logger = get_logger("test_logger", run_dir=tmp_path)
    logger.info("Test message")

    log_files = list(tmp_path.glob("*.log"))
    assert len(log_files) == 1