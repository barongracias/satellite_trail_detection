from pathlib import Path

from src.utils.logger import get_logger


def test_logger_creates_a_new_log_file_for_each_run(tmp_path: Path) -> None:
    first_logger = get_logger("test_logger", run_dir=tmp_path)
    first_logger.info("First run")

    second_logger = get_logger("test_logger", run_dir=tmp_path)
    second_logger.info("Second run")

    log_files = sorted(tmp_path.glob("*.log"))
    assert len(log_files) == 2
