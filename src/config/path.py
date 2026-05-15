"""Canonical project path helpers derived from the repository root."""

from pathlib import Path

# Repository root is two levels above this file (src/config/path.py -> src/ -> root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Data
DATA_ROOT: Path = PROJECT_ROOT / "data" / "subset" / "processed"
PATCH_DIR: Path = PROJECT_ROOT / "data" / "patches"
METADATA_CSV: Path = (
    PROJECT_ROOT / "data" / "subset" / "metadata" / "Satellites_Catalog_Application.csv"
)

# Results
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
CHECKPOINTS_DIR: Path = RESULTS_DIR / "checkpoints"
LOGS_DIR: Path = RESULTS_DIR / "logs"
CLASSICAL_DIR: Path = RESULTS_DIR / "classical"

# Configs
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
