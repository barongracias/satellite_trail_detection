"""Project-wide constants shared across training, evaluation, and data preparation."""

# Reproducibility
GLOBAL_SEED: int = 2804

# Patch extraction
PATCH_SIZE: int = 528
POS_NEG_RATIO: int = 3  # negative patches sampled per positive patch

# Dataset splits (image-level, stratified by trail-pixel count)
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
# TEST_RATIO is implicitly 1 - TRAIN_RATIO - VAL_RATIO = 0.15

# U-Net architecture (paper-faithful: filters 8 -> 128)
BASE_CHANNELS: int = 8

# Training defaults
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_LR: float = 1e-3
DEFAULT_EPOCHS: int = 50

# Threshold for binary prediction (swept on val; paper used 0.58)
DEFAULT_THRESHOLD: float = 0.5

# Reference paper metrics (Stoppa et al. 2024, threshold=0.58, 20k-patch test set)
PAPER_PRECISION: float = 0.94
PAPER_RECALL: float = 0.94
PAPER_FNR_PRE_HOUGH: float = 0.0698
PAPER_FNR_POST_HOUGH: float = 0.0338
