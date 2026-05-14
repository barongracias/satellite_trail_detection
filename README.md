# Automated Detection of Satellite Trails in Astronomical Images Using Deep Learning

This repository contains the code and analysis for a University of Cambridge MPhil dissertation project on automated detection of satellite trails in astronomical images.

The immediate dissertation objective is faithful replication of a paper that combines:

1. A U-Net segmentation pipeline for binary trail detection.
2. A classical Hough transform baseline.
3. Comparable evaluation metrics and reproducible experimental reporting.

Replication takes priority over extensions. Planned extension work is centred on a CNN-classifier two-stage detector (classifier filters patches before the U-Net), with robustness studies, a data-efficiency study, and an Attention U-Net comparison as secondary directions. Sky-coordinate and time-trend extensions are out of scope because the available data is PNG-only with no FITS headers.

## Current Status

Implemented and usable now:

- Patch-based image and mask loading under `src/data/`
- Metadata catalogue access under `src/data/catalog.py`
- Exploratory analysis utilities under `src/evaluation/eda.py`
- Binary segmentation metrics shared by U-Net and Hough evaluation under `src/evaluation/segmentation.py`
- A meeting-ready EDA notebook at `notebooks/01_dataset_eda.ipynb`
- A classical Hough-transform baseline under `src/classical/`
- A lightweight U-Net implementation under `src/models/unet.py`
- A train, validation, and test experiment entry point under `src/training/train_unet.py`
- Logging utilities that write timestamped runtime logs to `results/logs/`
- Unit tests for data handling, EDA helpers, evaluation utilities, the Hough baseline, and the U-Net path

Still to do:

- Verify EDA overlays and quantify mask undermasking before training metrics are trusted
- Switch the training loss from `BCEWithLogitsLoss(pos_weight=...)` to a combo loss (BCE + Dice) to match the target paper
- Move from patch-level to image-level train/val/test splits (70/15/15, stratified by trail-pixel count)
- Pre-compute 512×512 patches to disk with a configurable positive:negative ratio (default 1:3) instead of indexing on the fly
- Add training-set augmentation (flips, 90° rotations, optional selective shifts on positives)
- Add a determinism harness and an Optuna-driven hyperparameter sweep
- Tune the Hough baseline parameters against the target paper
- Select the U-Net probability threshold by a precision-recall sweep on the validation set
- Run a data-efficiency study at 30/50/70/100% of training images
- Add richer experiment configuration files under `configs/` and a paper-vs-ours replication comparison table

## Repository Layout

```text
.
├── configs/                  # Configuration scaffolding for later experiments
├── data/                     # Local dataset storage
├── notebooks/                # Exploratory analysis notebooks
├── results/                  # Logs, figures, checkpoints, and summaries
├── scripts/                  # Small package-safe helper entry points
├── slurm/                    # CSD3 job submission scripts
├── src/
│   ├── classical/            # Hough-transform baseline
│   ├── data/                 # Datasets, metadata, transforms, splits
│   ├── evaluation/           # EDA and segmentation evaluation helpers
│   ├── models/               # U-Net and future model variants
│   ├── training/             # Training entry points
│   └── utils/                # Logging and decorators
└── tests/                    # Unit tests
```

## Data Layout

The processed segmentation subset uses paired PNG files:

- Image: `*_red.fits_full.png`
- Mask: `*_red_mask.png`
- Metadata CSV: `data/subset/metadata/Satellites_Catalog_Application.csv`

The full astronomical images are very large, so patch-based loading is the intended training workflow.

## Environment Setup

### Local development

Use Python 3.11 if practical. Python 3.10+ is supported by the repository code.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r local-requirements.txt
```

Optional notebook kernel registration:

```bash
python -m ipykernel install --user --name satellite_trails
```

### HPC or CSD3 environment

Install the GPU-oriented environment separately:

```bash
pip install -r hpc-requirements.txt
```

## Core Commands

### Run tests

```bash
pytest -q
```

### Inspect the processed dataset from the command line

```bash
python -m scripts.inspect_dataset --data-root data/subset/processed --patch-size 512 --stride 512
```

### Launch the EDA notebook

```bash
jupyter notebook notebooks/01_dataset_eda.ipynb
```

The notebook saves discussion-ready figures to `results/figures/`.

### Run the classical Hough baseline

```bash
python -m src.classical.run_hough --data-root data/subset/processed
```

This writes per-image metrics to `results/classical/hough_metrics.csv` and logs the aggregated metrics.

### Run a short local U-Net smoke test

```bash
python -m src.training.train_unet \
  --data-root data/subset/processed \
  --epochs 1 \
  --max-steps 5 \
  --eval-max-batches 10 \
  --auto-pos-weight \
  --experiment-name unet_smoke
```

### Run a baseline train and validation experiment

```bash
python -m src.training.train_unet \
  --data-root data/subset/processed \
  --epochs 3 \
  --batch-size 2 \
  --auto-pos-weight \
  --experiment-name unet_baseline
```

This produces:

- a latest checkpoint
- a best-validation checkpoint
- a JSON experiment summary with train, validation, and test metrics

## Dataset Snapshot

On the current expanded local subset, using non-overlapping `512 x 512` patches:

- Total full image-mask pairs: `21`
- Total patches: `8400`
- Empty patches: `7880`
- Non-empty patches: `520`
- Empty patch fraction: `0.9381`
- Positive pixel fraction: `0.000714`

This remains a severely imbalanced segmentation problem. The current baseline therefore supports `BCEWithLogitsLoss(pos_weight=...)`, including automatic estimation from the processed masks.

## Evaluation

The current evaluation module provides reusable binary segmentation metrics for both learned and classical baselines:

- Precision
- Recall
- Dice or F1
- Intersection over Union
- Accuracy
- Specificity

These metrics are computed from shared confusion-count totals so the U-Net and Hough outputs can be compared on the same basis.

## Reproducibility Notes

- Reusable logic lives under `src/`.
- Notebook plotting relies on reusable helpers rather than large inline analysis blocks.
- Runtime logs are written to `results/logs/`.
- Figures from the EDA workflow are written to `results/figures/`.
- Training checkpoints and experiment summaries are written to `results/checkpoints/`.
- The Hough baseline writes per-image metrics to `results/classical/`.

## Near-Term Development Priorities

1. Verify EDA mask-quality and quantify undermasking before any training metrics are trusted.
2. Switch the training loss to the combo loss (BCE + Dice) and the splits to image-level 70/15/15.
3. Pre-compute patches to disk with a configurable positive:negative ratio and add training-set augmentation.
4. Add the config scaffold under `configs/`, a determinism harness, and an Optuna sweep driver.
5. Run the first real U-Net training on CSD3 and produce the paper-vs-ours replication table.
6. Tune the Hough baseline, select the U-Net threshold by PR sweep, and run the data-efficiency study.
7. Build the CNN-classifier two-stage detector as the primary dissertation extension.

## Author

Baron Gracias  
University of Cambridge  
MPhil Data Intensive Science
