# Automated Detection of Satellite Trails in Astronomical Images Using Deep Learning

Replication of the Stoppa et al. 2024 (A&A 692, A199) satellite trail detection pipeline on MeerLICHT telescope images, with extensions.

1. U-Net segmentation pipeline for binary trail detection.
2. Classical Hough transform baseline.

## Repository Layout

```text
.
├── configs/                  # Experiment configuration files
├── data/                     # Local dataset storage
├── notebooks/                # Exploratory analysis notebooks
├── report/                   # Dissertation LaTeX source
├── results/                  # Logs, figures, checkpoints, and summaries
├── scripts/                  # CLI helpers: dataset build/inspection, sweep aggregation, evaluation, figures
├── slurm/                    # CSD3 job submission scripts
├── src/
│   ├── classical/            # Hough-transform baseline
│   ├── config/               # Project constants and path helpers
│   ├── data/                 # Datasets, metadata, transforms, splits
│   ├── evaluation/           # EDA and segmentation evaluation helpers
│   ├── models/               # U-Net, Attention U-Net, patch classifier, shared loader
│   ├── training/             # Training entry points
│   └── utils/                # Logging, seeding, and decorators
└── tests/                    # Unit tests
```

## Data Layout

Processed segmentation dataset uses paired PNG files:

- Image: `*_red.fits_full.png`
- Mask: `*_red_mask.png`
- Metadata CSV: `data/Satellites_Catalog_Application.csv` (local EDA only)

## Environment Setup

### Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r local-requirements.txt
```

Optional notebook kernel:

```bash
python -m ipykernel install --user --name satellite_trails
```

### HPC / CSD3

```bash
pip install -r hpc-requirements.txt
```

## Core Commands

```bash
# Run tests
pytest -q

# Classical Hough baseline
python -m src.classical.run_hough --config configs/experiments/hough_baseline.yaml

# U-Net smoke test (local, CPU, 5 steps)
python -m src.training.train_unet --config configs/experiments/unet_smoke.yaml
```

## Reproducibility

- Global seed `2804`; multi-seed runs use `{2804, 1234, 42, 7, 13}`.
- Image-level split stratified by trail-pixel quartile. Target 70/15/15; the realised
  split on the 178-image subset is **122 train / 24 val / 32 test** (per-quartile
  flooring sends the remainder to test — leakage-safe and conservative).
- Model and threshold selection are **validation-only**; test metrics are never used
  for selection or tie-breaking.
- `results/` and `*.pth` are gitignored; selected deliverable JSONs and figures are
  force-added. Trained weights are kept local only.

## Author

Baron Gracias — University of Cambridge MPhil Data Intensive Science