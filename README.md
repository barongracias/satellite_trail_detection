# Automated Detection of Satellite Trails in Astronomical Images using Deep Learning

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Description

This repository contains the code and analysis for the Research Project submitted as part of the MPhil in Data Intensive Science at the University of Cambridge.

The objective of this project is to reproduce and extend the results of the target paper on automated detection of satellite trails in astronomical images using convolutional neural networks. The project focuses on:

1. Reproducing the U-Net segmentation pipeline.
2. Implementing a classical Hough transform baseline.
3. Replicating reported segmentation metrics.
4. Extending the work through robustness testing, temporal analysis, and alternative architectures.

The project report is located in `report/` and submission instructions are provided in `Instructions.md`.

---

## Project Structure

```
.
├── configs/              # YAML configuration files for experiments
├── data/                 # Local dataset (not version controlled)
├── notebooks/            # EDA and analysis notebooks
├── report/               # Dissertation report and summary
├── results/              # Logs, figures, checkpoints
├── slurm/                # HPC job submission scripts (CSD3)
├── src/
│   ├── classical/        # Hough transform baseline
│   ├── data/             # Dataset loaders and preprocessing
│   ├── evaluation/       # Metrics and validation tools
│   ├── models/           # U-Net and extensions
│   ├── training/         # Training loops
│   └── utils/            # Utilities
├── requirements.txt
└── README.md
```

---

## Data Availability

The dataset consists of astronomical images with corresponding binary masks of satellite trails.

* Images: `.png` astronomical exposures
* Masks: `_mask.png` corresponding segmentation masks
* Metadata: `Satellites_Catalog_Application.csv` containing RA/DEC information

The dataset is not stored in this repository due to size constraints.

---

## Installation

### Requirements

* Python 3.10+
* pip
* CUDA-enabled GPU (for training)
* SLURM access (for CSD3 cluster usage)
* Docker (optional, for containerisation)

---

### Local Setup (Recommended for Development)

1. Clone repository:

```
git clone <repo_url>
cd <repo_name>
```

2. Create virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. (Optional for notebooks)

```
python -m ipykernel install --user --name astro_env
```

---

## Usage

### Exploratory Data Analysis

```
jupyter notebook notebooks/01_dataset_eda.ipynb
```

---

### Training U-Net

```
python src/training/train_unet.py --config configs/unet.yaml
```

---

### Classical Hough Baseline

```
python src/classical/run_hough.py --config configs/hough.yaml
```

---

### Evaluation

```
python src/evaluation/evaluate.py --model checkpoint.pth
```

---

## Running on CSD3 (SLURM)

Submit training job:

```
sbatch slurm/train_gpu.sbatch
```

Logs are written to `results/logs/`.

---

## Docker Usage (Containerised Reproducibility)

Build image:

```
docker build -t satellite-trail .
```

Run container:

```
docker run --gpus all -it satellite-trail
```

---

## Extension Directions

The project explores:

* Performance across observation years
* Correlation between satellite frequency and launch growth
* Robustness to faint trails
* Attention-based U-Net variants
* Resolution sensitivity analysis

---

## Project Status

Currently in active development.
Reproduction phase ongoing.

---

## Author

Baron Gracias
University of Cambridge
MPhil Data Intensive Science