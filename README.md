# Automated Detection of Satellite Trails in Astronomical Images

Replication and extension of the Stoppa et al. 2024 (A&A 692, A199) satellite-trail
detection pipeline on a 178-image MeerLICHT subset. A U-Net segmenter produces a binary
trail mask, a classical Hough transform bridges gaps in that mask, and a suite of
extensions characterises and improves the detector. MPhil Data Intensive Science
dissertation, University of Cambridge.

The pipeline:

1. **Image-level split** of 178 image/mask pairs (stratified by trail-pixel quartile).
2. **Patch builder** — 528×528 patches at stride 528, written to disk with a manifest.
3. **U-Net training** with an Optuna sweep, multi-seed top-K retraining, and
   validation-only model selection.
4. **Hough post-processing** to recover gaps the U-Net misses.
5. **Classical Hough baseline** as a non-learned reference point.

## Repository Layout

```text
.
├── configs/experiments/      # Versioned experiment configs (YAML)
│   ├── restudy_topk/         #   per-trial × per-seed retrain configs (U-Net)
│   └── attention_topk/       #   per-trial × per-seed retrain configs (Attention U-Net)
├── docs/                     # Sphinx sources + preregistration record
├── report/                   # Dissertation LaTeX (gitignored)
├── results/                  # Logs, checkpoints, figures, JSON metrics (mostly gitignored)
├── scripts/
│   ├── data/                 # Patch build, image/noise stats, dataset inspection, hard-neg mining
│   ├── sweep/                # Top-K config generation + sweep/result aggregation
│   ├── evaluation/           # Threshold sweep, Hough, bootstrap CIs, geometry/FP analyses
│   └── figures/              # Thesis + extension figure generation, prediction visualisation
├── slurm/                    # CSD3 Slurm submission scripts (one per pipeline stage)
├── src/
│   ├── analysis/             # FP decomposition
│   ├── classical/            # Hough-transform baseline
│   ├── config/               # Project constants + canonical path helpers
│   ├── data/                 # Datasets, catalog metadata, joint transforms, image-level splits
│   ├── evaluation/           # EDA + segmentation metrics (incl. bootstrap CIs)
│   ├── inference/            # Two-stage (classifier → U-Net → Hough) inference
│   ├── models/               # U-Net, Attention U-Net, patch classifier, checkpoint loader
│   ├── training/             # U-Net / classifier training entry points, Optuna sweep driver
│   └── utils/                # Logging + seeding
└── tests/                    # Unit tests (run with `pytest -q`)
```

## Data Layout

Paired 8-bit PNG renders (display-scaled, **not** calibrated flux):

- Image: `data/Processed/*_red.fits_full.png`
- Mask:  `data/Processed/*_red_mask.png`
- Catalog CSV: `data/Satellites_Catalog_Application.csv` (local EDA only)

The patch builder writes `data/patches/{train,val,test}/` plus a `manifest.csv`
carrying per-patch split, positive-pixel fraction, and per-image normalisation stats.
`data/` is gitignored — patches are rebuilt on CSD3, not committed.

## Environment Setup

### Local development (CPU, tests + figures)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r local-requirements.txt
```

### HPC / CSD3 (CUDA training)

`hpc-requirements.txt` is a fully pinned lock of the CSD3 environment (CUDA 11.8
PyTorch wheels). `local-requirements.txt` is the same dependency set with relaxed
floors and no CUDA wheels, for CPU-only local work.

```bash
pip install -r hpc-requirements.txt
```

### Docker

The image builds the CSD3-aligned environment and runs the test suite:

```bash
docker build -t satellite-trails .
docker run --rm satellite-trails pytest -q
```

## CSD3 Quickstart

All long-running stages are Slurm jobs under `slurm/`. The end-to-end path:

```bash
# 1. Build the patch dataset, per-image stats, and background-noise calibration
#    (CPU jobs, chained so each waits for the previous).
rm -rf data/patches
jid1=$(sbatch --parsable slurm/build_patches_cpu.sbatch)
jid2=$(MANIFEST=data/patches/manifest.csv \
       sbatch --parsable --dependency=afterok:$jid1 slurm/compute_image_stats_cpu.sbatch)
MANIFEST=data/patches/manifest.csv \
  sbatch --dependency=afterok:$jid2 slurm/compute_background_noise_stats_cpu.sbatch

# 2. Optuna sweep (balanced batch-size allocation, validation-only objective)
STUDY_NAME=unet_paper_arch_noise_f1 \
CONFIG=configs/experiments/unet_paper_arch_noise_base.yaml \
N_TRIALS=45 SKIP_RETRAIN=1 \
  sbatch slurm/optuna_sweep.sbatch

# 3. Retrain top-K trials × 5 seeds, then sweep validation thresholds.
#    Per-trial/seed configs live in configs/experiments/restudy_topk/.
CONFIG=configs/experiments/restudy_topk/topk_t44_s2804.yaml \
  sbatch slurm/train_unet_ampere_long.sbatch

# 4. Evaluate the locked winner: test threshold sweep + Hough post-processing
CHECKPOINT=results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth \
THRESHOLD=0.45 \
  sbatch slurm/threshold_sweep.sbatch
sbatch slurm/hough_postprocess.sbatch

# 5. Prediction figures for the report
CHECKPOINT=results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_best.pth \
THRESHOLD=0.45 \
HOUGH_JSON=results/classical/hough_postprocess_winner_t44_s2804.json \
TAG=winner_t44_s2804 \
  sbatch slurm/visualise_predictions.sbatch
```

The **locked winner** is `unet_paper_arch_noise_topk_t44_s2804_best.pth` at threshold
`0.45` with `full_image` normalisation. It is fixed: no retraining, threshold tuning,
or model reselection downstream of step 3.

### Classical baseline (local or CSD3)

```bash
python -m src.classical.run_hough --config configs/experiments/hough_baseline.yaml
```

### Thesis / extension figures (local, from committed JSON metrics)

```bash
python -m scripts.figures.make_thesis_figures
python -m scripts.figures.make_extension_figures
```

## Extensions

Four themes, all scored on the same test split as the replication baseline and all
post-hoc on the locked winner (no re-selection):

- **A — Architectural.** Two-stage detector (CNN classifier → U-Net → Hough) and an
  Attention U-Net, each compared against the base U-Net.
- **B — Error characterisation.** FP decomposition (intra- vs inter-patch),
  boundary-tolerant evaluation, FP distance-to-mask, and clDice connectivity.
- **C — Training protocol.** Data-efficiency curve (30/50/70/100% of training images),
  hard-negative mining, and a dilated soft-label pilot.
- **D — Inference time.** Ensemble + test-time augmentation and probability-stratified
  Hough post-processing.

A qualitative cold-domain illustration applies the locked MeerLICHT model to DECam
frames (`scripts/figures/decam_cold_inference.py`) — visual inspection only, no
cross-dataset metrics.

## Reproducibility

- Global seed `2804`; multi-seed runs use `{2804, 1234, 42, 7, 13}`.
- Image-level split stratified by trail-pixel quartile. Target 70/15/15; the realised
  split on the 178-image subset is **122 train / 24 val / 32 test** (per-quartile
  flooring sends the remainder to test — leakage-safe and conservative).
- Model and threshold selection are **validation-only**; test metrics are never used
  for selection or tie-breaking.
- Every experiment is reproducible from a `configs/` file plus runtime overrides.
- `results/` and `*.pth` are gitignored; selected deliverable JSONs and figures are
  force-added. Trained weights are kept local only.

## Testing

```bash
pytest -q
```

## Documentation

The `src/` API reference is built with Sphinx (NumPy-style docstrings, autodoc):

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Heavy dependencies are mocked in `docs/conf.py`, so the same build runs on
ReadTheDocs (`.readthedocs.yaml`) without the CUDA/scientific stack.

## Author

Baron Gracias — University of Cambridge MPhil Data Intensive Science

Baron Gracias — University of Cambridge MPhil Data Intensive Science
