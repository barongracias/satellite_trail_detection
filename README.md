# Automated Detection of Satellite Trails in Astronomical Images

[![Documentation Status](https://readthedocs.org/projects/satellite-trail-detection/badge/?version=latest)](https://satellite-trail-detection.readthedocs.io/en/latest/)

Replication and diagnostic extension of the Stoppa et al. 2024 (A&A 692, A199)
satellite-trail detection pipeline on a 178-image MeerLICHT subset. A U-Net
segmenter produces a binary trail mask, a probabilistic Hough transform bridges
gaps in that mask, and post-hoc diagnostics place much of the remaining strict
precision gap near annotated boundaries, with a bounded empty-patch/background
component. MPhil Data Intensive Science dissertation, University of Cambridge.

📖 **Documentation:** <https://satellite-trail-detection.readthedocs.io/>

The pipeline:

1. **Image-level split** of 178 image/mask pairs (stratified by trail-pixel quartile).
2. **Patch builder** — 528×528 patches at stride 528, written to disk with a manifest.
3. **U-Net training** with an Optuna sweep, multi-seed top-K retraining, and
   validation-only model selection.
4. **Hough post-processing** to recover gaps the U-Net misses.
5. **Error diagnosis** — boundary-tolerant scoring, FP decomposition, full-frame
   Hough verification, and a blinded single-author re-annotation audit.
6. **Model variants** — classifier gating, Attention U-Net, training-protocol
   pilots, ensembling, and probability-stratified Hough.

## Live demo

[**trail-scope**](https://github.com/barongracias/trail-scope) — an inference demo of this thesis's locked satellite-trail detector (FastAPI + Next.js). Upload an astronomical image (or try examples) to see the locked U-Net + Hough overlay and a predicted mask.

## Repository Layout

```text
.
├── configs/experiments/      # Versioned experiment configs (YAML)
│   ├── restudy_topk/         #   per-trial × per-seed retrain configs (U-Net)
│   └── attention_topk/       #   per-trial × per-seed retrain configs (Attention U-Net)
├── docs/                     # Sphinx sources + preregistration record
├── report/                   # Dissertation LaTeX sources and compiled report
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

## Data

### Layout

Paired 8-bit PNG renders (display-scaled, **not** calibrated flux):

- Image: `data/Processed/*_red.fits_full.png`
- Mask:  `data/Processed/*_red_mask.png`
- Catalog CSV: `data/Satellites_Catalog_Application.csv` (local EDA only)

The patch builder writes `data/patches/{train,val,test}/` plus a `manifest.csv`
carrying per-patch split, positive-pixel fraction, and per-image normalisation stats.
`data/` is gitignored — patches are rebuilt on CSD3, not committed.

### Availability

**Primary (MeerLICHT).** Training, validation, and test use a 178-image MeerLICHT
subset with hand-verified trail masks, provided through the MeerLICHT collaboration. These are collaboration data and are not redistributed here; they are available on request, subject to the MeerLICHT data policy. All splits are reproducible from the image-level split logic (`src/data/splits.py`, seed `2804`) once the image/mask pairs are in place.

**Gold audit.** The blinded re-annotation audit uses private sealed crops and
single-author masks under `data/gold/`. Those crops, masks, verdicts, and sealed
manifest are not redistributed. Only the evaluation code and summary artefacts
are in the repository, including `results/classical/gold_audit_eval.json`,
`results/classical/gold_audit_bootstrap.json`, and the corresponding figure
scripts under `scripts/figures/`.

**Extension (DECam).** The qualitative cold-domain demo uses nine measured-streak DECam detector images from the public NSF NOIRLab Astro Data Archive, retrieved via the RECA codebase (Stoppa-adjacent; arXiv:2603.10790, `iausathub/reca-streaks`). The exact frames are predeclared in `results/classical/decam_cold_manifest.json` (expnum/detector), so the demo is reproducible from the archive without bundling raw FITS. Any use must carry the NOIRLab acknowledgement recorded in `results/classical/decam_cold_inference.json`.

## Trained Weights

The single locked winning model is committed, so the repository is self-contained:

- `results/checkpoints/model-best.pth` (5.7 MB; SHA-256
  `ff680804f6cf66d6948dcd76af4958c4427099ecdb45bab0140ac80314b8e55b`) — the reported
  model, evaluated at threshold `0.45` with `full_image` normalisation. It is a
  byte-identical copy of `unet_paper_arch_noise_topk_t44_s2804_best.pth`, the descriptive
  filename the result JSONs reference (architecture-faithful, noise-augmented, top-K trial
  44, seed 2804); the SHA-256 lets any reader verify the committed copy against that
  original. Pass it to any pipeline stage with `CHECKPOINT=results/checkpoints/model-best.pth`.

All other intermediate checkpoints (`*.pth`, ~120 MB) are gitignored and kept local.

## Environment Setup

### Local development (CPU, tests + figures)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r local-requirements.txt
pip install -e .
```

### HPC / CSD3 (CUDA training)

`hpc-requirements.txt` is a fully pinned lock of the CSD3 environment (CUDA 11.8
PyTorch wheels). `local-requirements.txt` is the same dependency set with relaxed
floors and no CUDA wheels, for CPU-only local work.

```bash
pip install -r hpc-requirements.txt
pip install -e .
```

### Docker

The image builds a portable CPU environment from `docker-requirements.txt`
(runtime deps + pytest/ruff, no Jupyter stack) and runs the test suite:

```bash
docker build -t satellite-trails .
docker run --rm satellite-trails pytest -q
```

## CSD3 Reproduction

Long-running training and evaluation stages are Slurm jobs under `slurm/`. The
full reproduction path rebuilds `data/patches/`, computes image/noise
statistics, runs the balanced Optuna sweep, retrains the top five trials across
five seeds, evaluates the locked winner, runs Hough post-processing, and
regenerates report figures from committed JSON artefacts.

The **locked winner** is committed as `results/checkpoints/model-best.pth`, a
byte-identical copy of `unet_paper_arch_noise_topk_t44_s2804_best.pth`, and is
evaluated at threshold `0.45` with `full_image` normalisation. Downstream
analyses are inference-only: no retraining, threshold reselection, or model
reselection. The exact CSD3 command sequence is maintained in the ReadTheDocs
reproduction guide.

### Classical baseline (local or CSD3)

```bash
python -m src.classical.run_hough --config configs/experiments/hough_baseline.yaml
```

### Thesis / extension figures

```bash
python -m scripts.figures.make_thesis_figures
python -m scripts.figures.make_extension_figures
```

Most report figures are generated from committed JSON metrics. Gold-audit
qualitative figures additionally require the private `data/gold/` crops and masks:

```bash
python -m scripts.figures.make_gold_audit_figure
python -m scripts.figures.make_gold_audit_annotation_examples
```

## Extensions

All extensions are post-hoc on the locked winner unless explicitly described as
separate validation-only model selection; no test metric is used for reselection.
The final diagnosis is that recall and Hough pixel completeness reproduce closely,
while diagnostics place much of the strict precision residual near annotated
boundaries. The single-author re-annotation supports this interpretation
directionally, without acting as a true-mask oracle.

- **A — Architectural.** Two-stage detector (CNN classifier → U-Net → Hough) and an
  Attention U-Net, each compared against the base U-Net.
- **B — Error characterisation.** FP decomposition (intra- vs inter-patch),
  boundary-tolerant evaluation, FP distance-to-mask, clDice connectivity, Hough
  line-thickness accounting, and five-seed diagnostic bands.
- **C — Training protocol.** Data-efficiency curve (30/50/70/100% of training images),
  hard-negative mining, and a dilated soft-label pilot.
- **D — Inference time.** Ensemble + test-time augmentation and probability-stratified
  Hough post-processing.
- **E — Blinded re-annotation audit.** 64 sealed crops were re-annotated under an
  evaluation-only protocol. The original MeerLICHT masks, the re-annotation, and
  the locked model all have median stroke width 6 px; strict micro-F1 against the
  re-annotation is 0.890 for the original labels and 0.893 for the model.

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
- Signal-dependent noise augmentation uses the tracked calibration
  `results/classical/background_noise_calibration.json` (`alpha=0.0209200478`,
  `beta=0.0885920600`, 9,894 empty train patches; manifest SHA-256
  `e4082a0c16fd58aeaafb061baf846555fe76543700058bd7119f08063c7cf24b`).
- `results/` and `*.pth` are gitignored; selected deliverable JSONs, figures, and the
  locked model (`results/checkpoints/model-best.pth`) are force-added. Other checkpoints
  are kept local only.
- The preliminary three-seed noise-multiplier screen and the early free-sampling Optuna
  sweep referenced in the report are **intentionally not committed**: the noise screen
  predates the architecture-faithful model, and the free-sampling study was superseded by
  the committed 45-trial balanced study (`unet_paper_arch_noise_f1.db`). The tracked
  calibration and that balanced study are the artefacts of record.
- Full-coverage tiled Hough verification is recorded in
  `results/classical/hough_fullframe_winner_t44_s2804.json`; it reproduces the
  full-coverage parity canvas pixel-for-pixel, so the Hough completeness result is
  not a patch-sampling or seam artefact.
- The gold audit is **evaluation-only**: no retraining, threshold reselection, or
  mask replacement. Its private inputs remain under gitignored `data/gold/`; the
  committed summaries are `results/classical/gold_audit_eval.json` (schema v3) and
  `results/classical/gold_audit_bootstrap.json`.

### Computational requirements

- **Hardware.** GPU training/inference ran on CSD3 (Cambridge), Ampere partition,
  one NVIDIA A100 (40 GB) per job. CPU-only paths — tests, figure generation, and the
  classical Hough baseline — need no GPU.
- **Software.** Python 3.11.x; PyTorch CUDA 11.8 wheels (`hpc-requirements.txt`). The
  full sweep was 45 Optuna trials (balanced batch-size allocation), then the top-5 trials
  retrained at 5 seeds for 75 epochs each; BF16 autocast and TF32 matmul enabled.
- **Determinism.** `torch`/`numpy`/`random` are seeded and deterministic algorithms are
  requested where available. Residual GPU non-determinism is handled by reporting
  5-seed means rather than a single run; the post-hoc OpenCV
  `cv2.HoughLinesP` stage is the remaining non-seeded classical step.

## Testing

```bash
pytest -q
```

## Documentation

Full documentation — install guide, usage examples, and the `src/` API reference —
is hosted on Read the Docs: **<https://satellite-trail-detection.readthedocs.io/>**.

To build it locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Heavy dependencies are mocked in `docs/conf.py`, so the same build runs on
ReadTheDocs (`.readthedocs.yaml`) without the CUDA/scientific stack.

## License

Released under the MIT License — see [`LICENSE`](LICENSE). The MeerLICHT and DECam
data are **not** covered by this licence and remain under their respective providers'
terms (see [Data](#data)).

## Citation

If you use this code, please cite the dissertation — see [`CITATION.cff`](CITATION.cff).
This work replicates and extends Stoppa et al. 2024 (A&A 692, A199).

## Author

Baron Gracias — University of Cambridge MPhil Data Intensive Science
(supervisor: Dr Eduardo Gonzalez-Solares)

## AI/LLM usage
I have used Codex (ChatGPT) and Claude Code to support me across this work, primarily in the following areas:

- Providing advice and usage support for libraries like PyTorch and PIL.
- Generating boilerplate code for plotting, testing and scripting.
- Generating function docstrings.
- Reviewing and auditing project structure for completeness, consistency, accuracy and best practices in software engineering.
