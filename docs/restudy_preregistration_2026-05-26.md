# Restudy Preregistration: Architecture-Faithful Robust U-Net

Date: 2026-05-26
Status: active M5.6 protocol; supersedes `agents/protocols/restudy_preregistration_2026-05-25.md` before any M5.6 test metrics were inspected

## Motivation

The 2026-05-25 M5.6 restudy fixed the loss, normalisation, precision, augmentation, and validation-only selection discipline, but inspection of the Keras model config embedded in `asta/model-best.h5` showed that the current PyTorch U-Net was only parameter-scale faithful. The released model differs in pooling, LeakyReLU slope, dropout type, dropout placement, dropout rates, and initialisation. Since the dissertation priority is paper-faithful replication, the current-architecture `unet_paper_noise_f1` run is superseded before test evaluation.

This protocol locks the architecture-faithful restart. No test metrics from M5.6 checkpoints may be inspected before validation-only selection is recorded in `agents/report_notes.md`.

## Superseded Run Handling

Cancel queued current-architecture top-K jobs on CSD3:

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.8T %.20R"
seq 29720721 29720750 | xargs -n 30 scancel
squeue -u "$USER" -o "%.18i %.9P %.30j %.8T %.20R"
```

Delete current-architecture M5.6 artefacts before launching the restart:

```bash
rm -f configs/experiments/unet_paper_noise_base.yaml
rm -f results/classical/unet_paper_noise_f1.db
rm -f results/classical/unet_paper_noise_f1_best.json
rm -f results/classical/unet_paper_noise_canary.db
rm -f results/classical/unet_paper_noise_canary_best.json
rm -f results/classical/threshold_sweep_unet_paper_noise_topk_t*_s*.json
rm -f results/checkpoints/unet_paper_noise_f1_trial_*_best.pth
rm -f results/checkpoints/unet_paper_noise_f1_trial_*_latest.pth
rm -f results/checkpoints/unet_paper_noise_f1_trial_*_summary.json
rm -f results/checkpoints/unet_paper_noise_canary_trial_000_best.pth
rm -f results/checkpoints/unet_paper_noise_canary_trial_000_latest.pth
rm -f results/checkpoints/unet_paper_noise_canary_trial_000_summary.json
rm -f results/checkpoints/unet_paper_noise_topk_t*_s*_best.pth
rm -f results/checkpoints/unet_paper_noise_topk_t*_s*_latest.pth
rm -f results/checkpoints/unet_paper_noise_topk_t*_s*_summary.json
rm -rf configs/experiments/restudy_topk
```

Do not delete `unet_ablation_0`, `unet_sweep_paper_faithful`, or summary diagnostic JSONs. Note: `configs/experiments/unet_ablation_0.yaml` (the training recipe YAML) was deliberately removed in the 2026-05-27 cleanup as a stale artefact; all reportable provenance is preserved in the CSD3 checkpoint (`results/checkpoints/unet_ablation_0_best.pth`), `results/checkpoints/unet_ablation_0_summary.json`, and the associated threshold/Hough/bootstrap JSONs under `results/classical/`. Historical 512×512 parity artefacts are provenance only and must not be used for the M5.6 528×528 winner.

## Fixed Architecture

Implement the U-Net architecture recovered from `asta/model-best.h5`:

- input/output path remains PyTorch logits internally; do not add a final sigmoid unless the loss/evaluation path is changed consistently;
- base filters `8 -> 16 -> 32 -> 64 -> 128` with four encoder downsampling stages;
- `AveragePool2d` downsampling in all four encoder stages;
- transposed-convolution upsampling;
- skip concatenations at matching decoder stages;
- LeakyReLU `negative_slope=0.3` for every convolutional activation;
- standard element-wise `Dropout`, not `Dropout2d`;
- dropout between the two convolutions in every double-conv block;
- dropout rates, from encoder stem through final decoder block: `[0.1, 0.1, 0.2, 0.2, 0.3, 0.2, 0.2, 0.1, 0.1]`;
- He-normal initialisation for 3x3 convolutions;
- Glorot-uniform initialisation for transposed convolutions and the final 1x1 convolution;
- zero biases;
- no BatchNorm.

Patch geometry is now locked to the released wrapper: 528x528 patches at stride 528. This divides the 10,560x10,560 MeerLICHT frames exactly into 20x20 grids and removes the previous 512x512 border-gap deviation.

## Fixed Training Bundle

The restart fixes the paper-faithful and robustness-practice axes:

- `patch_size: 528`, `stride: 528`
- `num_workers: 8`
- `use_amp: true`
- `amp_dtype: bfloat16`
- `float32_matmul_precision: high`
- `normalisation: full_image`
- `augment_train: true`
- no `pos_weight` in `BCEWithLogitsLoss` (`auto_pos_weight` machinery removed in D1 cleanup 2026-05-27)
- `dice_denominator_squared: true`
- `dice_smooth: 1.0e-4`
- `lr_scheduler: cosine`
- `noise_augment: true`
- `noise_std_multiplier: 1.0`

The 1.0x signal-dependent stochastic noise remains a robustness-motivated addition. The M5.5 screen did not prove a statistically decisive validation-F1 gain, but it showed the 1.0x setting was not materially harmful and is physically better motivated for detector/background robustness.

Before the restart, the sampled train/validation/test patch dataset must be rebuilt at 528x528 stride 528 and followed by `scripts/compute_image_stats.py --manifest data/patches/manifest.csv`. Historical 512x512 manifests are inadmissible for the architecture-faithful M5.6 run.

## Search Space

Search only:

- `learning_rate`: log-uniform `[1e-4, 1e-3]` for all batch sizes;
- `bce_weight`: uniform `[0.2, 0.8]`; `dice_weight = 1.0 - bce_weight`;
- `batch_size`: balanced allocation over `{8, 16, 32}`.

Do not search dropout, warmup, normalisation, noise multiplier, Dice formula, precision mode, or `pos_weight`. Do not add BS4. BS4 belongs to the legacy exploratory regime, does not address the larger-batch concern, and would dilute coverage of BS8/16/32.

## Optuna Design

Study naming:

- `STUDY_NAME=unet_paper_arch_noise_f1`
- base config: `configs/experiments/unet_paper_arch_noise_base.yaml`
- study DB: `results/classical/unet_paper_arch_noise_f1.db`
- best-params JSON: `results/classical/unet_paper_arch_noise_f1_best.json`

Run 45 trials with automatic single-best retraining disabled (`SKIP_RETRAIN=1` or `sweep.auto_retrain=false`). Allocate trials evenly: 15 BS8, 15 BS16, and 15 BS32. Batch size must be assigned by the balanced schedule, not by a free categorical TPE suggestion. TPE may sample the continuous parameters inside each stratum.

Use threshold-swept validation `val_f1` as the objective, not `val_dice@0.5`. Trial summaries and Optuna user attributes should record at least `val_f1`, `optimal_threshold`, `val_precision`, `val_recall`, and `batch_size`. Test metrics are inadmissible during the sweep.

Use MedianPruner only after enough startup coverage for all batch sizes. Preferred setting: `n_startup_trials=15` under a cyclic batch-size allocation, giving five startup trials per batch size, with at least 10 warmup epochs.

## Top-K Multi-Seed Retraining

After Optuna completes, select the top 5 completed trials ranked by Optuna validation `val_f1`. Retrain each selected trial for 75 epochs.

Seeds: `2804`, `1234`, `42`, `7`, `13`.

Config naming:

- `configs/experiments/restudy_topk/topk_t<N>_s<S>.yaml`
- experiment tag: `unet_paper_arch_noise_topk_t<N>_s<S>`

Each retrain gets a val-only threshold sweep. Selection is by mean threshold-swept validation `val_f1` across seeds. The primary winner is the highest mean. If candidates are within `0.001` validation F1 of the best mean, use the preregistered parsimony tie-breaker: prefer lower `batch_size`, then lower `learning_rate`, then lower trial number. Overlapping intervals must still be reported honestly; test metrics are inadmissible for tie-breaking.

## Final Selection Rule

Before inspecting or comparing any test metrics, append to `agents/report_notes.md` Section 3:

- date;
- the five per-seed validation F1 values for each top-K trial;
- mean validation F1 per trial;
- bootstrap CI per trial;
- selected restudy tag;
- explicit statement that selection used validation-only threshold-swept `val_f1`;
- confirmation that no test metrics were used for selection or tie-breaking.

Only after that note is written may the selected restudy checkpoint(s) be evaluated on the test and parity test sets.

## Final Evaluation After Selection

After validation-only selection is recorded:

1. Run threshold sweep on the selected restudy checkpoint for sampled-test metrics.
2. Run Hough post-processing at the selected validation threshold.
3. Rebuild `data/patches_test_full` at 528x528 if it has not already been rebuilt, run `compute_image_stats`, then run parity evaluation at the selected threshold.
4. Run parity Hough if the comparison table includes pre/post-Hough parity FNR.
5. Run cluster and patch bootstraps for the selected restudy checkpoint.
6. Regenerate prediction figures for the selected restudy checkpoint.
7. Update the final comparison table with historical comparators and the restudy winner as appropriate.

## Non-Actions

- Do not test batch sizes outside `{8, 16, 32}` under this protocol.
- Do not search dropout under this protocol.
- Do not search noise multiplier under this protocol.
- Do not use test metrics during Optuna, top-K selection, or tie-breaking.
- Do not delete historical comparator artefacts before the final comparison-table decision.
- Do not modify `data/`.
- Do not edit `report/thesis.tex` under this protocol unless explicitly requested.
