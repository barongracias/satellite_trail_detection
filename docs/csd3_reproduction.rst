=================
CSD3 Reproduction
=================

All long-running training and evaluation stages are Slurm jobs under ``slurm/``.
The sequence below rebuilds the patch dataset, runs the validation-only search,
reconstructs the locked winner evaluation, and regenerates the report-facing
prediction figures. The locked winner is committed as
``results/checkpoints/model-best.pth`` and is evaluated at threshold ``0.45``
with ``full_image`` normalisation. Downstream analyses are inference-only: no
retraining, threshold reselection, or model reselection.

Build patches and calibration artefacts
=======================================

.. code-block:: bash

   rm -rf data/patches
   jid1=$(sbatch --parsable slurm/build_patches_cpu.sbatch)
   jid2=$(MANIFEST=data/patches/manifest.csv \
          sbatch --parsable --dependency=afterok:$jid1 slurm/compute_image_stats_cpu.sbatch)
   jid3=$(MANIFEST=data/patches/manifest.csv \
          sbatch --parsable --dependency=afterok:$jid2 slurm/compute_background_noise_stats_cpu.sbatch)
   jid4=$(MANIFEST=data/patches/manifest.csv \
          sbatch --parsable --dependency=afterok:$jid3 slurm/audit_manifest_cpu.sbatch)

Run the validation-only search
==============================

.. code-block:: bash

   STUDY_NAME=unet_paper_arch_noise_f1 \
   CONFIG=configs/experiments/unet_paper_arch_noise_base.yaml \
   N_TRIALS=45 SKIP_RETRAIN=1 \
     sbatch --dependency=afterok:$jid4 slurm/optuna_sweep.sbatch

Retrain the selected top-K configuration
========================================

Per-trial and per-seed retrain configs live in
``configs/experiments/restudy_topk/``. The canonical locked winner is trial 44,
seed 2804.

.. code-block:: bash

   CONFIG=configs/experiments/restudy_topk/topk_t44_s2804.yaml \
     sbatch slurm/train_unet_ampere_long.sbatch

Recreate the locked evaluation
==============================

Do not set ``THRESHOLD`` on the threshold-sweep command. That would skip the
validation PR sweep and overwrite
``threshold_sweep_winner_t44_s2804.json`` with fixed-threshold output.

.. code-block:: bash

   CHECKPOINT=results/checkpoints/model-best.pth \
   TAG=winner_t44_s2804 \
     sbatch slurm/threshold_sweep.sbatch

   CHECKPOINT=results/checkpoints/model-best.pth \
   THRESHOLD=0.45 \
   OUT=results/classical/hough_postprocess_winner_t44_s2804.json \
     sbatch slurm/hough_postprocess.sbatch

Regenerate prediction figures
=============================

.. code-block:: bash

   CHECKPOINT=results/checkpoints/model-best.pth \
   THRESHOLD=0.45 \
   HOUGH_JSON=results/classical/hough_postprocess_winner_t44_s2804.json \
   TAG=winner_t44_s2804 \
     sbatch slurm/visualise_predictions.sbatch

Gold audit
==========

The blinded re-annotation audit is evaluation-only and uses private sealed crops
and masks under ``data/gold/``. Those inputs are not redistributed. The committed
summary artefacts are ``results/classical/gold_audit_eval.json`` and
``results/classical/gold_audit_bootstrap.json``.
