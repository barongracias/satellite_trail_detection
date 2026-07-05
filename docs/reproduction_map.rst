=======================
Thesis Reproduction Map
=======================

This page maps the dissertation claims, figures, and tables to the top-level
scripts, configuration files, and result artefacts that produced them. It is a
provenance map, not a full Python call graph: helper modules and imported
library code are intentionally omitted unless they define a thesis-facing
interface.

For the exact CSD3 job sequence used to rebuild patches, rerun the locked
training/evaluation path, and regenerate prediction figures, see
:doc:`csd3_reproduction`.

Chronological spine
===================

#. Build the patch dataset from the private MeerLICHT image/mask pairs with
   ``scripts/data/build_patch_dataset.py``. The image-level split logic lives in
   ``src/data/splits.py``.
#. Attach image and background statistics with
   ``scripts/data/compute_image_stats.py`` and
   ``scripts/data/compute_background_noise_stats.py``; audit the manifest with
   ``scripts/audit/audit_manifest.py``.
#. Run the Optuna search with ``src/training/sweep.py`` and
   ``configs/experiments/unet_paper_arch_noise_base.yaml``.
#. Generate and retrain the selected top-K configurations with
   ``scripts/sweep/generate_restudy_topk_configs.py`` and
   ``src/training/train_unet.py``. The locked winner is
   ``configs/experiments/restudy_topk/topk_t44_s2804.yaml`` and is committed as
   ``results/checkpoints/model-best.pth``.
#. Select the operating threshold and compute locked segmentation metrics with
   ``scripts/evaluation/threshold_sweep.py``; quantify uncertainty with
   ``scripts/evaluation/bootstrap_metrics.py``.
#. Run post-hoc Hough, geometry, false-positive, and boundary diagnostics with
   the evaluation scripts under ``scripts/evaluation/``.
#. Run optional variants and pilots under their stated protocols; these do not
   replace the locked model.
#. Regenerate thesis/report figures from committed JSON metrics and private
   inputs where required with the scripts under ``scripts/figures/``.

Component map
=============

.. list-table::
   :header-rows: 1
   :widths: 22 32 27 19

   * - Thesis component
     - Main entry points
     - Primary outputs
     - Thesis use and caveats
   * - Data split and patch construction
     - ``src/data/splits.py``;
       ``scripts/data/build_patch_dataset.py``;
       ``slurm/build_patches_cpu.sbatch``
     - ``data/patches/manifest.csv`` and train/validation/test patch folders
     - Chapters 2 and 4. Requires the private MeerLICHT image/mask pairs.
   * - Calibration and manifest checks
     - ``scripts/data/compute_image_stats.py``;
       ``scripts/data/compute_background_noise_stats.py``;
       ``scripts/audit/audit_manifest.py``
     - Per-image normalisation fields in the patch manifest;
       ``results/classical/background_noise_calibration.json``;
       manifest audit outputs
     - Supports Chapter 4 normalisation and the display-contrast diagnostics.
   * - Optuna search and locked U-Net selection
     - ``src/training/sweep.py``;
       ``src/training/train_unet.py``;
       ``configs/experiments/unet_paper_arch_noise_base.yaml``;
       ``slurm/optuna_sweep.sbatch``
     - Optuna study database and trial summaries under ``results/classical/``
       and ``results/checkpoints/``
     - Section 5.2. Validation-only selection; not test-set tuning.
   * - Top-K retraining and committed locked winner
     - ``scripts/sweep/generate_restudy_topk_configs.py``;
       ``src/training/train_unet.py``;
       ``configs/experiments/restudy_topk/topk_t44_s2804.yaml``;
       ``slurm/train_unet_ampere_long.sbatch``
     - ``results/checkpoints/model-best.pth``;
       ``results/checkpoints/unet_paper_arch_noise_topk_t44_s2804_summary.json``
     - Sections 5.1--5.2. The locked winner is trial 44, seed 2804, evaluated at
       threshold ``0.45``.
   * - Threshold sweeps, segmentation metrics, and uncertainty
     - ``scripts/evaluation/threshold_sweep.py``;
       ``scripts/evaluation/bootstrap_metrics.py``;
       ``slurm/threshold_sweep.sbatch``;
       ``slurm/bootstrap_metrics.sbatch``
     - ``results/classical/threshold_sweep_winner_t44_s*.json``;
       ``results/classical/bootstrap_winner_t44_s*.json``
     - Table 5.1 and Section 5.4. Cluster bootstrap is the primary uncertainty
       estimate; patch bootstrap is diagnostic only.
   * - Hough post-processing and parity comparison
     - ``scripts/evaluation/hough_postprocess.py``;
       ``scripts/evaluation/hough_fullframe.py``;
       ``src/classical/run_hough.py``;
       ``configs/experiments/hough_baseline.yaml``
     - ``results/classical/hough_postprocess_winner_t44_s*_parity.json``;
       ``results/classical/hough_fullframe_winner_t44_s2804.json``;
       ``results/classical/hough_summary.json``
     - Section 5.3 and Discussion. Completeness comparison only; pixel
       precision remains sensitive to line drawing and prevalence.
   * - Prediction overlays and DECam cold check
     - ``scripts/figures/visualise_predictions.py``;
       ``scripts/figures/decam_cold_inference.py``
     - ``results/figures/predictions/overlay_grid_winner_t44_s2804.png``;
       ``results/figures/decam_cold_inference_montage.pdf``;
       ``results/classical/decam_cold_inference.json``
     - Section 5.5. DECam has no masks here, so the result is qualitative only.
   * - Boundary and false-positive diagnostics
     - ``scripts/evaluation/fp_decompose.py``;
       ``scripts/evaluation/boundary_tolerant_eval.py``;
       ``scripts/evaluation/geometry_eval.py``;
       ``scripts/analysis/aggregate_diagnostics_multiseed.py``
     - ``results/classical/fp_decomposition_*.json``;
       ``results/classical/boundary_tolerant_*.json``;
       ``results/classical/geometry_eval_*.json``;
       ``results/classical/diagnostics_multiseed_summary.json``
     - Sections 5.6.1--5.6.3 and Appendix A. These place much of the error
       near labelled boundaries but do not by themselves prove label error.
   * - Display-contrast and topology checks
     - ``scripts/figures/faint_streak_analysis.py``;
       ``scripts/evaluation/geometry_eval.py``
     - ``results/classical/faint_streak_t44_s2804.json``;
       ``results/figures/faint_streak_*.pdf``;
       ``results/figures/fp_intensity.pdf``
     - Sections 5.6.4--5.6.5 and Appendix A. Display-space proxies only; not
       calibrated flux or SNR.
   * - Blinded single-author re-annotation audit
     - ``scripts/evaluation/export_audit_crops.py``;
       ``scripts/evaluation/validate_audit_annotations.py``;
       ``scripts/evaluation/gold_audit_eval.py``;
       ``scripts/analysis/gold_audit_bootstrap.py``;
       ``scripts/figures/make_gold_audit_figure.py``;
       ``scripts/figures/make_gold_audit_annotation_examples.py``
     - ``results/classical/gold_audit_eval.json``;
       ``results/classical/gold_audit_bootstrap.json``;
       ``results/figures/supp_3_gold_audit_overlays.pdf``;
       ``results/figures/supp_4_gold_audit_annotation_examples.pdf``
     - Section 5.6.6 and Appendix A. Requires private ``data/gold/`` inputs; the
       audit is directional, not an independent multi-annotator ground-truth mask.
   * - Classifier-gated two-stage detector
     - ``src/training/train_classifier.py``;
       ``src/inference/two_stage.py``;
       ``configs/experiments/classifier_base.yaml``;
       ``slurm/two_stage.sbatch``
     - ``results/classical/two_stage_t44_s2804.json``
     - Section 5.7.1. The classifier checkpoint is intentionally off-git unless
       retrained; this is a descriptive operating-characteristic pilot.
   * - Attention U-Net restudy
     - ``src/training/sweep.py``;
       ``src/training/train_unet.py``;
       ``configs/experiments/attention_topk/topk_t7_s2804.yaml``;
       ``scripts/sweep/aggregate_restudy.py``
     - ``results/classical/attention_topk_summary.json``;
       ``results/classical/bootstrap_attn_winner_t7_s*.json``;
       ``results/figures/predictions/fp_fn_gallery_attn_winner_t7_s2804.png``
     - Section 5.7.2. Matched-protocol evidence; not a new selected replacement.
   * - Training and inference sensitivity pilots
     - ``configs/experiments/unet_data_efficiency_f*.yaml``;
       ``configs/experiments/unet_hard_negative_t44_s2804.yaml``;
       ``configs/experiments/unet_soft_dilated_t44_s2804.yaml``;
       ``scripts/evaluation/ensemble_eval.py``;
       ``scripts/evaluation/hough_prob_stratified.py``
     - ``results/classical/data_efficiency_curve.json``;
       ``results/classical/hard_negative_t44_s2804_summary.json``;
       ``results/classical/soft_dilated_t44_s2804_summary.json``;
       ``results/classical/ensemble_t44.json``;
       ``results/classical/hough_prob_stratified_t44_s2804.json``
     - Sections 5.8.1--5.8.5. These are pilots or descriptive checks under
       stated protocols, not a revised locked model.
   * - Thesis and report figure generation
     - ``scripts/figures/make_thesis_figures.py``;
       ``scripts/figures/make_extension_figures.py``;
       ``scripts/figures/make_diagnostics_supplementary.py``;
       ``scripts/figures/make_unet_diagram.py``
     - ``results/figures/thesis_*.pdf``;
       ``results/figures/ext_*.pdf``;
       ``results/figures/supp_*.pdf``;
       ``results/figures/unet_architecture.pdf``
     - Figures in Chapters 2, 4, 5, Appendix A, and the executive summary. Most
       run locally from committed JSON metrics; gold-audit overlays require
       private audit inputs for native regeneration.

Scope notes
===========

- The committed locked U-Net checkpoint is sufficient for inference and most
  metric/figure regeneration once the private patch manifest is available.
- The MeerLICHT image/mask pairs and blinded audit crops are collaboration or
  private data and are not redistributed.
- The two-stage classifier path requires either retraining or an off-git
  classifier checkpoint with recorded provenance.
- Exploratory files such as ``configs/experiments/hough_tuned.yaml`` and
  ``scripts/evaluation/tune_hough.py`` are provenance artefacts; they are not
  part of the locked reproduction path.
