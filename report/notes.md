# Dissertation Report — Running Notes

A running log of decisions, results, methodology deviations, and open questions captured as the project progresses. The final 7000-word report and 1000-word executive summary will be drafted from this file; nothing in here is final prose, only material to draw on.

Word budget reminder: report ≤ 7000 words, executive summary ≤ 1000 words. Tables, figure captions, and appendices count toward the report budget; references and the autogeneration-tools declaration do not.

## How to use this file

- Add an entry under the relevant section whenever something is decided, surprises you, or would otherwise be forgotten by the time the report is written.
- Prefer short paragraphs with a date stamp. Quote numbers verbatim. Link to results files / figures by path.
- "Decisions" are choices that will be defended in the report. "Findings" are observed results. "Methodology deviations" are conscious departures from the target paper. "Open questions" are anything unresolved at the time of writing.
- Do not write polished prose here; that is the report's job.

---

## 1. Project framing

### Scope (as of 2026-05-14)

Primary aim: replicate the U-Net + Hough trail-detection pipeline from Stoppa et al. 2024 (A&A 692, A199 / arXiv 2407.19461) on the MeerLICHT-derived PNG subset. Primary extension: CNN-classifier two-stage detector. Secondary extensions (time permitting): robustness study, Attention U-Net, cross-dataset manual-masking test, demo UI.

Dropped from the original brief because the available data is PNG-only with no FITS headers:
- AITOFF / sky-coordinate projection plots.
- Horizon vs zenith angle analysis.
- Time-trend analysis as an extension. A single contextual EDA figure shows date coverage; no trend claim follows.

### Methodology decisions

| Decision | Choice | Rationale |
|---|---|---|
| Loss function | Combo BCE + Dice | Paper-faithful; reference in `asta/ASTA.py:121-140`. Replaces earlier BCEWithLogitsLoss(pos_weight=...). |
| Patch size | 512×512 | Power-of-two, U-Net-depth-4 friendly. Paper used 528; 512 is the nearest standard. Justified deviation. |
| Splits | Image-level 70/15/15, stratified by total trail-pixel count | Image-level avoids the same-frame patch leakage of the prior patch-level random_split. 70/15/15 leaves ~27 val and ~27 test images on the 178-image CSD3 set. |
| Positive:negative patch ratio | 1:3 default, in Optuna search space | 1:1 too aggressive (model never sees varied empty sky); 1:5+ keeps the imbalance problem. 1:3 is a standard medical/astro-imaging default. |
| Augmentation | Random h-flip, v-flip, k×90° rotations, optional shifts on positives | Paper: same set. Train-only. |
| Determinism | Seed torch/numpy/random + cudnn.deterministic + use_deterministic_algorithms | Where GPU non-determinism is unavoidable, 3-seed averaging is the fallback. |
| Threshold selection | PR sweep on val, F1-optimal | Paper used 0.58; ours is data-driven. |

### Reference paper headline metrics (to compare against)

Stoppa et al. 2024 reported on a 20,000-patch test set at threshold 0.58:
- Precision ≈ 0.94
- Recall ≈ 0.94
- False negative rate: 6.98% pre-Hough → 3.38% post-Hough (51% reduction)

Architecture: U-Net, ~485k parameters, 528×528 input, filters 8→128, LeakyReLU + dropout. The paper does not publish exact training hyperparameters (epochs, batch size, lr, optimizer); reproduction therefore requires tuning.

---

## 2. Decisions, findings, deviations (chronological)

### 2026-05-14 — supervisor meeting outcomes

- Dropped AITOFF, horizon-zenith, and time-trend analysis from extensions (PNG-only data).
- Primary extension is now CNN-classifier two-stage detector.
- Confirmed combo loss, image-level split, pre-disk patches, augmentation, determinism, Optuna, threshold sweep, data-efficiency study as the new replication critical path.
- Mask quality concern raised: some EDA overlays appear to under-cover the visible trail. Verification is now the first step before any training metric is reported.

### 2026-05-14 — EDA additions

- Added `parse_observation_datetime` to `src/data/indexing.py` (MeerLICHT filename → datetime).
- Added `compute_observation_date_dataframe` and `plot_observation_date_distribution` to `src/evaluation/eda.py` — produces a single contextual figure for dataset temporal span.
- Added `compute_mask_component_stats`, `plot_mask_thickness_distribution`, `plot_mask_inspection_grid` for the mask-quality verification flagged at the supervisor meeting.

---

## 3. Results snapshots

Reserve this section for headline numbers from completed runs. Each entry should give: date, run ID, config path, dataset split, headline metrics, link to the summary JSON.

(empty — no real training has run yet)

---

## 4. Open questions

Things to resolve before the report is written.

- Mask-quality response: dilate, re-annotate, or accept-and-document? Will be answered after running the new mask-inspection cells in `notebooks/01_dataset_eda.ipynb`.
- Eduardo's recommended cross-dataset reference paper / GitHub repo — still to be obtained. Required before the manual-masking extension can start.
- Determinism on A100: confirm via a 2-seed identical-run check that bitwise reproducibility is achievable; if not, document the 3-seed averaging policy and the operations that are non-deterministic.

---

## 5. Report outline (working draft)

Suggested section structure to support drafting later. Numbers are rough word budgets to keep within the 7000-word total.

1. Introduction and motivation — ~700 words
2. Related work and the target paper — ~700 words
3. Data — ~700 words (MeerLICHT subset, splits, augmentation, mask quality)
4. Methods — ~1500 words (U-Net architecture, combo loss, Hough post-processing, classifier two-stage extension, determinism and hyperparameter search)
5. Experiments and results — ~2000 words (replication metrics, data efficiency, threshold sweep, two-stage detector, robustness)
6. Discussion — ~800 words (limitations, deviations from paper, scientific implications)
7. Conclusions and future work — ~400 words

Executive summary (separate, ≤1000 words): self-contained one-page version, no figures, problem → approach → headline metric → significance.

---

## 6. Glossary of deviations from the target paper

Track every conscious deviation. The report's Discussion section will reference this list.

- Patch size 512 vs paper 528.
- Framework PyTorch vs paper Keras / TensorFlow.
- Exact architectural parameter counts may differ; documented once a final architecture is locked.
- Threshold selected from val-set PR sweep (not fixed at 0.58 as in paper) — defensible as a data-driven choice.
- Hough parameters tuned against held-out val rather than copied — explicit choice for reproducibility on different data scale.

---

## 7. Use of autogeneration tools (declaration draft)

A formal declaration accompanies the report and is required by the course handbook. Track usage here so the declaration can be assembled accurately at submission time.

- Claude Code (Anthropic) used throughout development for code review, planning, and documentation drafting. Specific touch points: PLAN.md and README.md edits, EDA helper drafting, test consolidation, dissertation-notes scaffolding. All committed code and prose was reviewed and approved by the author.
- ChatGPT / other LLMs: list here if used; otherwise mark "not used".
- Word count of declaration to be included on its front cover.
