# Re-annotation Audit Preregistration (M9.4)

Date: 2026-06-13
Status: active M9.4 protocol. Preregistered **before any crop is annotated and
before `gold_audit_eval.py` is run for real**. The blinded crops, sealed
mapping, and scoring scaffold already exist (commit `000c6e2`); this document
fixes the annotation rules, analyses, and decision rules in advance so the
interpretation cannot be chosen after seeing the numbers.

## Motivation

The thesis locates the strict precision gap against Stoppa et al. (2024) almost
entirely within ±1 px of the labelled trail boundaries, but phrases the cause
as "consistent with" boundary-scale annotation disagreement because a symmetric
tolerance cannot separate two mechanisms that produce the identical signature:
masks drawn systematically thinner than the visible trail, versus a model that
over-paints by ~1 px. This audit resolves that ambiguity by producing an
independent, blinded gold re-annotation of a stratified crop sample and
comparing the **original** masks, the **gold** masks, and the **model**
predictions directly. It also supplies the inter-annotator floor that
calibrates the NSD tolerance τ used in M9.3.

The audit measures the residual error of the **locked** pipeline only
(checkpoint `model-best.pth` = `unet_paper_arch_noise_topk_t44_s2804`, threshold
0.45, `full_image` normalisation). Nothing here re-selects, re-trains, or
re-thresholds anything.

## Scope and standing constraints

- **Evaluation-only.** Gold masks are never used for training, threshold
  selection, or to replace the original masks. The locked predictions are
  scored as-is.
- **Display-space caveat (stated now, not in rebuttal).** Gold masks are drawn
  on the same 8-bit display renders as the originals. The audit measures
  annotation **convention and granularity**, not flux-level truth. It cannot,
  and does not claim to, establish ground truth in calibrated units.
- **Data policy.** Crops and gold masks are derivatives of MeerLICHT
  collaboration data: they live under `data/gold/` (gitignored) on local/CSD3
  only and are never committed or redistributed. The scoring script and the
  aggregate result JSON (`results/classical/gold_audit_eval.json`, no raw
  pixels) are committable.

## Materials (already generated, commit `000c6e2`)

- **64 blinded crops**, 528×528, raw 8-bit pixels only, neutral shuffled names
  `c001.png … c064.png`, in `data/gold/audit_crops/`. Strata: 30 interior /
  15 endpoint / 12 FP / 7 decoy (`results/classical/audit_crops_summary.json`).
  Interior sample spans 20 distinct test images; FP/decoy drawn from 67 fired /
  2,549 quiet mask-empty test patches.
- **Sealed crop→frame mapping** at `data/gold/sealed_crop_manifest.json`
  (gitignored; SHA-256 `9fb33fb14a6fd972e2f27c21d1274ad28d40590ad7e324d7ef0a7b44f853ac4b`
  recorded in the public summary). **The annotator must not open this file
  until annotation is complete.** Opening it before annotation voids the
  blinding and the audit.
- **Scoring scaffold** `scripts/evaluation/gold_audit_eval.py` (+ 9 tests in
  `tests/test_gold_audit.py`), runs only after annotation.

## Annotation rules (frozen)

### Blinding

Annotate on the raw crop PNGs only. The original masks and the model
predictions are never displayed during annotation. Crops are already presented
under neutral, shuffled names. The author is the primary annotator and has seen
many predictions during the project; blinding, name shuffling, and (for the
self-annotation fallback) a time gap mitigate but do not eliminate anchoring —
this is disclosed in the paper and is the main reason a second annotator is
sought.

### Width rule — the object of study, so it is fixed verbatim

> Mask every pixel where the trail is visibly brighter than the local
> background, judged at a fixed 400 % zoom under the crop's native 8-bit display
> (no per-crop contrast adjustment; if a single fixed display gamma is used it
> is applied identically to every crop and recorded here). At an ambiguous edge
> pixel, **include** it.

The rule must not change once annotation begins; changing it mid-stream
invalidates the width comparison that is the audit's core discriminator.

### Tooling

Hard-edged 1-px **pencil**, anti-aliasing **off** (Fiji/LABKIT, GIMP pencil, or
napari labels layer — never a soft brush). Straight trails are drawn with
shift-click straight strokes; widen with parallel strokes where the trail is
visibly wider. Export a binary PNG per crop on the identical 528×528 grid, same
name as the crop. The scoring script asserts at most two distinct pixel values
per PNG and rejects anti-aliased exports, so tool choice is verified
automatically.

### Per-stratum annotation task

- **Interior / endpoint crops:** draw the trail under the width rule. If no
  trail is visible where one is expected, leave the mask empty and note the
  crop name under the **"not visible"** category (a label-error datum, excluded
  from the boundary-width statistics).
- **FP / decoy crops:** an existence judgement recorded in a separate CSV
  (`crop_name,verdict`) with verdict ∈ {`trail`, `no_trail`, `uncertain`}.
  `uncertain` crops are reported separately and never scored. Decoys are scored
  only as calibration controls: an annotator who marks decoys as `trail` is
  flagged as trigger-happy; decoys marked `no_trail` confirm the existence
  judgements are meaningful. The annotator does not know which crops are decoys.

### Disambiguating trails from artefacts

A **blinded context view is allowed** for disambiguation (the original ASTA
annotators had frame context, so using it keeps the gold comparable): each crop
`cNNN.png` has a companion `cNNN_context.png`, a larger window (default 1584 px,
3× the crop) centred on the same point, raw pixels only — no frame identity, no
stratum, no mask, no prediction. The annotator may consult it but annotates only
on the 528 px primary crop. This resolves the apparent tension with the sealed
mapping: context is available without unsealing frame coordinates. Opening the
sealed manifest itself remains forbidden, and the original masks and model
predictions remain off-screen throughout. Discriminators:

| Feature | Signature |
|---|---|
| Satellite trail | Long, straight, roughly constant width; crosses much/all of the frame; brightness varies smoothly (sometimes periodic from tumbling) |
| Diffraction spike | Anchored on a bright star, radial, symmetric pairs/cross |
| Bad column / bleed | Exactly aligned with the pixel grid (perfectly vertical/horizontal) |
| Cosmic ray | Short (≲50 px), sharp-edged, no extended continuation |

## Annotators

- **Primary:** the author, on all 64 crops.
- **Second annotator (preferred):** a supervisor or colleague on a ≥15-crop
  subsample spanning all four strata, given the identical crop folder, this
  written width rule, and a ~15-minute briefing using example trails **not** in
  the sample. Crop order is randomised independently per annotator. Session
  dates/times are logged so the paper can state annotation was blind,
  order-randomised, and time-bounded.
- **Fallback (only if no second annotator within the M9 week-2 window):** the
  author re-annotates the ≥15-crop subsample ≥5 days later with order
  reshuffled and the first-pass masks hidden. This is weaker (intra-annotator,
  anchoring-prone) and the paper must say so explicitly.

## Preregistered analyses

Run by `gold_audit_eval.py` after annotation; tolerances {0, ±1 px}, Chebyshev
neighbourhood (MORPH_RECT), matching the thesis boundary-tolerant metric.
Intervals by component-level bootstrap, cross-checked by image-level bootstrap.

- **(a) Label-noise floor.** Original-vs-gold strict and ±1 px P/R/F1 on the
  interior+endpoint crops. This is how far the *original labels themselves* sit
  from a careful independent annotation.
- **(b) Detector vs labels.** Model-vs-gold and model-vs-original on the same
  crops, side by side. Tests whether the locked detector agrees with careful
  annotation *better than the original labels do*.
- **(c) Width attribution (the discriminator).** Per-component median
  perpendicular widths of original mask, gold mask, and prediction (skeleton ×
  2 × Euclidean distance transform). Separates thin-labels from model
  over-paint.
- **(d) τ calibration.** Inter-annotator (or intra-annotator fallback)
  agreement at strict and ±1 px, feeding the NSD τ choice in M9.3.
- **Existence controls.** FP-crop verdict breakdown (trail / no_trail /
  uncertain) and decoy false-alarm rate.

## Preregistered decision rules (fixed before results)

Let `F1_orig_gold` = original-vs-gold strict F1 and `F1_model_orig` =
model-vs-original strict F1, **both computed on the same audited interior+endpoint
crops** (not the full-test-set headline number — the crop sample is
contrast-stratified, so comparing it to the whole-test F1 would not be
apples-to-apples). Both come from the scorer's crop-restricted aggregate. Use
median widths `w_orig`, `w_gold`, `w_pred` over the same crops.

1. **Label-noise floor dominates** if `F1_orig_gold` is no greater than
   `F1_model_orig + 0.02`, i.e. the original labels disagree with careful
   annotation by as much as (or more than) the model does. Then the strict
   precision gap is largely an annotation-floor effect.
2. **Thin-label mechanism** if `w_gold > w_orig` by ≥ 1 px (median) and
   `w_pred` ≈ `w_gold` (within 1 px): the model tracks the true trail width and
   the original masks are too thin.
3. **Model over-paint mechanism** if `w_gold ≈ w_orig` (within 1 px) and
   `w_pred > w_gold` by ≥ 1 px: the labels are fine and the model is genuinely
   wider.
4. **Mixed/indeterminate** otherwise: report both readings; the audit narrows
   but does not fully resolve the mechanism.
5. Any outcome is publishable. If (a) and (b) together show the detector agrees
   with gold better than the originals do, that is the strongest result; if not,
   the honest null still bounds the label-noise floor. No outcome triggers any
   re-selection, re-training, or re-thresholding of the locked pipeline.

## Caveats carried forward

- Small n (≈45 interior+endpoint components from 20 images): report
  distributions and bootstrap intervals, not point estimates alone; one odd
  component can move a percentage.
- The ±1 px tolerance is Chebyshev here (matching the thesis); NSD@τ in M9.3 is
  Euclidean, so τ calibration from (d) must respect that Chebyshev-1 ⊂
  Euclidean-2 when it feeds M9.3.
- Display-space and anchoring caveats above are repeated in the paper's
  limitations, not buried here.

## Execution checklist

1. Commit this document (done = preregistration sealed).
2. Annotate `data/gold/audit_crops/c001–c064.png` per the rules above; export
   binary PNGs to `data/gold/gold_masks/`; record FP/decoy verdicts CSV and the
   "not visible" list. Do **not** open the sealed manifest.
3. Second annotator (or fallback) on the ≥15-crop subsample.
4. `python scripts/evaluation/gold_audit_eval.py --annotation_dir
   data/gold/gold_masks --verdicts_csv <verdicts.csv>` → writes
   `results/classical/gold_audit_eval.json`.
5. Record results and the decision-rule outcome in `agents/report_notes.md`,
   then write up.
