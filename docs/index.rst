=========================
Satellite Trail Detection
=========================

Replication and diagnostic extension of the Stoppa et al. 2024 (A&A 692, A199)
satellite-trail detection pipeline on a 178-image MeerLICHT subset: a U-Net
segmenter produces a binary trail mask, a probabilistic Hough transform bridges
gaps in that mask, and post-hoc diagnostics localise the remaining strict
precision gap to boundary-scale label agreement plus a small over-firing
residual.

MPhil Data Intensive Science dissertation, University of Cambridge
(supervisor: Dr Eduardo Gonzalez-Solares).

The pipeline at a glance
------------------------

#. **Image-level split** of 178 image/mask pairs, stratified by trail-pixel quartile.
#. **Patch builder** — 528×528 patches at stride 528, written to disk with a manifest.
#. **U-Net training** with an Optuna sweep, multi-seed top-K retraining, and
   validation-only model selection.
#. **Hough post-processing** to recover gaps the U-Net misses.
#. **Error diagnosis** — boundary-tolerant scoring, FP decomposition, full-frame
   Hough verification, and a blinded single-author re-annotation audit.
#. **Model variants** — classifier gating, Attention U-Net, training-protocol
   pilots, ensembling, and probability-stratified Hough.

Quick links
-----------

- :doc:`getting_started` — install the package locally, on CSD3, or via Docker.
- :doc:`usage` — load the locked model and run inference, Hough, and the baseline.
- :doc:`reproduction_map` — map thesis sections, figures, and tables to the
  scripts, configs, and result artefacts that produced them.
- :doc:`csd3_reproduction` — rebuild the patch data and reproduce the locked CSD3
  training/evaluation sequence.
- :doc:`api/index` — the full ``src/`` API reference, grouped by sub-package.

Live demo
---------

`trail-scope <https://github.com/barongracias/trail-scope>`_ is a separate
FastAPI/Next.js demo of the locked detector. It is intended for interactive
inspection of U-Net + Hough overlays, not for reproducing the dissertation
experiments.

For the data policy and repository overview, see the project ``README`` on
`GitHub <https://github.com/barongracias/satellite_trail_detection>`_.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   getting_started
   usage
   reproduction_map
   csd3_reproduction
   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
