=========================
Satellite Trail Detection
=========================

Replication and extension of the Stoppa et al. 2024 (A&A 692, A199) satellite-trail
detection pipeline on a 178-image MeerLICHT subset: a U-Net segmenter produces a
binary trail mask, a classical Hough transform bridges gaps in that mask, and a suite
of extensions characterises and improves the detector.

MPhil Data Intensive Science dissertation, University of Cambridge
(supervisor: Dr Eduardo Gonzalez-Solares).

The pipeline at a glance
------------------------

#. **Image-level split** of 178 image/mask pairs, stratified by trail-pixel quartile.
#. **Patch builder** — 528×528 patches at stride 528, written to disk with a manifest.
#. **U-Net training** with an Optuna sweep, multi-seed top-K retraining, and
   validation-only model selection.
#. **Hough post-processing** to recover gaps the U-Net misses.
#. **Classical Hough baseline** as a non-learned reference point.

Quick links
-----------

- :doc:`getting_started` — install the package locally, on CSD3, or via Docker.
- :doc:`usage` — load the locked model and run inference, Hough, and the baseline.
- :doc:`api/index` — the full ``src/`` API reference, grouped by sub-package.

For the end-to-end CSD3 reproduction commands and the data policy, see the project
``README`` on `GitHub <https://github.com/barongracias/satellite_trail_detection>`_.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   getting_started
   usage
   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
