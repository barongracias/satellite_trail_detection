===============
Getting Started
===============

Sources
=======

- **Code:** https://github.com/barongracias/satellite_trail_detection
- **Replicated paper:** Stoppa et al. 2024, *Automated detection of satellite trails
  in ground-based observations using U-Net and Hough transform*, A&A 692, A199
  (`arXiv:2407.19461 <https://arxiv.org/abs/2407.19461>`_).

The MeerLICHT image/mask pairs are collaboration data and are **not** redistributed
with the code; all splits are reproducible from the image-level split logic once the
pairs are in place. See the ``Data`` section of the project ``README`` for the data
policy. The blinded re-annotation audit also uses private sealed crops and masks
under ``data/gold/``; those inputs are not redistributed, while the evaluation code
and committed summary JSONs remain available in the repository.

Requirements
============

- Python **3.11+**.
- For training/inference: a CUDA GPU is recommended (CSD3 Ampere/A100 was used). The
  test suite, figure generation, and the classical Hough baseline run CPU-only.

Two dependency sets are provided:

- ``hpc-requirements.txt`` — a fully pinned lock of the CSD3 environment (CUDA 11.8
  PyTorch wheels).
- ``local-requirements.txt`` — the same dependency set with relaxed floors and no CUDA
  wheels, for CPU-only local work.

Installation
============

Local development (CPU — tests and figures)
-------------------------------------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r local-requirements.txt
   pip install -e .

HPC / CSD3 (CUDA training)
--------------------------

.. code-block:: bash

   pip install -r hpc-requirements.txt
   pip install -e .

Docker (portable CPU image)
---------------------------

The image builds a CPU environment from ``docker-requirements.txt`` and runs the
test suite as its default command:

.. code-block:: bash

   docker build -t satellite-trails .
   docker run --rm satellite-trails

Verify the install
==================

.. code-block:: bash

   pytest -q
   python -c "from src.config.constants import GLOBAL_SEED; print(GLOBAL_SEED)"  # -> 2804

CSD3 quickstart
===============

The **locked winner** is committed as ``results/checkpoints/model-best.pth``,
evaluated at threshold ``0.45`` with ``full_image`` normalisation (a byte-identical
copy of ``unet_paper_arch_noise_topk_t44_s2804_best.pth``, whose descriptive name
records the provenance). It is fixed downstream of selection: no
retraining, threshold tuning, or model reselection. The full Slurm command
sequence for rebuilding patches, rerunning the search, evaluating the locked
winner, and regenerating figures is in :doc:`csd3_reproduction`.

Building these docs
===================

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html

Heavy dependencies are mocked in ``docs/conf.py``, so the same build runs on
ReadTheDocs without the CUDA/scientific stack.
