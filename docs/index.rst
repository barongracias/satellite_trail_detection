Satellite Trail Detection
=========================

Replication and extension of the Stoppa et al. 2024 (A&A 692, A199) satellite-trail
detection pipeline on a MeerLICHT subset: a U-Net segmenter produces a binary trail
mask, a classical Hough transform bridges gaps in that mask, and a suite of extensions
characterises and improves the detector.

This site documents the public API under ``src/``. For installation, the CSD3
quickstart, and the reproduction commands, see the project ``README``.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
