========
Training
========

The U-Net training loop, the patch-classifier training entry point (validation-only, no
test loop), and the Optuna sweep driver with balanced batch-size allocation and a
threshold-swept validation-F1 objective.

.. autosummary::

   src.training.train_unet
   src.training.train_classifier
   src.training.sweep

.. automodule:: src.training.train_unet
.. automodule:: src.training.train_classifier
.. automodule:: src.training.sweep
