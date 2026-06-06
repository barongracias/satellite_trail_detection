======
Models
======

The architecture-faithful U-Net, the Attention U-Net variant, the lightweight patch
classifier used by the two-stage gate, and the checkpoint loader that dispatches on the
recorded ``model_type``.

.. autosummary::

   src.models.unet
   src.models.attention_unet
   src.models.classifier
   src.models.loading

.. automodule:: src.models.unet
.. automodule:: src.models.attention_unet
.. automodule:: src.models.classifier
.. automodule:: src.models.loading
