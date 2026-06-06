=====
Usage
=====

These snippets use the committed locked winner
(``results/checkpoints/model-best.pth``) and the public helpers under :doc:`api/index`.
They assume the package is installed (:doc:`getting_started`).

Run the U-Net on a single patch
===============================

:func:`src.models.loading.load_segmentation_model` loads a checkpoint, dispatches on the
recorded architecture, and returns the eval-mode model together with its normalisation
mode. :func:`src.data.transforms.normalise_tensor` applies that normalisation to a
``[0, 1]`` image tensor.

.. code-block:: python

   import torch
   import torchvision.transforms.functional as TF
   from PIL import Image

   from src.models.loading import load_segmentation_model
   from src.data.transforms import normalise_tensor

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model, normalisation = load_segmentation_model(
       "results/checkpoints/model-best.pth", device
   )

   with Image.open("data/patches/test/<patch>_image.png") as img:
       x = TF.pil_to_tensor(img.convert("L")).float() / 255.0
   x = normalise_tensor(x, mode=normalisation).unsqueeze(0).to(device)

   with torch.no_grad():
       prob = torch.sigmoid(model(x).squeeze()).cpu().numpy()

   mask = prob >= 0.45  # the locked operating threshold

.. note::

   The locked model uses ``full_image`` normalisation, i.e. a per-source-image z-score
   read from the manifest's ``image_mean`` / ``image_std`` columns. Passing those values
   to ``normalise_tensor`` reproduces the pipeline exactly; when they are omitted (as
   above) the helper falls back to a per-patch z-score, so the snippet stays
   self-contained.

Bridge gaps with the Hough transform
====================================

:func:`src.classical.hough_runner.run_hough_on_canvas` applies the main threshold and a
lower-threshold probabilistic Hough pass to a reconstructed per-image probability canvas,
returning the pre/post-Hough detection flags, pixel-coverage counts, and the boolean
canvases for patch-level aggregation.

.. code-block:: python

   from src.classical.hough_runner import run_hough_on_canvas

   result = run_hough_on_canvas(
       prob_canvas,                  # (H, W) float probabilities for one source image
       gt_canvas,                    # (H, W) uint8 ground-truth mask, > 0 on trail pixels
       threshold=0.45,
       hough_input_threshold=0.1,
       hough_threshold=50,
       min_line_length=100,
       max_line_gap=250,
       line_thickness=3,
   )

   print(result.detected_pre, result.detected_post)
   print(result.pixels_pre, result.pixels_post, result.gt_pixels)

Command-line entry points
=========================

Classical Hough baseline (local or CSD3):

.. code-block:: bash

   python -m src.classical.run_hough --config configs/experiments/hough_baseline.yaml

Two-stage detector (CNN classifier gate → U-Net → Hough):

.. code-block:: bash

   python -m src.inference.two_stage \
       --classifier results/checkpoints/classifier_base_latest.pth \
       --clf-threshold 0.18 \
       --unet results/checkpoints/model-best.pth \
       --unet-threshold 0.45 \
       --manifest data/patches/manifest.csv \
       --split test \
       --out results/classical/two_stage_demo.json

Thesis and extension figures (local, from committed JSON metrics):

.. code-block:: bash

   python -m scripts.figures.make_thesis_figures
   python -m scripts.figures.make_extension_figures
