"""Shared image helpers for the figure and visualisation scripts.

These utilities are deliberately dependency-light (NumPy + OpenCV) so they can be
reused across the locked-winner figure scripts without pulling in matplotlib or
torch.
"""

from __future__ import annotations

import cv2
import numpy as np


def resize_for_display(arr: np.ndarray, max_dim: int = 1500, nearest: bool = False) -> np.ndarray:
    """Downscale a 2-D array so its largest side is at most ``max_dim`` pixels.

    Arrays already within the limit are returned unchanged. Used purely to keep
    figure file sizes manageable; the original arrays drive all metrics.

    Parameters
    ----------
    arr:
        Two-dimensional image or mask array.
    max_dim:
        Maximum allowed height or width after resizing.
    nearest:
        Use nearest-neighbour interpolation (for binary masks) instead of the
        area-averaging default (for continuous images).

    Returns
    -------
    numpy.ndarray
        The resized array as ``float32``, or the input unchanged when no
        downscaling is required.
    """
    h, w = arr.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return arr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(arr.astype(np.float32), (new_w, new_h), interpolation=interpolation)


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask keeping only the largest 8-connected component.

    Empty or single-component masks are returned unchanged (as ``bool``).

    Parameters
    ----------
    mask:
        Two-dimensional array interpreted as a boolean mask.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the largest connected component.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))
