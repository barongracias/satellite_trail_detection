"""Determinism harness for reproducible PyTorch experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.config.constants import GLOBAL_SEED


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    """Seed all random number generators for reproducible training.

    Sets Python, NumPy, PyTorch CPU and CUDA seeds and enables cuDNN
    deterministic mode. Call once at the start of any training entry point,
    before model construction and data loading.

    Parameters
    ----------
    seed:
        Integer seed. Defaults to ``GLOBAL_SEED`` (2804).

    Notes
    -----
    ``torch.use_deterministic_algorithms(True)`` requires the environment
    variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` on CUDA >= 10.2, which this
    function sets automatically. A small number of PyTorch operations have no
    deterministic implementation on GPU; ``warn_only=True`` logs a warning
    rather than raising, so training is not blocked. Where bitwise
    reproducibility is critical, use 3-seed averaging instead.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
