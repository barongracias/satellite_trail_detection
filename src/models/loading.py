"""Single source of truth for loading a trained segmentation checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.models.attention_unet import AttentionUNet
from src.models.unet import UNet


def load_segmentation_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, str]:
    """Load a U-Net / Attention U-Net checkpoint, dispatching on ``config.model_type``.

    Returns the eval-mode model on ``device`` and its normalisation mode. Centralises
    the loader so the ``model_type`` dispatch lives in exactly one place (a prior
    bug had it hardcoded to ``UNet`` in several scripts).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cls = AttentionUNet if cfg.get("model_type", "unet") == "attention_unet" else UNet
    model = model_cls(base_channels=cfg.get("base_channels", 8)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg.get("normalisation", "fixed")
