"""Lightweight CNN classifier for satellite-trail patch pre-filtering."""

from __future__ import annotations

import torch
from torch import nn


class PatchClassifier(nn.Module):
    """Binary patch classifier returning one logit per input patch."""

    def __init__(self, in_channels: int = 1, channels: tuple[int, ...] = (16, 32, 64, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_channels = in_channels
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            prev_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(prev_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return raw classifier logits with shape ``(batch,)``."""
        features = self.encoder(inputs)
        pooled = self.pool(features).flatten(1)
        return self.head(pooled).squeeze(1)
