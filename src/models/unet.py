"""Minimal U-Net implementation for binary satellite trail segmentation.

Architecture follows Stoppa et al. 2024: filters 8→128, LeakyReLU, spatial dropout.
Dropout is applied in the bottleneck and the three deepest decoder blocks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """Two conv + LeakyReLU stages with optional spatial dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DownBlock(nn.Module):
    """Max-pool downsample followed by a double-conv block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(inputs))


class UpBlock(nn.Module):
    """Transposed-conv upsample, skip-connection fusion, and double-conv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(inputs)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([skip, upsampled], dim=1))


class UNet(nn.Module):
    """U-Net for 512×512 binary segmentation (paper-faithful: 8→128 filters, LeakyReLU, Dropout)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 8,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        B = base_channels
        # Encoder — no dropout (preserve gradient flow through skip connections)
        self.stem = DoubleConv(in_channels, B)
        self.down1 = DownBlock(B, B * 2)
        self.down2 = DownBlock(B * 2, B * 4)
        self.down3 = DownBlock(B * 4, B * 8)
        # Bottleneck + decoder — dropout applied here
        self.bottleneck = DownBlock(B * 8, B * 16, dropout_rate=dropout_rate)
        self.up1 = UpBlock(B * 16, B * 8, B * 8, dropout_rate=dropout_rate)
        self.up2 = UpBlock(B * 8, B * 4, B * 4, dropout_rate=dropout_rate)
        self.up3 = UpBlock(B * 4, B * 2, B * 2, dropout_rate=dropout_rate)
        self.up4 = UpBlock(B * 2, B, B)
        self.head = nn.Conv2d(B, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return per-pixel segmentation logits."""
        enc1 = self.stem(inputs)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        bottleneck = self.bottleneck(enc4)
        dec1 = self.up1(bottleneck, enc4)
        dec2 = self.up2(dec1, enc3)
        dec3 = self.up3(dec2, enc2)
        dec4 = self.up4(dec3, enc1)
        return self.head(dec4)
