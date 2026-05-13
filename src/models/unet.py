"""Minimal U-Net implementation for binary satellite trail segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """Apply two convolution, batch normalisation, and ReLU stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the paired convolution block."""
        return self.layers(inputs)


class DownBlock(nn.Module):
    """Downsample the feature map and apply a double convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply max pooling followed by feature extraction."""
        return self.layers(inputs)


class UpBlock(nn.Module):
    """Upsample decoder features and fuse them with the matching skip map."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample the decoder state and concatenate the skip features."""
        upsampled = self.up(inputs)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        merged = torch.cat([skip, upsampled], dim=1)
        return self.conv(merged)


class UNet(nn.Module):
    """A lightweight U-Net suitable for `512 x 512` binary segmentation patches."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        """
        Initialise the encoder-decoder network.

        Parameters
        ----------
        in_channels:
            Number of image channels.
        out_channels:
            Number of output channels. Use `1` for binary segmentation logits.
        base_channels:
            Channel width of the first encoder stage.
        """
        super().__init__()

        self.stem = DoubleConv(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = DownBlock(base_channels * 8, base_channels * 16)

        self.up1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

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
