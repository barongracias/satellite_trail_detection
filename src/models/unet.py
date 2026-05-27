"""Architecture-faithful U-Net for binary satellite trail segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


DROPOUT_RATES: tuple[float, ...] = (0.1, 0.1, 0.2, 0.2, 0.3, 0.2, 0.2, 0.1, 0.1)


class DoubleConv(nn.Module):
    """Two conv + LeakyReLU stages with ASTA-style in-block dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DownBlock(nn.Module):
    """Average-pool downsample followed by a double-conv block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
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
        dropout_rate: float,
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
    """U-Net reconstructed from the released ASTA Keras model config."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 8,
    ) -> None:
        super().__init__()
        B = base_channels
        rates = DROPOUT_RATES

        self.stem = DoubleConv(in_channels, B, dropout_rate=rates[0])
        self.down1 = DownBlock(B, B * 2, dropout_rate=rates[1])
        self.down2 = DownBlock(B * 2, B * 4, dropout_rate=rates[2])
        self.down3 = DownBlock(B * 4, B * 8, dropout_rate=rates[3])
        self.bottleneck = DownBlock(B * 8, B * 16, dropout_rate=rates[4])
        self.up1 = UpBlock(B * 16, B * 8, B * 8, dropout_rate=rates[5])
        self.up2 = UpBlock(B * 8, B * 4, B * 4, dropout_rate=rates[6])
        self.up3 = UpBlock(B * 4, B * 2, B * 2, dropout_rate=rates[7])
        self.up4 = UpBlock(B * 2, B, B, dropout_rate=rates[8])
        self.head = nn.Conv2d(B, out_channels, kernel_size=1)

        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Apply Keras-compatible initialisers from the recovered ASTA config."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (3, 3):
                    # Keras he_normal uses gain=sqrt(2), matching relu here even
                    # though the actual activation is LeakyReLU(alpha=0.3).
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
