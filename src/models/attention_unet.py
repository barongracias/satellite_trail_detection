"""Attention-gated U-Net for binary satellite trail segmentation.

Identical skeleton to ``unet.py`` (same DoubleConv / DownBlock blocks, AveragePool2d,
LeakyReLU slope 0.3, element-wise Dropout between the two convolutions in every block
with the same fixed rate schedule, He-normal 3×3 conv init / Glorot-uniform
transposed+final conv init, no BatchNorm, logits out).  The only structural addition
is an :class:`AttentionGate` on each of the four skip connections before the concat in
the decoder, following the additive attention formulation of Oktay et al. (2018).

Reference: Oktay, O. et al. (2018). Attention U-Net: Learning where to look for the
pancreas. MIDL. arXiv:1804.03999.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.unet import DROPOUT_RATES, DoubleConv, DownBlock


class AttentionGate(nn.Module):
    """Additive soft attention gate for one skip connection.

    Parameters
    ----------
    x_channels:
        Number of channels in the encoder skip feature ``x``.
    g_channels:
        Number of channels in the decoder gating signal ``g``
        (at the same spatial resolution as ``x`` after upsampling).
    int_channels:
        Intermediate bottleneck size.  Defaults to ``x_channels // 2``.
    """

    def __init__(
        self,
        x_channels: int,
        g_channels: int,
        int_channels: int | None = None,
    ) -> None:
        super().__init__()
        if int_channels is None:
            int_channels = max(1, x_channels // 2)
        self.theta_x = nn.Conv2d(x_channels, int_channels, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(g_channels, int_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(int_channels, 1, kernel_size=1, bias=True)
        # Initialised centrally by AttentionUNet._initialise_weights (1×1 Glorot branch).

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """Return the attention-gated skip feature ``alpha * x``.

        Parameters
        ----------
        x:
            Encoder skip feature ``(B, C_x, H, W)``.
        g:
            Decoder gating signal ``(B, C_g, H_g, W_g)``.  Up-sampled to
            match ``x`` spatially if dimensions differ.
        """
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.relu(self.theta_x(x) + self.phi_g(g), inplace=True)))
        return alpha * x


class AttentionUpBlock(nn.Module):
    """Transposed-conv upsample, attention-gated skip fusion, and double-conv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.gate = AttentionGate(
            x_channels=skip_channels,
            g_channels=out_channels,
        )
        self.conv = DoubleConv(
            out_channels + skip_channels, out_channels, dropout_rate=dropout_rate
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(inputs)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        gated_skip = self.gate(skip, upsampled)
        return self.conv(torch.cat([gated_skip, upsampled], dim=1))


class AttentionUNet(nn.Module):
    """U-Net with additive attention gates on every skip connection.

    Identical to :class:`~src.models.unet.UNet` in every respect except that each
    :class:`~src.models.unet.UpBlock` is replaced with an
    :class:`AttentionUpBlock` that gates the encoder skip feature before
    concatenation.  Parameter count is larger than the base U-Net (additional
    theta/phi/psi convolutions per gate level).
    """

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
        self.up1 = AttentionUpBlock(B * 16, B * 8, B * 8, dropout_rate=rates[5])
        self.up2 = AttentionUpBlock(B * 8, B * 4, B * 4, dropout_rate=rates[6])
        self.up3 = AttentionUpBlock(B * 4, B * 2, B * 2, dropout_rate=rates[7])
        self.up4 = AttentionUpBlock(B * 2, B, B, dropout_rate=rates[8])
        self.head = nn.Conv2d(B, out_channels, kernel_size=1)

        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Keras-compatible initialisers matching the recovered ASTA config.

        3×3 Conv2d: He-normal (kaiming_normal fan_in).
        ConvTranspose2d and 1×1 Conv2d: Glorot-uniform (xavier_uniform).
        Biases: zeros.
        AttentionGate 1×1 convs fall into the 1×1 Glorot-uniform branch above.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (3, 3):
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )
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
