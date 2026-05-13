import pytest


torch = pytest.importorskip("torch")

from src.models.unet import UNet  # noqa: E402


def test_unet_preserves_spatial_dimensions() -> None:
    model = UNet(in_channels=1, out_channels=1, base_channels=16)
    inputs = torch.randn(2, 1, 128, 128)

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 128, 128)
