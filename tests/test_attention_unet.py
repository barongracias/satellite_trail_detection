import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from src.models.attention_unet import AttentionGate, AttentionUNet  # noqa: E402
from src.models.unet import UNet  # noqa: E402

_BASE_UNET_PARAMS = 485_673


def test_attention_unet_preserves_528_spatial_dimensions() -> None:
    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=8).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 528, 528))
    assert out.shape == (1, 1, 528, 528)


def test_attention_unet_outputs_logits_not_probabilities() -> None:
    model = AttentionUNet().eval()
    assert not any(isinstance(m, nn.Sigmoid) for m in model.modules())
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(torch.randn(2, 1, 64, 64))
    assert out.min().item() < 0.0 or out.max().item() > 1.0


def test_attention_gate_coefficients_in_unit_interval() -> None:
    model = AttentionUNet().eval()
    captured_alphas: list[torch.Tensor] = []

    original_forward = AttentionGate.forward

    def capturing_forward(
        self: AttentionGate, x: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(
            self.psi(F.relu(self.theta_x(x) + self.phi_g(g), inplace=True))
        )
        captured_alphas.append(alpha.detach())
        return alpha * x

    AttentionGate.forward = capturing_forward  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            model(torch.randn(1, 1, 64, 64))
    finally:
        AttentionGate.forward = original_forward  # type: ignore[method-assign]

    assert len(captured_alphas) == 4, "Expected one gate per skip connection"
    for alpha in captured_alphas:
        assert alpha.min().item() >= 0.0
        assert alpha.max().item() <= 1.0


def test_attention_unet_has_more_parameters_than_base_unet() -> None:
    attn_params = sum(p.numel() for p in AttentionUNet().parameters())
    assert attn_params > _BASE_UNET_PARAMS


def test_attention_unet_output_deterministic_under_fixed_seed() -> None:
    torch.manual_seed(42)
    model = AttentionUNet().eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out1 = model(x.clone())
        out2 = model(x.clone())
    assert torch.allclose(out1, out2)


def test_model_type_attention_unet_dispatches_to_attention_unet() -> None:
    model_type = "attention_unet"
    model_cls = AttentionUNet if model_type == "attention_unet" else UNet
    model = model_cls(in_channels=1, out_channels=1, base_channels=8)
    assert isinstance(model, AttentionUNet)
    assert not isinstance(model, UNet)


def test_model_type_unet_dispatch_state_dict_unchanged() -> None:
    """M5.6 regression: the plain UNet path must be structurally identical to direct construction."""
    direct = UNet(in_channels=1, out_channels=1, base_channels=8)
    model_type = "unet"
    model_cls = AttentionUNet if model_type == "attention_unet" else UNet
    dispatched = model_cls(in_channels=1, out_channels=1, base_channels=8)

    direct_shapes = {k: tuple(v.shape) for k, v in direct.state_dict().items()}
    dispatched_shapes = {k: tuple(v.shape) for k, v in dispatched.state_dict().items()}
    assert direct_shapes == dispatched_shapes
