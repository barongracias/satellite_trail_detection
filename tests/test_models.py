import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.models.unet import DROPOUT_RATES, DoubleConv, UNet  # noqa: E402


def test_unet_preserves_528_spatial_dimensions() -> None:
    model = UNet(in_channels=1, out_channels=1, base_channels=8).eval()
    inputs = torch.randn(1, 1, 528, 528)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (1, 1, 528, 528)


def test_unet_parameter_count_matches_recovered_architecture() -> None:
    model = UNet(in_channels=1, out_channels=1, base_channels=8)

    assert sum(p.numel() for p in model.parameters()) == 485_673


def test_unet_uses_recovered_pooling_activation_and_dropout_modules() -> None:
    model = UNet()

    assert sum(isinstance(module, nn.AvgPool2d) for module in model.modules()) == 4
    assert not any(isinstance(module, nn.MaxPool2d) for module in model.modules())
    assert not any(isinstance(module, nn.Dropout2d) for module in model.modules())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert not any(isinstance(module, nn.Sigmoid) for module in model.modules())
    assert all(
        module.negative_slope == pytest.approx(0.3)
        for module in model.modules()
        if isinstance(module, nn.LeakyReLU)
    )


def test_unet_dropout_rates_and_placement_match_recovered_blocks() -> None:
    model = UNet()
    double_convs = [module for module in model.modules() if isinstance(module, DoubleConv)]
    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout)]

    assert [module.p for module in dropouts] == pytest.approx(DROPOUT_RATES)
    assert len(double_convs) == len(DROPOUT_RATES) == 9
    for block in double_convs:
        assert [type(module) for module in block.layers] == [
            nn.Conv2d,
            nn.LeakyReLU,
            nn.Dropout,
            nn.Conv2d,
            nn.LeakyReLU,
        ]


def test_unet_biases_initialise_to_zero() -> None:
    model = UNet()

    for name, param in model.named_parameters():
        if name.endswith("bias"):
            assert torch.count_nonzero(param).item() == 0


def test_load_segmentation_model_dispatches_on_model_type(tmp_path) -> None:
    # The single loader must reconstruct the right class from config.model_type
    # (this is the regression guard for the dispatch bug once spread across loaders).
    from src.models.attention_unet import AttentionUNet
    from src.models.loading import load_segmentation_model

    device = torch.device("cpu")
    for model_cls, mtype, norm in [(UNet, "unet", "full_image"),
                                   (AttentionUNet, "attention_unet", "fixed")]:
        m = model_cls(base_channels=8)
        ckpt = tmp_path / f"{mtype}.pth"
        torch.save({"model_state_dict": m.state_dict(),
                    "config": {"model_type": mtype, "base_channels": 8, "normalisation": norm}}, ckpt)
        loaded, loaded_norm = load_segmentation_model(ckpt, device)
        assert isinstance(loaded, model_cls)
        assert loaded_norm == norm
        assert not loaded.training   # eval mode

    # Missing model_type defaults to UNet.
    m = UNet(base_channels=8)
    ckpt = tmp_path / "legacy.pth"
    torch.save({"model_state_dict": m.state_dict(), "config": {"base_channels": 8}}, ckpt)
    loaded, loaded_norm = load_segmentation_model(ckpt, device)
    assert isinstance(loaded, UNet) and loaded_norm == "fixed"


def test_unet_initialiser_calls_match_keras_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    kaiming_calls = []
    xavier_calls = []

    def fake_kaiming(tensor: torch.Tensor, **kwargs: object) -> torch.Tensor:
        kaiming_calls.append(kwargs)
        return tensor

    def fake_xavier(tensor: torch.Tensor, **kwargs: object) -> torch.Tensor:
        xavier_calls.append(kwargs)
        return tensor

    monkeypatch.setattr(nn.init, "kaiming_normal_", fake_kaiming)
    monkeypatch.setattr(nn.init, "xavier_uniform_", fake_xavier)

    UNet()

    assert len(kaiming_calls) == 18
    assert all(call == {"mode": "fan_in", "nonlinearity": "relu"} for call in kaiming_calls)
    assert len(xavier_calls) == 5
