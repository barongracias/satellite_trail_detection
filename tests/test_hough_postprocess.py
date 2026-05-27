"""Tests for the batched Hough post-processing helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

# Add the repo root to sys.path so `scripts/` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from scripts.hough_postprocess import (   # noqa: E402
    _HOUGH_MAX_BATCH,
    _chunks,
    _infer_batch,
    _load_normalised_patch,
)
from src.models.unet import UNet  # noqa: E402


def _make_patch_files(tmp_path: Path, n: int, value: int = 128) -> list[Path]:
    """Write n grayscale PNG patches and return their paths."""
    paths = []
    for i in range(n):
        arr = np.full((16, 16), value + i, dtype=np.uint8)
        p = tmp_path / f"patch{i}_{0}_{0}_image.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def _build_tiny_unet() -> UNet:
    """A 1×16×16 UNet that's small enough to forward on CPU."""
    torch.manual_seed(0)
    return UNet(base_channels=4).eval()


def test_batched_hough_matches_single_patch_inference(tmp_path: Path) -> None:
    """Batched forward output must be numerically equivalent (within LSB tolerance)
    to per-patch forwards. Covers the full_image stats path as well."""
    paths = _make_patch_files(tmp_path, n=2)
    model = _build_tiny_unet()
    device = torch.device("cpu")

    # full_image stats path: pass per-patch (mean, std) on the [0, 1] scale.
    stats = [(0.5, 0.25), (0.6, 0.2)]

    single = []
    for p, (mean, std) in zip(paths, stats):
        t = _load_normalised_patch(str(p), "full_image", mean, std)
        single_prob = _infer_batch([t], model, device)
        assert single_prob.shape == (1, 16, 16)
        single.append(single_prob[0])

    batched_tensors = [
        _load_normalised_patch(str(p), "full_image", mean, std)
        for p, (mean, std) in zip(paths, stats)
    ]
    batched_probs = _infer_batch(batched_tensors, model, device)
    assert batched_probs.shape == (2, 16, 16)

    for s, b in zip(single, batched_probs):
        assert np.allclose(s, b, atol=1e-5)


def test_batched_hough_chunks_correctly(tmp_path: Path) -> None:
    """Chunked load+forward at MAX_BATCH=1 and MAX_BATCH=2 must produce identical
    canvases on a 5-patch image. Exercises the _chunks slicing logic."""
    paths = _make_patch_files(tmp_path, n=5)
    model = _build_tiny_unet()
    device = torch.device("cpu")
    tensors = [_load_normalised_patch(str(p), "fixed") for p in paths]

    def _run(max_batch: int) -> np.ndarray:
        outs = []
        for chunk in _chunks(tensors, max_batch):
            outs.append(_infer_batch(chunk, model, device))
        return np.concatenate(outs, axis=0)

    canvas_1 = _run(1)
    canvas_2 = _run(2)
    canvas_default = _run(_HOUGH_MAX_BATCH)
    assert canvas_1.shape == canvas_2.shape == canvas_default.shape == (5, 16, 16)
    assert np.allclose(canvas_1, canvas_2, atol=1e-5)
    assert np.allclose(canvas_1, canvas_default, atol=1e-5)


def test_streaming_sweep_arithmetic_matches_known_counts() -> None:
    """Direct arithmetic regression: feed streaming_sweep_thresholds a known
    constant-prob batch and verify per-threshold TP/FP/FN/precision/recall
    against hand-computed values. Locks the per-batch sweep math against
    accidental drift if the helper is refactored."""
    from scripts.threshold_sweep import streaming_sweep_thresholds   # noqa: E402

    class _ConstantModel(torch.nn.Module):
        """Returns logits that sigmoid to a known constant — independent of input."""
        def __init__(self, target_prob: float) -> None:
            super().__init__()
            # logit such that sigmoid(logit) ≈ target_prob
            self.logit = float(np.log(target_prob / (1 - target_prob)))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, self.logit)

    # 2 batches × 1 sample × 1 × 4×4 = 32 pixels total.
    # Half of pixels are positive (= 1.0), half are negative (= 0.0).
    targets = [
        torch.tensor([[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]],
                     dtype=torch.float32),
        torch.tensor([[[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]],
                     dtype=torch.float32),
    ]
    loader = [{"image": t.clone(), "mask": t} for t in targets]
    n_pos = int(sum(t.sum().item() for t in targets))   # = 8
    n_neg = sum(t.numel() for t in targets) - n_pos      # = 24

    # Constant probability 0.7. At threshold 0.5 → all pixels predicted positive.
    model = _ConstantModel(target_prob=0.7)
    thresholds = np.array([0.5, 0.8])

    sweep = streaming_sweep_thresholds(model, loader, torch.device("cpu"), thresholds)
    # At t=0.5: all 32 pixels predicted positive → tp=n_pos, fp=n_neg, fn=0
    row_low = sweep[0]
    assert row_low["threshold"] == 0.5
    assert row_low["recall"] == 1.0   # tp/(tp+fn) = 8/8
    assert abs(row_low["precision"] - n_pos / (n_pos + n_neg)) < 1e-6
    # At t=0.8: prob 0.7 < 0.8 → no pixels predicted positive → tp=0, fn=n_pos
    row_high = sweep[1]
    assert row_high["threshold"] == 0.8
    assert row_high["recall"] == 0.0   # tp/(tp+fn) = 0/8
    # When tp+fp=0, precision falls through _safe_divide to 0.0
    assert row_high["precision"] == 0.0
