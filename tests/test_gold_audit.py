"""Tests for the M9.4 audit-crop export tooling and gold-audit scoring scaffold.

All tests run on synthetic inputs — no CSD3 data, no model weights. The real
gold_audit_eval.py run happens only after annotation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from scripts.evaluation.export_audit_crops import (   # noqa: E402
    CROP_SIZE,
    component_endpoints,
    crop_window,
    sample_point_on_component,
    stratified_component_sample,
)
from scripts.evaluation.gold_audit_eval import (   # noqa: E402
    load_annotation_mask,
    load_verdicts,
    score_mask_pair,
    width_median,
)


# ---------------------------------------------------------------- export side

def test_crop_window_clamps_to_frame() -> None:
    shape = (1000, 2000)
    # Interior centre: window centred on it.
    assert crop_window(500, 1000, shape) == (500 - CROP_SIZE // 2, 1000 - CROP_SIZE // 2)
    # Corners: clamped to the frame, never negative, never overhanging.
    assert crop_window(0, 0, shape) == (0, 0)
    assert crop_window(999, 1999, shape) == (1000 - CROP_SIZE, 2000 - CROP_SIZE)
    with pytest.raises(ValueError):
        crop_window(10, 10, (100, 100))   # frame smaller than crop


def test_sample_point_on_component_lies_on_component() -> None:
    mask = np.zeros((800, 800), dtype=bool)
    mask[400, 100:700] = True   # horizontal trail
    rng = np.random.default_rng(2804)
    for _ in range(20):
        y, x = sample_point_on_component(mask, rng)
        assert mask[y, x]
    # Determinism under a fixed seed.
    a = sample_point_on_component(mask, np.random.default_rng(7))
    b = sample_point_on_component(mask, np.random.default_rng(7))
    assert a == b


def test_component_endpoints_are_extremes() -> None:
    mask = np.zeros((600, 600), dtype=bool)
    mask[100:500, 300] = True   # vertical trail
    (y1, x1), (y2, x2) = component_endpoints(mask)
    assert {y1, y2} == {100, 499}
    assert x1 == x2 == 300


def _fake_components(n_per_tier: int = 6, n_images: int = 12) -> list[dict]:
    comps = []
    idx = 0
    for tier in ("low", "mid", "high"):
        for i in range(n_per_tier):
            comps.append({
                "source_image": f"img_{idx % n_images:02d}.png",
                "component_index": idx + 1,
                "contrast_tier": tier,
            })
            idx += 1
    return comps


def test_stratified_component_sample_quotas_and_disjointness() -> None:
    comps = _fake_components()
    rng = np.random.default_rng(2804)
    interior, endpoint = stratified_component_sample(
        comps, n_interior=9, n_endpoint=5, min_distinct_images=3, rng=rng,
    )
    assert len(interior) == 9
    assert len(endpoint) == 5
    tiers = [c["contrast_tier"] for c in interior]
    assert tiers.count("low") == tiers.count("mid") == tiers.count("high") == 3
    keys = lambda cs: {(c["source_image"], c["component_index"]) for c in cs}
    assert keys(interior) & keys(endpoint) == set()
    # Deterministic for a fixed seed.
    interior2, endpoint2 = stratified_component_sample(
        comps, n_interior=9, n_endpoint=5, min_distinct_images=3,
        rng=np.random.default_rng(2804),
    )
    assert keys(interior) == keys(interior2)
    assert keys(endpoint) == keys(endpoint2)


def test_stratified_component_sample_unreachable_image_count_raises() -> None:
    comps = _fake_components(n_per_tier=2, n_images=1)   # all on one image
    with pytest.raises(RuntimeError):
        stratified_component_sample(
            comps, n_interior=3, n_endpoint=1, min_distinct_images=2,
            rng=np.random.default_rng(0),
        )


# ------------------------------------------------------------------ gold side

def test_load_annotation_mask_contract(tmp_path: Path) -> None:
    # Valid binary annotation.
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[10, 5:20] = 255
    p = tmp_path / "c001.png"
    Image.fromarray(arr).save(p)
    mask = load_annotation_mask(p)
    assert mask.dtype == bool and mask.sum() == 15

    # All-background single-valued PNG is allowed (empty annotation).
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(tmp_path / "c002.png")
    assert not load_annotation_mask(tmp_path / "c002.png").any()

    # Anti-aliased / >2-valued exports must be rejected.
    bad = np.zeros((32, 32), dtype=np.uint8)
    bad[10, 5:20] = 255
    bad[11, 5:20] = 127
    Image.fromarray(bad).save(tmp_path / "c003.png")
    with pytest.raises(ValueError, match="binary"):
        load_annotation_mask(tmp_path / "c003.png")

    # Constant non-zero PNG is malformed, not an empty annotation.
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(tmp_path / "c004.png")
    with pytest.raises(ValueError, match="all-background"):
        load_annotation_mask(tmp_path / "c004.png")


def test_score_mask_pair_strict_vs_tolerant_on_1px_offset() -> None:
    """The audit's central measurement: a 1 px boundary offset is a strict
    error but perfect at +/-1 px."""
    gold = np.zeros((64, 64), dtype=bool)
    other = np.zeros((64, 64), dtype=bool)
    gold[30, 10:50] = True
    other[31, 10:50] = True
    counts = score_mask_pair(gold, other)
    from src.evaluation.segmentation import boundary_tolerant_metrics

    strict = boundary_tolerant_metrics(counts[0])
    tol1 = boundary_tolerant_metrics(counts[1])
    assert strict["precision"] == 0.0 and strict["recall"] == 0.0
    assert tol1["precision"] == 1.0 and tol1["recall"] == 1.0 and tol1["f1"] == 1.0


def test_width_median_recovers_known_widths() -> None:
    empty = np.zeros((32, 32), dtype=bool)
    assert width_median(empty) is None

    three_px = np.zeros((64, 64), dtype=bool)
    three_px[30:33, 5:60] = True   # 3 px wide horizontal band
    w3 = width_median(three_px)
    assert w3 == pytest.approx(3.0, abs=1.0)

    one_px = np.zeros((64, 64), dtype=bool)
    one_px[30, 5:60] = True
    w1 = width_median(one_px)
    assert w1 == pytest.approx(2.0, abs=1.0)   # 2*EDT of a 1px line is 2
    assert w1 < w3


def test_load_verdicts_validates(tmp_path: Path) -> None:
    good = tmp_path / "verdicts.csv"
    good.write_text("crop_name,verdict\nc001.png,trail\nc002.png,uncertain\n")
    v = load_verdicts(good)
    assert v == {"c001.png": "trail", "c002.png": "uncertain"}

    bad = tmp_path / "bad.csv"
    bad.write_text("crop_name,verdict\nc001.png,maybe\n")
    with pytest.raises(ValueError, match="verdict"):
        load_verdicts(bad)
