"""Synthetic tests for audit-crop export, annotation validation, and scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from scripts.evaluation.export_audit_crops import (  # noqa: E402
    CROP_SIZE,
    component_endpoints,
    crop_window,
    sample_point_on_component,
    stratified_component_sample,
)
from scripts.evaluation.gold_audit_eval import (  # noqa: E402
    classify_mechanism,
    load_annotation_mask,
    load_verdicts,
    score_mask_pair,
    width_median,
)
from scripts.evaluation.validate_audit_annotations import (  # noqa: E402
    validate_annotations,
)


def test_crop_window_clamps_to_frame() -> None:
    shape = (1000, 2000)
    assert crop_window(500, 1000, shape) == (
        500 - CROP_SIZE // 2,
        1000 - CROP_SIZE // 2,
    )
    assert crop_window(0, 0, shape) == (0, 0)
    assert crop_window(999, 1999, shape) == (
        1000 - CROP_SIZE,
        2000 - CROP_SIZE,
    )
    with pytest.raises(ValueError):
        crop_window(10, 10, (100, 100))


def test_sample_point_on_component_lies_on_component() -> None:
    mask = np.zeros((800, 800), dtype=bool)
    mask[400, 100:700] = True
    rng = np.random.default_rng(2804)
    for _ in range(20):
        y, x = sample_point_on_component(mask, rng)
        assert mask[y, x]
    assert sample_point_on_component(mask, np.random.default_rng(7)) == (
        sample_point_on_component(mask, np.random.default_rng(7))
    )


def test_component_endpoints_are_extremes() -> None:
    mask = np.zeros((600, 600), dtype=bool)
    mask[100:500, 300] = True
    (y1, x1), (y2, x2) = component_endpoints(mask)
    assert {y1, y2} == {100, 499}
    assert x1 == x2 == 300


def _fake_components(n_per_tier: int = 6, n_images: int = 12) -> list[dict]:
    components = []
    index = 0
    for tier in ("low", "mid", "high"):
        for _ in range(n_per_tier):
            components.append(
                {
                    "source_image": f"img_{index % n_images:02d}.png",
                    "component_index": index + 1,
                    "contrast_tier": tier,
                }
            )
            index += 1
    return components


def test_stratified_component_sample_quotas_and_disjointness() -> None:
    components = _fake_components()
    interior, endpoint = stratified_component_sample(
        components,
        n_interior=9,
        n_endpoint=5,
        min_distinct_images=3,
        rng=np.random.default_rng(2804),
    )
    assert len(interior) == 9
    assert len(endpoint) == 5
    tiers = [component["contrast_tier"] for component in interior]
    assert tiers.count("low") == tiers.count("mid") == tiers.count("high") == 3
    keys = lambda values: {
        (component["source_image"], component["component_index"])
        for component in values
    }
    assert not keys(interior) & keys(endpoint)
    interior_2, endpoint_2 = stratified_component_sample(
        components,
        n_interior=9,
        n_endpoint=5,
        min_distinct_images=3,
        rng=np.random.default_rng(2804),
    )
    assert keys(interior) == keys(interior_2)
    assert keys(endpoint) == keys(endpoint_2)


def test_stratified_component_sample_unreachable_image_count_raises() -> None:
    with pytest.raises(RuntimeError):
        stratified_component_sample(
            _fake_components(n_per_tier=2, n_images=1),
            n_interior=3,
            n_endpoint=1,
            min_distinct_images=2,
            rng=np.random.default_rng(0),
        )


def test_load_annotation_mask_contract(tmp_path: Path) -> None:
    array = np.zeros((32, 32), dtype=np.uint8)
    array[10, 5:20] = 255
    path = tmp_path / "c001.png"
    Image.fromarray(array).save(path)
    mask = load_annotation_mask(path)
    assert mask.dtype == bool and mask.sum() == 15

    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(tmp_path / "c002.png")
    assert not load_annotation_mask(tmp_path / "c002.png").any()

    bad = array.copy()
    bad[11, 5:20] = 127
    Image.fromarray(bad).save(tmp_path / "c003.png")
    with pytest.raises(ValueError, match="binary"):
        load_annotation_mask(tmp_path / "c003.png")

    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(tmp_path / "c004.png")
    with pytest.raises(ValueError, match="all-background"):
        load_annotation_mask(tmp_path / "c004.png")


def test_score_mask_pair_strict_vs_tolerant_on_1px_offset() -> None:
    reference = np.zeros((64, 64), dtype=bool)
    other = np.zeros((64, 64), dtype=bool)
    reference[30, 10:50] = True
    other[31, 10:50] = True
    counts = score_mask_pair(reference, other)
    from src.evaluation.segmentation import boundary_tolerant_metrics

    strict = boundary_tolerant_metrics(counts[0])
    tolerant = boundary_tolerant_metrics(counts[1])
    assert strict["precision"] == 0.0 and strict["recall"] == 0.0
    assert tolerant["precision"] == 1.0
    assert tolerant["recall"] == 1.0
    assert tolerant["f1"] == 1.0


def test_width_median_recovers_known_widths() -> None:
    assert width_median(np.zeros((32, 32), dtype=bool)) is None
    three_px = np.zeros((64, 64), dtype=bool)
    three_px[30:33, 5:60] = True
    one_px = np.zeros((64, 64), dtype=bool)
    one_px[30, 5:60] = True
    assert width_median(three_px) == pytest.approx(3.0, abs=1.0)
    assert width_median(one_px) == pytest.approx(2.0, abs=1.0)
    assert width_median(one_px) < width_median(three_px)


def test_load_verdicts_validates(tmp_path: Path) -> None:
    good = tmp_path / "verdicts.csv"
    good.write_text("crop_name,verdict\nc001.png,trail\nc002.png,uncertain\n")
    assert load_verdicts(good) == {
        "c001.png": "trail",
        "c002.png": "uncertain",
    }
    bad = tmp_path / "bad.csv"
    bad.write_text("crop_name,verdict\nc001.png,maybe\n")
    with pytest.raises(ValueError, match="verdict"):
        load_verdicts(bad)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ((0.80, 0.85, 5.0, 6.0, 6.5), "thin-original support"),
        ((0.80, 0.85, 6.0, 6.0, 7.0), "model-overpaint support"),
        ((0.80, 0.85, 5.0, 5.4, 5.5), "mixed/inconclusive"),
        ((0.80, 0.85, 5.0, None, 5.5), "mixed/inconclusive"),
    ],
)
def test_classify_mechanism_rules(arguments: tuple, expected: str) -> None:
    assert classify_mechanism(*arguments) == expected


def test_validate_annotations_checks_files_grid_and_verdicts(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "masks"
    annotation_dir.mkdir()
    names = ["c001.png", "c002.png"]
    for index, name in enumerate(names):
        array = np.zeros((528, 528), dtype=np.uint8)
        if index == 0:
            array[100, 100:200] = 255
        Image.fromarray(array).save(annotation_dir / name)
    verdicts = tmp_path / "verdicts.csv"
    verdicts.write_text("crop_name,verdict\nc001.png,trail\nc002.png,no_trail\n")
    assert validate_annotations(annotation_dir, verdicts, names) is None

    bad = np.zeros((528, 528), dtype=np.uint8)
    bad[0, 0] = 127
    Image.fromarray(bad).save(annotation_dir / "c002.png")
    with pytest.raises(ValueError, match="expected values"):
        validate_annotations(annotation_dir, verdicts, names)
