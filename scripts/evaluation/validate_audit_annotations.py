#!/usr/bin/env python
"""Validate the 64 blinded masks and verdicts without opening the seal."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ALLOWED_VERDICTS = {"trail", "no_trail", "uncertain"}
CROP_SIZE = 528


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--verdicts_csv", required=True)
    parser.add_argument("--expected_count", type=int, default=64)
    return parser.parse_args()


def validate_annotations(
    annotation_dir: str | Path,
    verdicts_csv: str | Path,
    expected_names: list[str],
) -> None:
    """Require the exact filenames, binary 528-pixel grids, and all verdicts."""
    directory = Path(annotation_dir)
    actual = sorted(path.name for path in directory.glob("*.png"))
    expected = sorted(expected_names)
    if actual != expected:
        raise ValueError(
            f"Annotation file set mismatch: missing={sorted(set(expected)-set(actual))}, "
            f"extra={sorted(set(actual)-set(expected))}"
        )

    for name in expected:
        path = directory / name
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        if array.shape != (CROP_SIZE, CROP_SIZE):
            raise ValueError(f"{path}: shape {array.shape} != {(CROP_SIZE, CROP_SIZE)}")
        values = set(np.unique(array).tolist())
        if not values.issubset({0, 255}):
            raise ValueError(f"{path}: expected values in {{0,255}}, found {sorted(values)}")
        if values == {255}:
            raise ValueError(f"{path}: a single-valued mask must be all-background (0)")

    with open(verdicts_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"crop_name", "verdict"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"{verdicts_csv}: expected crop_name,verdict columns")
        rows = list(reader)
    names = [row["crop_name"].strip() for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"{verdicts_csv}: duplicate crop names")
    if sorted(names) != expected:
        raise ValueError("Verdict crop-name set does not match the expected masks")
    for row in rows:
        verdict = row["verdict"].strip()
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"Unknown verdict {verdict!r} for {row['crop_name']}")


def main() -> None:
    args = parse_args()
    expected = [f"c{index:03d}.png" for index in range(1, args.expected_count + 1)]
    validate_annotations(args.annotation_dir, args.verdicts_csv, expected)
    print(f"PASS: {len(expected)} masks and verdicts")


if __name__ == "__main__":
    main()
