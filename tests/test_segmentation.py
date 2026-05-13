import numpy as np

from src.evaluation.segmentation import (
    combine_counts,
    compute_segmentation_counts,
    compute_segmentation_metrics,
)


def test_segmentation_metrics_work_for_binary_masks_and_logits() -> None:
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    target = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    counts = compute_segmentation_counts(prediction, target)
    metrics = compute_segmentation_metrics(prediction, target)

    assert counts.true_positive == 1
    assert counts.false_positive == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.dice == 2.0 / 3.0
    assert metrics.iou == 0.5

    logits = np.array([[8.0, -8.0], [8.0, -8.0]], dtype=np.float32)
    counts = compute_segmentation_counts(logits, target, from_logits=True)
    combined = combine_counts([counts, counts])

    assert combined.true_positive == 2
