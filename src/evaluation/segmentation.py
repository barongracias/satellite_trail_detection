"""Basic segmentation metrics shared by the U-Net and Hough baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class SegmentationCounts:
    """Store aggregated confusion-count totals for binary segmentation."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def total(self) -> int:
        """Return the total number of evaluated pixels."""
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )


@dataclass(frozen=True)
class SegmentationMetrics:
    """Store standard binary segmentation metrics for experiment reporting."""

    precision: float
    recall: float
    dice: float
    iou: float
    accuracy: float
    specificity: float
    predicted_positive_fraction: float
    target_positive_fraction: float


@dataclass(frozen=True)
class DatasetEvaluationResult:
    """Store dataset-level counts, metrics, and an optional mean loss."""

    counts: SegmentationCounts
    metrics: SegmentationMetrics
    mean_loss: float | None = None


def _safe_divide(numerator: int, denominator: int) -> float:
    """Return a floating-point division result with zero-protection."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _to_numpy_binary_mask(
    values: Any,
    threshold: float = 0.5,
    from_logits: bool = False,
) -> np.ndarray:
    """Convert logits, probabilities, or binary masks to a flat boolean mask."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    else:
        values = np.asarray(values)

    if values.dtype == np.bool_:
        return values.reshape(-1)

    if np.issubdtype(values.dtype, np.integer) and not from_logits:
        return (values > 0).reshape(-1)

    float_values = values.astype(np.float32, copy=False)
    if from_logits:
        float_values = 1.0 / (1.0 + np.exp(-float_values))
    return (float_values >= threshold).reshape(-1)


def combine_counts(counts: Iterable[SegmentationCounts]) -> SegmentationCounts:
    """Combine multiple confusion-count summaries into a single total."""
    counts = list(counts)

    return SegmentationCounts(
        true_positive=sum(item.true_positive for item in counts),
        false_positive=sum(item.false_positive for item in counts),
        true_negative=sum(item.true_negative for item in counts),
        false_negative=sum(item.false_negative for item in counts),
    )


def compute_segmentation_counts(
    prediction: Any,
    target: Any,
    threshold: float = 0.5,
    from_logits: bool = False,
) -> SegmentationCounts:
    """Compute confusion counts for a binary segmentation prediction."""
    prediction_mask = _to_numpy_binary_mask(
        prediction,
        threshold=threshold,
        from_logits=from_logits,
    )
    target_mask = _to_numpy_binary_mask(target, threshold=0.5, from_logits=False)

    if prediction_mask.shape != target_mask.shape:
        raise ValueError("Prediction and target masks must share the same shape")

    true_positive = int(np.logical_and(prediction_mask, target_mask).sum())
    false_positive = int(np.logical_and(prediction_mask, ~target_mask).sum())
    true_negative = int(np.logical_and(~prediction_mask, ~target_mask).sum())
    false_negative = int(np.logical_and(~prediction_mask, target_mask).sum())

    return SegmentationCounts(
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def compute_metrics_from_counts(counts: SegmentationCounts) -> SegmentationMetrics:
    """Convert confusion counts into standard binary segmentation metrics."""
    precision = _safe_divide(
        counts.true_positive,
        counts.true_positive + counts.false_positive,
    )
    recall = _safe_divide(
        counts.true_positive,
        counts.true_positive + counts.false_negative,
    )
    dice = _safe_divide(
        2 * counts.true_positive,
        2 * counts.true_positive + counts.false_positive + counts.false_negative,
    )
    iou = _safe_divide(
        counts.true_positive,
        counts.true_positive + counts.false_positive + counts.false_negative,
    )
    accuracy = _safe_divide(
        counts.true_positive + counts.true_negative,
        counts.total(),
    )
    specificity = _safe_divide(
        counts.true_negative,
        counts.true_negative + counts.false_positive,
    )
    predicted_positive_fraction = _safe_divide(
        counts.true_positive + counts.false_positive,
        counts.total(),
    )
    target_positive_fraction = _safe_divide(
        counts.true_positive + counts.false_negative,
        counts.total(),
    )

    return SegmentationMetrics(
        precision=precision,
        recall=recall,
        dice=dice,
        iou=iou,
        accuracy=accuracy,
        specificity=specificity,
        predicted_positive_fraction=predicted_positive_fraction,
        target_positive_fraction=target_positive_fraction,
    )


def compute_segmentation_metrics(
    prediction: Any,
    target: Any,
    threshold: float = 0.5,
    from_logits: bool = False,
) -> SegmentationMetrics:
    """Compute binary segmentation metrics directly from predictions and targets."""
    counts = compute_segmentation_counts(
        prediction,
        target,
        threshold=threshold,
        from_logits=from_logits,
    )
    return compute_metrics_from_counts(counts)


def evaluate_model_on_dataloader(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    device: Any,
    threshold: float = 0.5,
    criterion: Any | None = None,
    max_batches: int | None = None,
) -> DatasetEvaluationResult:
    """
    Evaluate a segmentation model on a batched dataloader.

    The dataloader must yield dictionaries containing `image` and `mask`.
    """
    import torch

    model.eval()
    all_counts: list[SegmentationCounts] = []
    loss_total = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            images = batch["image"].to(device=device, dtype=torch.float32)
            masks = batch["mask"].to(device=device, dtype=torch.float32)
            logits = model(images)

            if criterion is not None:
                loss_total += float(criterion(logits, masks).item())
                batch_count += 1

            all_counts.append(
                compute_segmentation_counts(
                    logits,
                    masks,
                    threshold=threshold,
                    from_logits=True,
                )
            )

    combined_counts = combine_counts(all_counts)
    mean_loss = (
        None if criterion is None or batch_count == 0 else loss_total / batch_count
    )
    return DatasetEvaluationResult(
        counts=combined_counts,
        metrics=compute_metrics_from_counts(combined_counts),
        mean_loss=mean_loss,
    )
