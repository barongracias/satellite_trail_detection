"""Basic segmentation metrics shared by the U-Net and Hough baselines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from src.config.constants import GLOBAL_SEED


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


def _as_writable_tensor(values: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Wrap torch.as_tensor with writability handling for numpy inputs.

    PIL-backed numpy arrays are non-writable; ``torch.as_tensor`` accepts them
    but emits a noisy "non-writable tensor" UserWarning. Take a cheap copy
    when that's the case so the warning never fires through the public API.
    """
    if torch.is_tensor(values):
        return values.detach()
    arr = np.asarray(values)
    if not arr.flags.writeable:
        arr = arr.copy()
    return torch.from_numpy(arr)


def _to_binary_mask(
    values: torch.Tensor | np.ndarray,
    threshold: float,
    from_logits: bool,
    device: torch.device,
) -> torch.Tensor:
    """Convert logits/probabilities/binary masks to a bool tensor on `device`."""
    t = _as_writable_tensor(values).to(device)

    if t.dtype == torch.bool:
        return t

    if not torch.is_floating_point(t) and not from_logits:
        return t > 0

    t = t.float()
    if from_logits:
        t = torch.sigmoid(t)
    return t >= threshold


def _segmentation_counts_torch(
    pred_mask: torch.Tensor, target_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute confusion counts as four int64 scalar tensors on the input device."""
    tp = torch.logical_and(pred_mask, target_mask).sum(dtype=torch.int64)
    fp = torch.logical_and(pred_mask, ~target_mask).sum(dtype=torch.int64)
    tn = torch.logical_and(~pred_mask, ~target_mask).sum(dtype=torch.int64)
    fn = torch.logical_and(~pred_mask, target_mask).sum(dtype=torch.int64)
    return tp, fp, tn, fn


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
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    from_logits: bool = False,
) -> SegmentationCounts:
    """Compute confusion counts for a binary segmentation prediction.

    Public API guarantee: returned fields are Python ints (not tensors) so
    SegmentationCounts remains safe to serialise via asdict()/json.dump.
    """
    # Convert prediction first so its device is the source of truth for
    # subsequent target alignment (target may be on CPU/numpy initially).
    pred = _as_writable_tensor(prediction)
    device = pred.device
    targ = _as_writable_tensor(target)

    # Strict shape check BEFORE any flattening, so same-numel-different-shape
    # mismatches still raise (would otherwise be hidden by ravel/reshape).
    if pred.shape != targ.shape:
        raise ValueError(
            f"Prediction and target shapes differ: "
            f"{tuple(pred.shape)} vs {tuple(targ.shape)}"
        )

    pred_mask = _to_binary_mask(pred, threshold, from_logits, device)
    target_mask = _to_binary_mask(targ, threshold=0.5, from_logits=False, device=device)

    tp, fp, tn, fn = _segmentation_counts_torch(pred_mask, target_mask)

    return SegmentationCounts(
        true_positive=int(tp.item()),
        false_positive=int(fp.item()),
        true_negative=int(tn.item()),
        false_negative=int(fn.item()),
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
    """Evaluate a segmentation model on a batched dataloader.

    Accumulates confusion counts and loss as on-device tensors and syncs to
    Python ints once at the end (five .item() calls per pass, regardless of
    batch count). The dataloader must yield dicts containing `image` and `mask`.
    """
    model.eval()
    # int64 running totals; loss accumulates as float on the same device.
    tp_total = torch.zeros((), dtype=torch.int64, device=device)
    fp_total = torch.zeros((), dtype=torch.int64, device=device)
    tn_total = torch.zeros((), dtype=torch.int64, device=device)
    fn_total = torch.zeros((), dtype=torch.int64, device=device)
    loss_total = torch.zeros((), dtype=torch.float32, device=device)
    batch_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            images = batch["image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            masks = batch["mask"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            logits = model(images)

            if criterion is not None:
                loss_total += criterion(logits, masks).detach()
                batch_count += 1

            pred_mask = _to_binary_mask(logits, threshold, True, device)
            target_mask = _to_binary_mask(masks, 0.5, False, device)
            tp, fp, tn, fn = _segmentation_counts_torch(pred_mask, target_mask)
            tp_total += tp
            fp_total += fp
            tn_total += tn
            fn_total += fn

    combined_counts = SegmentationCounts(
        true_positive=int(tp_total.item()),
        false_positive=int(fp_total.item()),
        true_negative=int(tn_total.item()),
        false_negative=int(fn_total.item()),
    )
    mean_loss = (
        None
        if criterion is None or batch_count == 0
        else float(loss_total.item()) / batch_count
    )
    return DatasetEvaluationResult(
        counts=combined_counts,
        metrics=compute_metrics_from_counts(combined_counts),
        mean_loss=mean_loss,
    )


_BOOTSTRAP_METRIC_NAMES = ("precision", "recall", "dice", "iou")


def _percentile(values: list[float], pct: float) -> float:
    """Linearly-interpolated percentile (pct on [0, 100]) without numpy."""
    if not values:
        raise ValueError("percentile requires non-empty values")
    sorted_values = sorted(values)
    n = len(sorted_values)
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    weight = rank - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _bootstrap_resample_counts(
    units: Sequence[SegmentationCounts],
    n_resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    """Resample `units` with replacement and compute metrics per resample."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = random.Random(seed)
    n = len(units)
    samples: dict[str, list[float]] = {name: [] for name in _BOOTSTRAP_METRIC_NAMES}
    for _ in range(n_resamples):
        draw = [units[rng.randrange(n)] for _ in range(n)]
        m = compute_metrics_from_counts(combine_counts(draw))
        samples["precision"].append(m.precision)
        samples["recall"].append(m.recall)
        samples["dice"].append(m.dice)
        samples["iou"].append(m.iou)
    return samples


def _summarise_bootstrap(
    samples: dict[str, list[float]],
    point: SegmentationMetrics,
    ci: float,
) -> dict[str, tuple[float, float, float]]:
    """Return ``{metric: (point, lo, hi)}`` from per-metric resample lists."""
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie strictly between 0 and 1")
    pct_lo = 100.0 * (1.0 - ci) / 2.0
    pct_hi = 100.0 - pct_lo
    point_values = {
        "precision": point.precision,
        "recall": point.recall,
        "dice": point.dice,
        "iou": point.iou,
    }
    return {
        name: (
            point_values[name],
            _percentile(samples[name], pct_lo),
            _percentile(samples[name], pct_hi),
        )
        for name in _BOOTSTRAP_METRIC_NAMES
    }


def bootstrap_metrics_cluster(
    per_image_counts: Mapping[str, SegmentationCounts],
    n_resamples: int = 1000,
    seed: int = GLOBAL_SEED,
    ci: float = 0.95,
) -> dict[str, tuple[float, float, float]]:
    """Cluster bootstrap by source image on pre-aggregated per-image counts.

    Resamples image keys with replacement; sums each draw's counts; computes
    P/R/Dice/IoU on the aggregate. Use when patches share within-image
    correlation (lighting, noise, annotator) — the i.i.d. patch-level
    bootstrap understates uncertainty in that regime.
    """
    if not per_image_counts:
        raise ValueError("per_image_counts must contain at least one image")
    units = list(per_image_counts.values())
    point = compute_metrics_from_counts(combine_counts(units))
    samples = _bootstrap_resample_counts(units, n_resamples, seed)
    return _summarise_bootstrap(samples, point, ci)


def bootstrap_metrics_patch(
    patch_counts: Sequence[SegmentationCounts],
    n_resamples: int = 1000,
    seed: int = GLOBAL_SEED,
    ci: float = 0.95,
) -> dict[str, tuple[float, float, float]]:
    """Patch-level (i.i.d.) bootstrap on per-patch counts.

    Provided alongside :func:`bootstrap_metrics_cluster` as a sanity-check
    comparison: the gap between the two CI widths quantifies the within-image
    correlation effect that cluster bootstrap captures.
    """
    if not patch_counts:
        raise ValueError("patch_counts must be non-empty")
    units = list(patch_counts)
    point = compute_metrics_from_counts(combine_counts(units))
    samples = _bootstrap_resample_counts(units, n_resamples, seed)
    return _summarise_bootstrap(samples, point, ci)
