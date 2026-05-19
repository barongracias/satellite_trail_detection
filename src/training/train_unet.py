"""U-Net training entry point for local validation runs and early experiments."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config.constants import GLOBAL_SEED
from src.data.dataset import PatchDatasetConfig, PatchDirectoryDataset, SatelliteTrailPatchDataset
from src.data.indexing import discover_image_mask_pairs
from src.data.splits import create_splits
from src.data.transforms import get_patch_transforms
from src.evaluation.segmentation import (
    DatasetEvaluationResult,
    evaluate_model_on_dataloader,
)
from src.models.unet import UNet
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class TrainingConfig:
    """Collect the runtime configuration for a local U-Net experiment."""

    data_root: str | Path
    patch_size: int = 512
    stride: int | None = None
    batch_size: int = 2
    learning_rate: float = 1e-3
    epochs: int = 3
    max_steps: int | None = None
    num_workers: int = 0
    patch_dir: str | Path | None = None
    pos_weight: float | None = None
    auto_pos_weight: bool = False
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    dice_smooth: float = 1e-4
    dice_denominator_squared: bool = True
    use_amp: bool = False
    lr_scheduler: str | None = None
    base_channels: int = 8
    dropout_rate: float = 0.5
    normalisation: str = "fixed"
    augment_train: bool = True
    device: str = "auto"
    checkpoint_dir: str | Path = Path("results/checkpoints")
    log_dir: str | Path = Path("results/logs")
    experiment_name: str = "unet_baseline"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = GLOBAL_SEED
    threshold: float = 0.5
    eval_max_batches: int | None = None

    def __post_init__(self) -> None:
        """Normalise paths and validate the few settings that must be sane."""
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        object.__setattr__(self, "log_dir", Path(self.log_dir))
        if self.patch_dir is not None:
            object.__setattr__(self, "patch_dir", Path(self.patch_dir))

        if self.patch_size <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("patch_size, batch_size, and epochs must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.eval_max_batches is not None and self.eval_max_batches <= 0:
            raise ValueError("eval_max_batches must be positive when provided")
        if self.learning_rate <= 0 or self.base_channels <= 0:
            raise ValueError("learning_rate and base_channels must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.train_ratio <= 0 or self.val_ratio <= 0:
            raise ValueError("train_ratio and val_ratio must be positive")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio and val_ratio must leave room for a test split")
        if self.threshold <= 0 or self.threshold >= 1:
            raise ValueError("threshold must lie between 0 and 1")
        if self.normalisation not in ("fixed", "per_image", "full_image"):
            raise ValueError("normalisation must be 'fixed', 'per_image', or 'full_image'")
        if self.dice_smooth <= 0:
            raise ValueError("dice_smooth must be positive")
        if self.lr_scheduler not in (None, "cosine"):
            raise ValueError("lr_scheduler must be None or 'cosine'")


def parse_args() -> TrainingConfig:
    """Load a TrainingConfig from a YAML file given by --config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML experiment config file (e.g. configs/experiments/unet_baseline.yaml).",
    )
    parser.add_argument(
        "--patch_dir",
        type=Path,
        default=None,
        help="Override patch_dir from the YAML config (e.g. data/patches on CSD3).",
    )
    args = parser.parse_args()
    with open(args.config) as f:
        data = yaml.safe_load(f)
    if args.patch_dir is not None:
        data["patch_dir"] = str(args.patch_dir)
    return TrainingConfig(**data)


def _serialise_config(config: TrainingConfig) -> dict[str, Any]:
    """Convert a training configuration into a JSON-friendly dictionary."""
    data = asdict(config)
    data["data_root"] = str(config.data_root)
    data["checkpoint_dir"] = str(config.checkpoint_dir)
    data["log_dir"] = str(config.log_dir)
    data["patch_dir"] = str(config.patch_dir) if config.patch_dir is not None else None
    return data


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested runtime device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _worker_init_fn(worker_id: int, seed: int = GLOBAL_SEED) -> None:
    """Seed each DataLoader worker independently."""
    seed_everything(seed + worker_id)


def build_dataloaders(
    config: TrainingConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create deterministic train, validation, and test dataloaders.

    When ``config.patch_dir`` is set, loads pre-built patches from that directory
    tree using :class:`PatchDirectoryDataset`.  Otherwise falls back to on-the-fly
    patch extraction from ``config.data_root`` via :class:`SatelliteTrailPatchDataset`.

    ``device`` is the resolved training device; passed in so ``pin_memory`` only
    fires when the actual runtime is CUDA (not merely available on the host).
    """
    if config.patch_dir is not None:
        manifest = config.patch_dir / "manifest.csv"
        train_ds = PatchDirectoryDataset(
            config.patch_dir / "train", manifest, config.normalisation,
            augment_train=config.augment_train,
        )
        val_ds = PatchDirectoryDataset(
            config.patch_dir / "val", manifest, config.normalisation,
        )
        test_ds = PatchDirectoryDataset(
            config.patch_dir / "test", manifest, config.normalisation,
        )

        if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
            raise ValueError(
                "PatchDirectoryDataset produced an empty train, validation, or test split"
            )

        def worker_init(wid: int) -> None:
            _worker_init_fn(wid, config.seed)

        def _make_loader(ds: Any, shuffle: bool) -> DataLoader:
            return DataLoader(
                ds,
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=config.num_workers,
                worker_init_fn=worker_init if config.num_workers > 0 else None,
                generator=torch.Generator().manual_seed(config.seed),
                pin_memory=device.type == "cuda",
                persistent_workers=config.num_workers > 0,
            )

        return (
            _make_loader(train_ds, shuffle=True),
            _make_loader(val_ds, shuffle=False),
            _make_loader(test_ds, shuffle=False),
        )

    # Legacy path: on-the-fly patch extraction.
    image_transform, mask_transform = get_patch_transforms()
    dataset = SatelliteTrailPatchDataset(
        root_dir=PatchDatasetConfig(
            root_dir=config.data_root,
            patch_size=config.patch_size,
            stride=config.stride,
        ),
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    train_subset, val_subset, test_subset = create_splits(
        dataset,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    if min(len(train_subset), len(val_subset), len(test_subset)) == 0:
        raise ValueError(
            "Dataset split produced an empty train, validation, or test set"
        )

    return (
        DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        ),
        DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        DataLoader(
            test_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    )


def estimate_positive_pixel_fraction(config: TrainingConfig) -> float:
    """Estimate the positive pixel fraction without building a full patch DataFrame."""
    dataset_config = PatchDatasetConfig(
        root_dir=config.data_root,
        patch_size=config.patch_size,
        stride=config.stride,
    )
    positive_pixels = 0
    total_pixels = 0

    for pair in discover_image_mask_pairs(dataset_config):
        with Image.open(pair.mask_path) as mask_image:
            mask_array = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0

        for y in range(
            0,
            pair.height - dataset_config.patch_size + 1,
            dataset_config.resolved_stride,
        ):
            for x in range(
                0,
                pair.width - dataset_config.patch_size + 1,
                dataset_config.resolved_stride,
            ):
                patch_mask = mask_array[
                    y : y + dataset_config.patch_size,
                    x : x + dataset_config.patch_size,
                ]
                positive_pixels += int(patch_mask.sum())
                total_pixels += dataset_config.patch_size * dataset_config.patch_size

    if total_pixels == 0 or positive_pixels == 0:
        raise ValueError("Cannot infer pos_weight when no positive pixels are present")

    return float(positive_pixels) / float(total_pixels)


def infer_pos_weight(config: TrainingConfig) -> float:
    """Estimate a positive class weight for the training split.

    When ``patch_dir`` is set the manifest is read directly — this is both
    faster and more accurate because it reflects the 1:3 pos/neg sampling
    applied at patch-build time rather than the raw (unsampled) image set.
    """
    if config.patch_dir is not None:
        manifest = pd.read_csv(Path(config.patch_dir) / "manifest.csv")
        train_rows = manifest[manifest["split"] == "train"]
        # Fraction of patches that contain any trail pixels (reflects 1:3 sampling).
        # Using mean pixel density would give ~0.006, not ~0.25, producing pos_weight ~170.
        pos_patch_frac = float((train_rows["positive_pixel_fraction"] > 0).mean())
        if pos_patch_frac <= 0.0:
            raise ValueError("No positive patches found in the train split manifest")
        return (1.0 - pos_patch_frac) / pos_patch_frac
    positive_fraction = estimate_positive_pixel_fraction(config)
    return (1.0 - positive_fraction) / positive_fraction


class ComboBCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation.

    ``dice_denominator_squared=True`` and ``dice_smooth=1e-4`` reproduce the
    formula in the paper's reference implementation (``asta/ASTA.py:121-139``);
    the linear-denominator alternative with ``dice_smooth=1e-6`` is the legacy
    formula used in pre-refactor baseline runs.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
        device: torch.device | None = None,
        dice_smooth: float = 1e-4,
        dice_denominator_squared: bool = True,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.dice_denominator_squared = dice_denominator_squared
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        spatial = tuple(range(2, logits.ndim))
        intersection = (probs * targets).sum(dim=spatial)
        if self.dice_denominator_squared:
            denominator = (probs ** 2).sum(dim=spatial) + (targets ** 2).sum(dim=spatial)
        else:
            denominator = probs.sum(dim=spatial) + targets.sum(dim=spatial)
        dice_loss = (
            1.0 - (2.0 * intersection + self.dice_smooth) / (denominator + self.dice_smooth)
        ).mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_loss_function(
    device: torch.device,
    pos_weight: float | None = None,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    dice_smooth: float = 1e-4,
    dice_denominator_squared: bool = True,
) -> nn.Module:
    """Construct the combined BCE + Dice segmentation loss."""
    return ComboBCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
        device=device,
        dice_smooth=dice_smooth,
        dice_denominator_squared=dice_denominator_squared,
    )


def save_checkpoint(
    model: nn.Module,
    optimiser: Adam,
    config: TrainingConfig,
    epoch: int,
    steps_completed: int,
    train_loss: float,
    val_result: DatasetEvaluationResult,
    filename: str,
    effective_eval_max_batches: int | None,
    test_result: DatasetEvaluationResult | None = None,
) -> Path:
    """Persist a model checkpoint and return its path."""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_dir / filename
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "config": _serialise_config(config),
        "epoch": epoch,
        "steps_completed": steps_completed,
        "train_loss": train_loss,
        "smoke_run": config.max_steps is not None,
        "effective_eval_max_batches": effective_eval_max_batches,
        "val_counts": asdict(val_result.counts),
        "val_metrics": asdict(val_result.metrics),
        "val_loss": val_result.mean_loss,
    }
    if test_result is not None:
        payload["test_counts"] = asdict(test_result.counts)
        payload["test_metrics"] = asdict(test_result.metrics)
        payload["test_loss"] = test_result.mean_loss

    torch.save(
        payload,
        checkpoint_path,
    )
    return checkpoint_path


def save_training_summary(
    config: TrainingConfig,
    history: list[dict[str, float | int]],
    best_checkpoint: Path,
    latest_checkpoint: Path,
    effective_eval_max_batches: int | None,
    test_result: DatasetEvaluationResult,
) -> Path:
    """Save a JSON summary of the experiment history and final test metrics."""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.checkpoint_dir / f"{config.experiment_name}_summary.json"
    summary = {
        "config": _serialise_config(config),
        "smoke_run": config.max_steps is not None,
        "effective_eval_max_batches": effective_eval_max_batches,
        "history": history,
        "best_checkpoint": str(best_checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
        "test_counts": asdict(test_result.counts),
        "test_metrics": asdict(test_result.metrics),
        "test_loss": test_result.mean_loss,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary_path


_LOG_EVERY_N_STEPS = 10


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimiser: Adam,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    steps_completed: int,
    max_steps: int | None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, int]:
    """Run one training epoch and return the mean loss and updated step count.

    Loss is accumulated as an on-device scalar; .item() is called once at end
    of epoch. Per-batch logging is throttled to every _LOG_EVERY_N_STEPS to
    avoid forcing a host/device sync on each step.
    """
    model.train()
    amp_enabled = scaler is not None and scaler.is_enabled()
    loss_total = torch.zeros((), dtype=torch.float32, device=device)
    batch_count = 0
    last_loss_value: float | None = None
    last_pos_fraction: float | None = None

    for batch in dataloader:
        if max_steps is not None and steps_completed >= max_steps:
            break

        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
        # Loss in FP32 always — Dice's spatial-sum denominator can overflow FP16
        # on 512x512 patches (max ~65504 vs sum up to 262144).
        loss = criterion(logits.float(), masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        steps_completed += 1
        batch_count += 1
        loss_total += loss.detach()

        if steps_completed % _LOG_EVERY_N_STEPS == 0:
            last_loss_value = float(loss.item())
            last_pos_fraction = float(masks.mean().item())
            logger.info(
                "Epoch %d | step %d | train_loss=%.6f | batch_positive_fraction=%.6f",
                epoch, steps_completed, last_loss_value, last_pos_fraction,
            )

    if batch_count == 0:
        raise RuntimeError("No training batches were processed")

    mean_loss = float(loss_total.item()) / batch_count
    # Ensure end-of-epoch progress is visible even if the last logged step
    # wasn't a multiple of _LOG_EVERY_N_STEPS.
    logger.info(
        "Epoch %d | end | steps=%d | mean_train_loss=%.6f",
        epoch, steps_completed, mean_loss,
    )
    return mean_loss, steps_completed


def save_training_curves(
    history: list[dict[str, float | int]],
    experiment_name: str,
    figures_dir: Path,
) -> Path:
    """Save a two-panel training curve figure (loss + val dice) to figures_dir."""
    epochs = [e["epoch"] for e in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [e["train_loss"] for e in history], label="train")
    axes[0].plot(epochs, [e["val_loss"] for e in history], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [e["val_dice"] for e in history])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Dice")
    axes[1].set_title("Validation Dice")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{experiment_name}_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_training(
    config: TrainingConfig,
    pruner_callback: Callable[[int, float], bool] | None = None,
    save_curves: bool = True,
) -> dict[str, Path]:
    """Run a train, validation, and test experiment and save the outputs."""
    seed_everything(config.seed)
    logger = get_logger(name=config.experiment_name, run_dir=config.log_dir)
    device = resolve_device(config.device)
    train_loader, val_loader, test_loader = build_dataloaders(config, device)
    eval_max_batches = config.eval_max_batches
    if eval_max_batches is None and config.max_steps is not None:
        eval_max_batches = 10

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=config.base_channels,
        dropout_rate=config.dropout_rate,
    ).to(device)
    optimiser = Adam(model.parameters(), lr=config.learning_rate)

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=config.epochs
        )

    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=config.use_amp and device.type == "cuda",
    )

    pos_weight = config.pos_weight
    if config.auto_pos_weight:
        pos_weight = infer_pos_weight(config)

    criterion = build_loss_function(
        device=device,
        pos_weight=pos_weight,
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        dice_smooth=config.dice_smooth,
        dice_denominator_squared=config.dice_denominator_squared,
    )
    logger.info("Starting U-Net experiment")
    logger.info("Training configuration: %s", _serialise_config(config))
    logger.info("Resolved device: %s", device)
    logger.info(
        "Split sizes | train=%d | val=%d | test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    if pos_weight is not None:
        logger.info("Using pos_weight=%.6f", pos_weight)
    logger.info(
        "Perf settings | AMP=%s (requested=%s) | scheduler=%s | batch_size=%d | num_workers=%d",
        scaler.is_enabled(), config.use_amp,
        config.lr_scheduler or "none",
        config.batch_size, config.num_workers,
    )
    if eval_max_batches is not None:
        logger.info(
            "Validation and test evaluation limited to %d batches",
            eval_max_batches,
        )
    if config.max_steps is not None:
        logger.warning(
            "This is a step-limited smoke run, not a full baseline experiment"
        )

    history: list[dict[str, float | int]] = []
    best_dice = -1.0
    steps_completed = 0
    latest_checkpoint = config.checkpoint_dir / f"{config.experiment_name}_latest.pth"
    best_checkpoint = config.checkpoint_dir / f"{config.experiment_name}_best.pth"
    latest_val_result: DatasetEvaluationResult | None = None

    for epoch in range(1, config.epochs + 1):
        train_loss, steps_completed = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimiser=optimiser,
            device=device,
            epoch=epoch,
            logger=logger,
            steps_completed=steps_completed,
            max_steps=config.max_steps,
            scaler=scaler,
        )
        if scheduler is not None:
            scheduler.step()
        val_result = evaluate_model_on_dataloader(
            model=model,
            dataloader=val_loader,
            device=device,
            threshold=config.threshold,
            criterion=criterion,
            max_batches=eval_max_batches,
        )
        latest_val_result = val_result

        history_entry: dict[str, float | int] = {
            "epoch": epoch,
            "steps_completed": steps_completed,
            "train_loss": train_loss,
            "val_loss": 0.0 if val_result.mean_loss is None else val_result.mean_loss,
            "val_dice": val_result.metrics.dice,
            "val_iou": val_result.metrics.iou,
        }
        history.append(history_entry)

        logger.info(
            "Epoch %d summary | train_loss=%.6f | val_loss=%.6f | val_dice=%.6f | val_iou=%.6f",
            epoch,
            train_loss,
            history_entry["val_loss"],
            val_result.metrics.dice,
            val_result.metrics.iou,
        )

        latest_checkpoint = save_checkpoint(
            model=model,
            optimiser=optimiser,
            config=config,
            epoch=epoch,
            steps_completed=steps_completed,
            train_loss=train_loss,
            val_result=val_result,
            filename=f"{config.experiment_name}_latest.pth",
            effective_eval_max_batches=eval_max_batches,
        )

        if val_result.metrics.dice > best_dice:
            best_dice = val_result.metrics.dice
            best_checkpoint = save_checkpoint(
                model=model,
                optimiser=optimiser,
                config=config,
                epoch=epoch,
                steps_completed=steps_completed,
                train_loss=train_loss,
                val_result=val_result,
                filename=f"{config.experiment_name}_best.pth",
                effective_eval_max_batches=eval_max_batches,
            )
            logger.info("Updated best checkpoint with validation Dice %.6f", best_dice)

        if config.max_steps is not None and steps_completed >= config.max_steps:
            logger.info("Reached max_steps=%d, stopping early", config.max_steps)
            break

        if pruner_callback is not None and pruner_callback(epoch, val_result.metrics.dice):
            logger.info("Trial pruned at epoch %d", epoch)
            break

    if latest_val_result is None:
        raise RuntimeError("Training completed without a validation pass")

    best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    logger.info("Reloaded best checkpoint (epoch %d) for test evaluation", best_state["epoch"])

    test_result = evaluate_model_on_dataloader(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=config.threshold,
        criterion=criterion,
        max_batches=eval_max_batches,
    )
    logger.info(
        "Final test metrics | loss=%.6f | dice=%.6f | iou=%.6f | precision=%.6f | recall=%.6f",
        0.0 if test_result.mean_loss is None else test_result.mean_loss,
        test_result.metrics.dice,
        test_result.metrics.iou,
        test_result.metrics.precision,
        test_result.metrics.recall,
    )

    # Augment the best checkpoint with the test metrics rather than re-saving
    # *_latest.pth with best weights but final-epoch metadata, which would mix
    # two epochs' state in one file and mislead later audits. The in-loop
    # *_latest.pth from the final epoch is left untouched.
    best_state["test_counts"] = asdict(test_result.counts)
    best_state["test_metrics"] = asdict(test_result.metrics)
    best_state["test_loss"] = test_result.mean_loss
    torch.save(best_state, best_checkpoint)
    logger.info("Augmented best checkpoint with test metrics: %s", best_checkpoint)

    summary_path = save_training_summary(
        config=config,
        history=history,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        effective_eval_max_batches=eval_max_batches,
        test_result=test_result,
    )
    logger.info("Latest checkpoint (from final epoch): %s", latest_checkpoint)
    logger.info("Best checkpoint (with test metrics):  %s", best_checkpoint)
    logger.info("Saved experiment summary to %s", summary_path)

    if save_curves and config.max_steps is None:
        figures_dir = config.checkpoint_dir.parent / "figures"
        curves_path = save_training_curves(history, config.experiment_name, figures_dir)
        logger.info("Saved training curves to %s", curves_path)

    return {
        "latest_checkpoint": latest_checkpoint,
        "best_checkpoint": best_checkpoint,
        "summary_path": summary_path,
    }


def main() -> None:
    """Parse CLI arguments and run the U-Net experiment."""
    run_training(parse_args())


if __name__ == "__main__":
    main()
