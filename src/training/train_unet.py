"""U-Net training entry point for local validation runs and early experiments."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.dataset import PatchDatasetConfig, SatelliteTrailPatchDataset
from src.data.indexing import discover_image_mask_pairs
from src.data.splits import create_splits
from src.data.transforms import get_patch_transforms
from src.evaluation.segmentation import (
    DatasetEvaluationResult,
    evaluate_model_on_dataloader,
)
from src.models.unet import UNet
from src.utils.logger import get_logger


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
    pos_weight: float | None = None
    auto_pos_weight: bool = False
    base_channels: int = 32
    device: str = "auto"
    checkpoint_dir: str | Path = Path("results/checkpoints")
    log_dir: str | Path = Path("results/logs")
    experiment_name: str = "unet_baseline"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    threshold: float = 0.5
    eval_max_batches: int | None = None

    def __post_init__(self) -> None:
        """Normalise paths and validate the few settings that must be sane."""
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        object.__setattr__(self, "log_dir", Path(self.log_dir))

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


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the U-Net training entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/subset/processed"),
        help="Directory containing processed image and mask PNG pairs.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Square patch size used by the dataset.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Patch stride used by the dataset. Defaults to patch size.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional global optimiser-step limit for a short smoke run.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Optional positive class weight for BCEWithLogitsLoss.",
    )
    parser.add_argument(
        "--auto-pos-weight",
        action="store_true",
        help="Estimate pos_weight from the processed dataset masks.",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/checkpoints"),
        help="Directory where checkpoints and summaries will be saved.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/logs"),
        help="Directory where training logs will be saved.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="unet_experiment",
        help="Name used for logs, checkpoints, and summary files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=None,
        help="Optional limit on validation and test batches for quick smoke runs.",
    )
    return parser


def parse_args() -> TrainingConfig:
    """Parse CLI arguments into a validated configuration object."""
    return TrainingConfig(**vars(build_parser().parse_args()))


def _serialise_config(config: TrainingConfig) -> dict[str, Any]:
    """Convert a training configuration into a JSON-friendly dictionary."""
    data = asdict(config)
    data["data_root"] = str(config.data_root)
    data["checkpoint_dir"] = str(config.checkpoint_dir)
    data["log_dir"] = str(config.log_dir)
    return data


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested runtime device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_dataloaders(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create deterministic train, validation, and test dataloaders."""
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
    """Estimate a positive class weight from the processed mask coverage."""
    positive_fraction = estimate_positive_pixel_fraction(config)
    return (1.0 - positive_fraction) / positive_fraction


def build_loss_function(
    device: torch.device,
    pos_weight: float | None,
) -> nn.Module:
    """Construct the binary segmentation loss function."""
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()

    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)


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
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


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
) -> tuple[float, int]:
    """Run one training epoch and return the mean loss and updated step count."""
    model.train()
    batch_losses: list[float] = []

    for batch in dataloader:
        if max_steps is not None and steps_completed >= max_steps:
            break

        images = batch["image"].to(device=device, dtype=torch.float32)
        masks = batch["mask"].to(device=device, dtype=torch.float32)

        optimiser.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimiser.step()

        steps_completed += 1
        loss_value = float(loss.item())
        batch_losses.append(loss_value)

        logger.info(
            "Epoch %d | step %d | train_loss=%.6f | batch_positive_fraction=%.6f",
            epoch,
            steps_completed,
            loss_value,
            float(masks.mean().item()),
        )

    if not batch_losses:
        raise RuntimeError("No training batches were processed")

    return sum(batch_losses) / len(batch_losses), steps_completed


def run_training(config: TrainingConfig) -> dict[str, Path]:
    """Run a train, validation, and test experiment and save the outputs."""
    logger = get_logger(name=config.experiment_name, run_dir=config.log_dir)
    device = resolve_device(config.device)
    train_loader, val_loader, test_loader = build_dataloaders(config)
    eval_max_batches = config.eval_max_batches
    if eval_max_batches is None and config.max_steps is not None:
        eval_max_batches = 10

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=config.base_channels,
    ).to(device)
    optimiser = Adam(model.parameters(), lr=config.learning_rate)

    pos_weight = config.pos_weight
    if config.auto_pos_weight:
        pos_weight = infer_pos_weight(config)

    criterion = build_loss_function(device=device, pos_weight=pos_weight)
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
        )
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

    if latest_val_result is None:
        raise RuntimeError("Training completed without a validation pass")

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

    latest_checkpoint = save_checkpoint(
        model=model,
        optimiser=optimiser,
        config=config,
        epoch=int(history[-1]["epoch"]),
        steps_completed=steps_completed,
        train_loss=float(history[-1]["train_loss"]),
        val_result=latest_val_result,
        filename=f"{config.experiment_name}_latest.pth",
        effective_eval_max_batches=eval_max_batches,
        test_result=test_result,
    )
    summary_path = save_training_summary(
        config=config,
        history=history,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        effective_eval_max_batches=eval_max_batches,
        test_result=test_result,
    )
    logger.info("Saved latest checkpoint to %s", latest_checkpoint)
    logger.info("Saved best checkpoint to %s", best_checkpoint)
    logger.info("Saved experiment summary to %s", summary_path)

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
