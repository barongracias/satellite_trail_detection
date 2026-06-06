"""Train a validation-only satellite-trail patch classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from src.config.constants import GLOBAL_SEED
from src.data.transforms import NORMALISATION_MODES, normalise_tensor
from src.models.classifier import PatchClassifier
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


Image.MAX_IMAGE_PIXELS = None


TEST_KEYS = {"test_precision", "test_recall", "test_f1", "test_fnr", "test_auc", "test_metrics"}


@dataclass(frozen=True)
class ClassifierConfig:
    """Runtime configuration for classifier training."""

    patch_dir: str | Path
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 30
    max_steps: int | None = None
    num_workers: int = 4
    normalisation: str = "full_image"
    high_recall_target: float = 0.99
    device: str = "auto"
    checkpoint_dir: str | Path = Path("results/checkpoints")
    log_dir: str | Path = Path("results/logs")
    experiment_name: str = "classifier_base"
    seed: int = GLOBAL_SEED

    def __post_init__(self) -> None:
        object.__setattr__(self, "patch_dir", Path(self.patch_dir))
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        object.__setattr__(self, "log_dir", Path(self.log_dir))
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.normalisation not in NORMALISATION_MODES:
            raise ValueError(f"normalisation must be one of {NORMALISATION_MODES}")
        if self.high_recall_target <= 0 or self.high_recall_target > 1:
            raise ValueError("high_recall_target must lie in (0, 1]")


class ClassifierDataset(Dataset[dict[str, torch.Tensor]]):
    """Read classifier images and labels directly from a patch manifest."""

    def __init__(
        self,
        patch_dir: str | Path,
        split: str,
        normalisation: str = "full_image",
        manifest_path: str | Path | None = None,
    ) -> None:
        self.patch_dir = Path(patch_dir)
        self.split = split
        self.normalisation = normalisation
        if manifest_path is None:
            manifest_path = self.patch_dir / "manifest.csv"
        manifest = pd.read_csv(manifest_path)
        self.records = manifest[manifest["split"] == split].reset_index(drop=True)
        self._rows = self.records.to_dict("records")
        if len(self.records) == 0:
            raise ValueError(f"manifest contains no rows for split={split!r}")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._rows[idx]
        with Image.open(row["patch_path"]) as img:
            image = TF.pil_to_tensor(img.convert("L")).float() / 255.0
        image = normalise_tensor(
            image,
            self.normalisation,
            full_image_mean=row.get("image_mean"),
            full_image_std=row.get("image_std"),
        )
        label = torch.tensor(float(row["positive_pixel_fraction"] > 0), dtype=torch.float32)
        return {"image": image, "label": label}


def parse_args() -> ClassifierConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to classifier YAML config.",
    )
    parser.add_argument(
        "--patch_dir",
        type=Path,
        default=None,
        help="Override patch_dir from the YAML config.",
    )
    args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text())
    if args.patch_dir is not None:
        data["patch_dir"] = str(args.patch_dir)
    return ClassifierConfig(**data)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _serialise_config(config: ClassifierConfig) -> dict[str, Any]:
    data = asdict(config)
    data["patch_dir"] = str(config.patch_dir)
    data["checkpoint_dir"] = str(config.checkpoint_dir)
    data["log_dir"] = str(config.log_dir)
    return data


def compute_pos_weight(records: pd.DataFrame) -> torch.Tensor:
    labels = records["positive_pixel_fraction"] > 0
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("classifier training requires both positive and negative patches")
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def _worker_init_fn(worker_id: int, seed: int) -> None:
    seed_everything(seed + worker_id)


def build_dataloaders(config: ClassifierConfig, device: torch.device) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    train_ds = ClassifierDataset(config.patch_dir, "train", config.normalisation)
    val_ds = ClassifierDataset(config.patch_dir, "val", config.normalisation)
    pos_weight = compute_pos_weight(train_ds.records)

    def worker_init(worker_id: int) -> None:
        _worker_init_fn(worker_id, config.seed)

    def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
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

    return make_loader(train_ds, True), make_loader(val_ds, False), pos_weight


def _metrics_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred = probs >= threshold
    target = labels.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fnr = fn / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fnr": fnr,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def _roc_auc(probs: np.ndarray, labels: np.ndarray) -> float | None:
    target = labels.astype(bool)
    n_pos = int(target.sum())
    n_neg = int((~target).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(probs)
    sorted_probs = probs[order]
    ranks = np.empty(len(probs), dtype=np.float64)
    start = 0
    while start < len(probs):
        end = start + 1
        while end < len(probs) and sorted_probs[end] == sorted_probs[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = float(ranks[target].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def select_operating_points(
    probs: np.ndarray,
    labels: np.ndarray,
    high_recall_target: float = 0.99,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.955, 0.01), 2)
    rows = [_metrics_at_threshold(probs, labels, float(t)) for t in thresholds]
    max_f1 = max(rows, key=lambda row: (row["f1"], row["threshold"]))
    high_recall_candidates = [
        row for row in rows if row["recall"] >= high_recall_target
    ]
    high_recall = (
        max(high_recall_candidates, key=lambda row: (row["threshold"], row["f1"]))
        if high_recall_candidates
        else max_f1
    )
    return {
        "max_f1": max_f1,
        "high_recall": high_recall,
        "high_recall_target": high_recall_target,
        "auc_roc": _roc_auc(probs, labels),
        "thresholds": rows,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimiser: Adam,
    device: torch.device,
    max_steps: int | None,
    steps_completed: int,
) -> tuple[float, int]:
    model.train()
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    batches = 0
    for batch in dataloader:
        if max_steps is not None and steps_completed >= max_steps:
            break
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        labels = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.detach()
        batches += 1
        steps_completed += 1
    if batches == 0:
        raise RuntimeError("No classifier training batches were processed")
    return float(total_loss.item()) / batches, steps_completed


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    high_recall_target: float,
) -> dict[str, Any]:
    model.eval()
    probs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(images)
            probs.append(torch.sigmoid(logits).cpu())
            labels.append(batch["label"].float().cpu())
    probs_np = torch.cat(probs).numpy()
    labels_np = torch.cat(labels).numpy()
    return select_operating_points(probs_np, labels_np, high_recall_target)


def save_checkpoint(
    model: nn.Module,
    optimiser: Adam,
    config: ClassifierConfig,
    epoch: int,
    train_loss: float,
    val_metrics: dict[str, Any],
    filename: str,
) -> Path:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = config.checkpoint_dir / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "config": _serialise_config(config),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        },
        path,
    )
    return path


def save_classifier_summary(
    config: ClassifierConfig,
    history: list[dict[str, Any]],
    best_checkpoint: Path,
    latest_checkpoint: Path,
) -> Path:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = config.checkpoint_dir / f"{config.experiment_name}_summary.json"
    payload = {
        "config": _serialise_config(config),
        "history": history,
        "best_checkpoint": str(best_checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
    }
    present_test_keys = TEST_KEYS.intersection(payload)
    if present_test_keys:
        raise RuntimeError(f"classifier summary contains test keys: {sorted(present_test_keys)}")
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    return path


def run_training(config: ClassifierConfig) -> dict[str, Path]:
    seed_everything(config.seed)
    logger = get_logger(config.experiment_name, run_dir=config.log_dir)
    device = resolve_device(config.device)
    train_loader, val_loader, pos_weight = build_dataloaders(config, device)
    model = PatchClassifier().to(device)
    optimiser = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    steps_completed = 0
    best_checkpoint = config.checkpoint_dir / f"{config.experiment_name}_best.pth"
    latest_checkpoint = config.checkpoint_dir / f"{config.experiment_name}_latest.pth"

    logger.info("Starting classifier training")
    logger.info("Training configuration: %s", _serialise_config(config))
    logger.info("Split sizes | train=%d | val=%d", len(train_loader.dataset), len(val_loader.dataset))
    logger.info("Train pos_weight=%.4f", float(pos_weight.item()))

    for epoch in range(1, config.epochs + 1):
        train_loss, steps_completed = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimiser,
            device,
            config.max_steps,
            steps_completed,
        )
        val_metrics = evaluate_classifier(model, val_loader, device, config.high_recall_target)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc_roc": val_metrics["auc_roc"],
            "val_max_f1": val_metrics["max_f1"],
            "val_high_recall": val_metrics["high_recall"],
        }
        history.append(row)
        latest_checkpoint = save_checkpoint(
            model,
            optimiser,
            config,
            epoch,
            train_loss,
            val_metrics,
            f"{config.experiment_name}_latest.pth",
        )
        if val_metrics["max_f1"]["f1"] > best_f1:
            best_f1 = float(val_metrics["max_f1"]["f1"])
            best_checkpoint = save_checkpoint(
                model,
                optimiser,
                config,
                epoch,
                train_loss,
                val_metrics,
                f"{config.experiment_name}_best.pth",
            )
        logger.info(
            "Epoch %d | train_loss=%.6f | val_max_f1=%.4f | val_high_recall_fnr=%.4f",
            epoch,
            train_loss,
            val_metrics["max_f1"]["f1"],
            val_metrics["high_recall"]["fnr"],
        )

    summary_path = save_classifier_summary(config, history, best_checkpoint, latest_checkpoint)
    return {
        "best_checkpoint": best_checkpoint,
        "latest_checkpoint": latest_checkpoint,
        "summary_path": summary_path,
    }


def main() -> None:
    config = parse_args()
    results = run_training(config)
    print(f"Best checkpoint: {results['best_checkpoint']}")
    print(f"Summary: {results['summary_path']}")


if __name__ == "__main__":
    main()
