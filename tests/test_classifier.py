from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models.classifier import PatchClassifier  # noqa: E402
from src.training import train_classifier  # noqa: E402
from src.training.train_classifier import (  # noqa: E402
    ClassifierConfig,
    ClassifierDataset,
    compute_pos_weight,
    save_classifier_summary,
    select_operating_points,
)


def _write_manifest_fixture(tmp_path: Path) -> Path:
    patch_root = tmp_path / "patches"
    train_dir = patch_root / "train"
    train_dir.mkdir(parents=True)
    rows = []
    for idx, pos_frac in enumerate([0.0, 0.25]):
        image_path = train_dir / f"patch{idx}_image.png"
        Image.fromarray(np.full((8, 8), 128 + idx, dtype=np.uint8)).save(image_path)
        rows.append(
            {
                "split": "train",
                "source_image": f"source_{idx}.png",
                "patch_path": str(image_path),
                "mask_path": str(train_dir / f"patch{idx}_mask.png"),
                "positive_pixel_fraction": pos_frac,
                "image_mean": 0.5,
                "image_std": 0.25,
            }
        )
    manifest_path = patch_root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_patch_classifier_outputs_one_logit_per_patch() -> None:
    model = PatchClassifier().eval()
    inputs = torch.randn(2, 1, 528, 528)

    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (2,)


def test_classifier_dataset_labels_and_full_image_normalisation(tmp_path: Path) -> None:
    manifest_path = _write_manifest_fixture(tmp_path)

    dataset = ClassifierDataset(
        patch_dir=manifest_path.parent,
        split="train",
        normalisation="full_image",
        manifest_path=manifest_path,
    )

    first = dataset[0]
    second = dataset[1]
    expected = ((128 / 255.0) - 0.5) / (0.25 + 1e-6)
    assert first["label"].item() == 0.0
    assert second["label"].item() == 1.0
    assert torch.allclose(first["image"].mean(), torch.tensor(expected), atol=1e-4)


def test_classifier_pos_weight_uses_train_split_class_balance() -> None:
    records = pd.DataFrame({"positive_pixel_fraction": [0.0, 0.0, 0.0, 0.1]})

    pos_weight = compute_pos_weight(records)

    assert pos_weight.item() == pytest.approx(3.0)


def test_classifier_config_yaml_loads_without_test_eval_flag() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "classifier_base.yaml"
    config = ClassifierConfig(**yaml.safe_load(path.read_text()))

    assert config.patch_dir == Path("data/patches")
    assert config.normalisation == "full_image"
    assert config.batch_size == 32
    assert not hasattr(config, "skip_test_eval")


def test_classifier_threshold_helper_returns_max_f1_and_high_recall_points() -> None:
    probs = np.array([0.95, 0.70, 0.40, 0.20], dtype=float)
    labels = np.array([1, 1, 0, 0], dtype=float)
    thresholds = np.array([0.35, 0.65, 0.85], dtype=float)

    selected = select_operating_points(
        probs,
        labels,
        high_recall_target=0.99,
        thresholds=thresholds,
    )

    assert selected["max_f1"]["threshold"] == pytest.approx(0.65)
    assert selected["high_recall"]["threshold"] == pytest.approx(0.65)
    assert selected["high_recall"]["threshold"] >= selected["max_f1"]["threshold"]
    assert selected["auc_roc"] == pytest.approx(1.0)


def test_classifier_summary_omits_test_keys_and_phase_a_has_no_test_loop(tmp_path: Path) -> None:
    config = ClassifierConfig(
        patch_dir=tmp_path / "patches",
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
    )
    summary = save_classifier_summary(
        config=config,
        history=[
            {
                "epoch": 1,
                "train_loss": 0.5,
                "val_max_f1": {"f1": 0.8},
                "val_high_recall": {"recall": 1.0},
            }
        ],
        best_checkpoint=tmp_path / "best.pth",
        latest_checkpoint=tmp_path / "latest.pth",
    )
    data = yaml.safe_load(summary.read_text())
    source = Path(train_classifier.__file__).read_text()

    assert not any(key.startswith("test") for key in data)
    assert "test_loader" not in source
    assert "skip_test_eval" not in source
