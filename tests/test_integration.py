"""Cross-component integration tests.

Offline and fast: random tensors, no dataset download, no pretrained weights.
They push data through the real code the way the project does:

    build_model -> fit -> checkpoint -> app model loader -> evaluate

and check that the config files contain what every component reads.

NOTE: the app's `load_model(cfg, weights_path) -> (model, cfg)` is used as in
app/app.py. If its signature changes, adjust `reload_like_the_app` below.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))  # for the app's model_loader module

from plantdisease.evaluate import evaluate
from plantdisease.models import build_model
from plantdisease.train import fit
from plantdisease.utils import load_config

NUM_CLASSES = 39
CONFIGS = ["default.yaml", "transfer.yaml"]
_MISSING = object()

# Keys that some component reads, so every config must provide them.
REQUIRED_KEYS = [
    ("seed",),
    ("data", "dataset", "id"),     # load_and_split
    ("model", "num_classes"),      # build_model, app, evaluate
    ("model", "input_size"),       # loaders and app preprocessing
    ("train", "epochs"),           # fit
    ("train", "lr"),               # fit
]
# Names the "train" section may contain. fit() reads the first group; the
# second group is used elsewhere or planned.
FIT_READS = {
    "epochs", "lr", "weight_decay", "patience", "checkpoint_path",
    "optimizer", "scheduler",
}
USED_ELSEWHERE = {"batch_size"}


def dig(cfg, path):
    for key in path:
        if not isinstance(cfg, dict) or key not in cfg:
            return _MISSING
        cfg = cfg[key]
    return cfg


@pytest.mark.parametrize("name", CONFIGS)
def test_config_has_the_keys_every_component_reads(name):
    cfg = load_config(ROOT / "configs" / name)
    missing = [".".join(p) for p in REQUIRED_KEYS if dig(cfg, p) is _MISSING]
    assert not missing, f"{name} is missing: {missing}"


@pytest.mark.parametrize("name", CONFIGS)
def test_train_section_only_uses_names_fit_understands(name):
    train = load_config(ROOT / "configs" / name).get("train", {})
    unknown = set(train) - FIT_READS - USED_ELSEWHERE
    assert not unknown, (
        f"{name}: fit() ignores these train keys: {sorted(unknown)} "
        "(for example it reads 'patience', not 'early_stopping_patience')"
    )


def tiny_cfg(kind, tmp_path):
    """The project's own config, shrunk so the test runs in seconds."""
    cfg = copy.deepcopy(
        load_config(ROOT / "configs" / ("default.yaml" if kind == "cnn" else "transfer.yaml"))
    )
    if kind == "cnn":
        cfg["model"].update(input_size=32, channels=[4, 8], dense_units=[8])
    else:
        cfg["model"].update(input_size=64, pretrained=False)  # no weight download
    cfg["train"] = {
        "epochs": 2, "lr": 1e-3, "patience": 2,
        "checkpoint_path": str(tmp_path / "best.pt"),
    }
    return cfg


def random_loaders(size):
    gen = torch.Generator().manual_seed(0)

    def dataset(n):
        images = torch.rand(n, 3, size, size, generator=gen)
        labels = torch.randint(0, NUM_CLASSES, (n,), generator=gen)
        return TensorDataset(images, labels)

    train = DataLoader(
        dataset(64), batch_size=16, shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    return train, DataLoader(dataset(32), batch_size=16)  # no shuffle: order matters for evaluate


def reload_like_the_app(cfg, checkpoint):
    from model_loader import load_model

    return load_model(cfg, checkpoint)


@pytest.mark.parametrize("kind", ["cnn", "resnet18"])
def test_trained_model_survives_save_load_and_evaluation(kind, tmp_path):
    cfg = tiny_cfg(kind, tmp_path)
    device = torch.device("cpu")
    torch.manual_seed(0)
    try:
        model = build_model(cfg)
    except KeyError as error:
        pytest.fail(
            f"build_model cannot build '{kind}' from its config (missing key {error}). "
            "Add the resnet18 case to build_model."
        )

    train_loader, val_loader = random_loaders(cfg["model"]["input_size"])
    result = fit(model, train_loader, val_loader, cfg, device)
    checkpoint = Path(result["checkpoint_path"])
    assert checkpoint.is_file()

    direct = evaluate(model, val_loader, device, num_classes=NUM_CLASSES)
    assert direct["confusion_matrix"].shape == (NUM_CLASSES, NUM_CLASSES)
    assert direct["confusion_matrix"].sum() == direct["num_samples"] == 32
    assert 0 <= direct["accuracy"] <= direct["top5_accuracy"] <= 1

    # Hand the app the WRONG fallback config (default.yaml). The checkpoint's
    # own config must win, otherwise a ResNet checkpoint would be built as a CNN.
    fallback = load_config(ROOT / "configs" / "default.yaml")
    reloaded, loaded_cfg = reload_like_the_app(fallback, checkpoint)
    assert loaded_cfg["model"]["num_classes"] == NUM_CLASSES
    assert loaded_cfg["model"]["input_size"] == cfg["model"]["input_size"]
    assert type(reloaded) is type(model)

    again = evaluate(reloaded, val_loader, device, num_classes=NUM_CLASSES)
    assert np.array_equal(
        direct["predictions"]["preds"], again["predictions"]["preds"]
    ), "the app's reloaded model predicts differently from the trained one"


@pytest.mark.real_data
def test_app_class_names_match_the_dataset_label_order(raw_dataset):
    """A mismatch would show wrong disease names without any error."""
    from model_loader import load_class_names

    dataset_names = raw_dataset.features["label"].names
    app_names = load_class_names(ROOT / "app" / "class_names.json")
    assert list(app_names) == list(dataset_names)
