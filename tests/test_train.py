import argparse
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import plantdisease.train as training

ROOT = Path(__file__).resolve().parents[1]


class RecordingLogger:
    def __init__(self):
        self.entries = []

    def report_scalar(self, **entry):
        self.entries.append(entry)


def load_train_script():
    spec = importlib.util.spec_from_file_location(
        "train_script", ROOT / "scripts" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_args(**overrides):
    values = {
        "config": str(ROOT / "configs" / "default.yaml"),
        "fraction": None,
        "full_pool": False,
        "epochs": None,
        "batch_size": None,
        "num_workers": 0,
        "checkpoint": None,
        "offline": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def train_cfg(tmp_path, **overrides):
    values = {
        "epochs": 1,
        "lr": 0.01,
        "patience": 1,
        "checkpoint_path": str(tmp_path / "best.pt"),
    }
    values.update(overrides)
    return {"train": values}


def fake_epochs(monkeypatch, validation_results, step_optimizer=False):
    results = iter(validation_results)

    def fake_epoch(model, loader, criterion, device, optimizer=None):
        if optimizer is not None:
            if step_optimizer:
                optimizer.zero_grad(set_to_none=True)
                optimizer.step()
            return 0.5, 0.5
        return next(results)

    monkeypatch.setattr(training, "_run_epoch", fake_epoch)


def test_fit_updates_saves_logs_and_restores_best(tmp_path):
    torch.manual_seed(7)

    images = torch.randn(32, 4)
    labels = (images[:, 0] > 0).long()

    train_loader = DataLoader(
        TensorDataset(images[:24], labels[:24]),
        batch_size=7,
    )
    val_loader = DataLoader(
        TensorDataset(images[24:], labels[24:]),
        batch_size=3,
    )

    model = torch.nn.Linear(4, 2)
    initial = {
        name: value.clone()
        for name, value in model.state_dict().items()
    }
    logger = RecordingLogger()
    path = tmp_path / "best.pt"

    cfg = {"train": {
        "epochs": 3,
        "lr": 0.05,
        "patience": 3,
        "checkpoint_path": str(path),
    }}

    result = training.fit(
        model, train_loader, val_loader, cfg, "cpu", logger
    )

    assert path.is_file()
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    assert any(
        not torch.equal(initial[name], value)
        for name, value in model.state_dict().items()
    )
    assert checkpoint["epoch"] == result["best_epoch"]

    for name, value in model.state_dict().items():
        assert torch.equal(
            value, checkpoint["model_state_dict"][name]
        )

    with torch.no_grad():
        reference_loss = torch.nn.functional.cross_entropy(
            model(images[24:]), labels[24:]
        ).item()

    assert result["best_val_loss"] == pytest.approx(
        reference_loss, abs=1e-6
    )
    assert len(logger.entries) == 5 * len(result["history"])

    lr_entries = [
        entry for entry in logger.entries if entry["title"] == "lr"
    ]
    assert len(lr_entries) == len(result["history"])
    assert all(entry["series"] == "train" for entry in lr_entries)
    assert all(
        entry["value"] == pytest.approx(0.05) for entry in lr_entries
    )


def test_early_stopping_restores_best_not_last(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()

    validation_losses = iter([1.0, 0.8, 0.9, 1.1])

    def fake_epoch(model, loader, criterion, device, optimizer=None):
        if optimizer is not None:
            with torch.no_grad():
                model.weight.add_(1.0)
            return 0.5, 0.5

        return next(validation_losses), 0.5

    monkeypatch.setattr(training, "_run_epoch", fake_epoch)

    cfg = {"train": {
        "epochs": 10,
        "lr": 0.01,
        "patience": 2,
        "checkpoint_path": str(tmp_path / "best.pt"),
    }}

    result = training.fit(model, [], [], cfg, "cpu")

    assert len(result["history"]) == 4
    assert result["best_epoch"] == 2
    assert result["best_val_loss"] == pytest.approx(0.8)
    assert torch.equal(
        model.weight, torch.full_like(model.weight, 2.0)
    )


@pytest.mark.parametrize(
    ("optimizer_name", "expected_class"),
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        (None, torch.optim.AdamW),
    ],
)
def test_optimizer_selection(
    tmp_path, monkeypatch, optimizer_name, expected_class
):
    seen = {}

    def fake_epoch(model, loader, criterion, device, optimizer=None):
        if optimizer is not None:
            seen["optimizer"] = optimizer
            return 0.5, 0.5
        return 0.5, 0.5

    monkeypatch.setattr(training, "_run_epoch", fake_epoch)

    overrides = {"weight_decay": 0.1}
    if optimizer_name is not None:
        overrides["optimizer"] = optimizer_name

    training.fit(
        torch.nn.Linear(1, 2),
        [],
        [],
        train_cfg(tmp_path, **overrides),
        "cpu",
    )

    optimizer = seen["optimizer"]
    assert isinstance(optimizer, expected_class)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)


def test_invalid_optimizer_raises(tmp_path):
    with pytest.raises(ValueError, match="optimizer"):
        training.fit(
            torch.nn.Linear(1, 2),
            [],
            [],
            train_cfg(tmp_path, optimizer="sgd"),
            "cpu",
        )


def test_cosine_scheduler_steps_and_is_saved(tmp_path, monkeypatch):
    fake_epochs(monkeypatch, [(0.5, 0.5)] * 3, step_optimizer=True)
    logger = RecordingLogger()

    result = training.fit(
        torch.nn.Linear(1, 2),
        [],
        [],
        train_cfg(
            tmp_path, epochs=3, lr=0.1, patience=10, scheduler="cosine"
        ),
        "cpu",
        logger,
    )

    lrs = [
        entry["value"]
        for entry in logger.entries
        if entry["title"] == "lr"
    ]
    assert len(lrs) == 3
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[1] < lrs[0]
    assert lrs[2] < lrs[1]

    checkpoint = torch.load(
        tmp_path / "best.pt", map_location="cpu", weights_only=True
    )
    assert "scheduler_state_dict" in checkpoint
    assert result["best_val_accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize("scheduler", [None, "none"])
def test_no_scheduler_state_by_default(tmp_path, monkeypatch, scheduler):
    fake_epochs(monkeypatch, [(0.5, 0.5)])

    overrides = {}
    if scheduler is not None:
        overrides["scheduler"] = scheduler

    training.fit(
        torch.nn.Linear(1, 2),
        [],
        [],
        train_cfg(tmp_path, **overrides),
        "cpu",
    )

    checkpoint = torch.load(
        tmp_path / "best.pt", map_location="cpu", weights_only=True
    )
    assert "scheduler_state_dict" not in checkpoint


def test_invalid_scheduler_raises(tmp_path):
    with pytest.raises(ValueError, match="scheduler"):
        training.fit(
            torch.nn.Linear(1, 2),
            [],
            [],
            train_cfg(tmp_path, scheduler="step"),
            "cpu",
        )


def test_accuracy_improvement_resets_patience(tmp_path, monkeypatch):
    fake_epochs(
        monkeypatch,
        [(1.0, 0.5), (1.1, 0.6), (1.2, 0.7), (1.3, 0.8)],
    )

    result = training.fit(
        torch.nn.Linear(1, 2),
        [],
        [],
        train_cfg(tmp_path, epochs=4, patience=2),
        "cpu",
    )

    assert len(result["history"]) == 4
    assert result["best_epoch"] == 4
    assert result["best_val_loss"] == pytest.approx(1.3)
    assert result["best_val_accuracy"] == pytest.approx(0.8)


def test_accuracy_plateau_early_stops(tmp_path, monkeypatch):
    fake_epochs(monkeypatch, [(1.0, 0.5)] * 5)

    result = training.fit(
        torch.nn.Linear(1, 2),
        [],
        [],
        train_cfg(tmp_path, epochs=10, patience=2),
        "cpu",
    )

    assert len(result["history"]) == 3
    assert result["best_epoch"] == 1
    assert result["best_val_accuracy"] == pytest.approx(0.5)


def test_accuracy_drop_is_not_improvement_even_if_loss_falls(
    tmp_path, monkeypatch
):
    model = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()

    validation_results = iter([
        (1.0, 0.5),
        (0.5, 0.1),
        (0.2, 0.05),
    ])

    def fake_epoch(model, loader, criterion, device, optimizer=None):
        if optimizer is not None:
            with torch.no_grad():
                model.weight.add_(1.0)
            return 0.5, 0.5
        return next(validation_results)

    monkeypatch.setattr(training, "_run_epoch", fake_epoch)

    cfg = {"train": {
        "epochs": 3,
        "lr": 0.01,
        "patience": 1,
        "checkpoint_path": str(tmp_path / "best.pt"),
    }}

    result = training.fit(model, [], [], cfg, "cpu")

    assert len(result["history"]) == 2
    assert result["best_epoch"] == 1
    assert result["best_val_loss"] == pytest.approx(1.0)
    assert result["best_val_accuracy"] == pytest.approx(0.5)
    assert torch.equal(
        model.weight, torch.full_like(model.weight, 1.0)
    )

    checkpoint = torch.load(
        tmp_path / "best.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["epoch"] == 1


def test_equal_accuracy_prefers_lower_loss(tmp_path, monkeypatch):
    fake_epochs(monkeypatch, [(1.0, 0.5), (0.8, 0.5), (0.9, 0.5)])

    result = training.fit(
        torch.nn.Linear(1, 2),
        [],
        [],
        train_cfg(tmp_path, epochs=3, patience=3),
        "cpu",
    )

    assert result["best_epoch"] == 2
    assert result["best_val_loss"] == pytest.approx(0.8)

    checkpoint = torch.load(
        tmp_path / "best.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["val_loss"] == pytest.approx(0.8)
    assert checkpoint["val_accuracy"] == pytest.approx(0.5)


def test_checkpoint_has_no_config_and_loads_weights_only(
    tmp_path, monkeypatch
):
    fake_epochs(monkeypatch, [(0.5, 0.75)])

    training.fit(
        torch.nn.Linear(1, 2),
        [],
        [],
        train_cfg(tmp_path),
        "cpu",
    )

    checkpoint = torch.load(
        tmp_path / "best.pt", map_location="cpu", weights_only=True
    )
    assert "config" not in checkpoint
    assert set(checkpoint) == {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "val_loss",
        "val_accuracy",
    }
    assert checkpoint["val_accuracy"] == pytest.approx(0.75)


def test_train_script_rejects_config_without_model_section(tmp_path):
    script = load_train_script()
    config_path = tmp_path / "transfer.yaml"
    config_path.write_text(
        "data: {}\ntrain: {}\nclearml: {}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="has no 'model' section yet"):
        script.build_config(script_args(config=str(config_path)))


def test_train_script_applies_cli_overrides(tmp_path):
    script = load_train_script()
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "\n".join([
            "seed: 1",
            "model: {}",
            "data:",
            "  prototype_fraction: 0.25",
            "  use_full_development_pool: false",
            "train:",
            "  epochs: 30",
            "  batch_size: 32",
            "  checkpoint_path: old.pt",
            "clearml:",
            "  project_name: PlantDisease-CW",
            "  task_name_prefix: custom",
            "  offline: false",
        ]) + "\n",
        encoding="utf-8",
    )

    cfg = script.build_config(script_args(
        config=str(config_path),
        fraction=0.5,
        full_pool=True,
        epochs=2,
        batch_size=8,
        checkpoint="new.pt",
    ))

    assert cfg["data"]["prototype_fraction"] == 0.5
    assert cfg["data"]["use_full_development_pool"] is True
    assert cfg["train"]["epochs"] == 2
    assert cfg["train"]["batch_size"] == 8
    assert cfg["train"]["checkpoint_path"] == str(ROOT / "new.pt")

    absolute = str(tmp_path / "absolute.pt")
    cfg = script.build_config(script_args(
        config=str(config_path), checkpoint=absolute
    ))
    assert cfg["train"]["checkpoint_path"] == absolute
