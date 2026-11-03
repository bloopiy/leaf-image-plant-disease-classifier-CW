import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import plantdisease.train as training


class RecordingLogger:
    def __init__(self):
        self.entries = []

    def report_scalar(self, **entry):
        self.entries.append(entry)


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
    assert len(logger.entries) == 4 * len(result["history"])


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
