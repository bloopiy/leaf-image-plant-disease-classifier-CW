from pathlib import Path

import torch
from torch import nn


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss encountered.")

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("DataLoader produced no examples.")

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


def fit(model, train_loader, val_loader, cfg, device, logger=None):
    """Train, save the best validation-loss checkpoint, and restore it."""
    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg.get("patience", 5))

    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must both be positive.")

    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = Path(
        train_cfg.get("checkpoint_path", "checkpoints/best.pt")
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_loss = float("inf")
    best_epoch = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = _run_epoch(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        if logger is not None:
            for title, series, value in (
                ("loss", "train", train_loss),
                ("loss", "validation", val_loss),
                ("accuracy", "train", train_accuracy),
                ("accuracy", "validation", val_accuracy),
            ):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=value,
                    iteration=epoch,
                )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.3f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "checkpoint_path": str(checkpoint_path),
    }
