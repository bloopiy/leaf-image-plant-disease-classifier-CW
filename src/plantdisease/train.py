"""Training loop with per-epoch metrics and early stopping."""
import torch
import torch.nn as nn


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over loader. Trains if optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)  # sets Dropout/BatchNorm behaviour explicitly
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def build_optimizer(model, tcfg: dict):
    params = [p for p in model.parameters() if p.requires_grad]
    cls = torch.optim.AdamW if tcfg["optimizer"] == "adamw" else torch.optim.Adam
    return cls(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])


def fit(model, train_loader, val_loader, tcfg, device, logger=None, ckpt_path="checkpoints/best.pt"):
    """Train with early stopping on validation accuracy. Returns best val accuracy.

    `logger` is a ClearML Logger (or None); metrics are reported every epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, tcfg)
    best_acc, bad_epochs = 0.0, 0
    for epoch in range(tcfg["epochs"]):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        print(f"epoch {epoch}: train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")
        if logger:
            for name, val in [("loss/train", tr_loss), ("loss/val", va_loss),
                              ("acc/train", tr_acc), ("acc/val", va_acc)]:
                logger.report_scalar(name.split("/")[0], name.split("/")[1], val, iteration=epoch)
        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["early_stopping_patience"]:
                print("Early stopping.")
                break
    return best_acc
