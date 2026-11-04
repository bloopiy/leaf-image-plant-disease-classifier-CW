"""Week 1 sanity check for SimpleCNN (no dataset, no ClearML needed).

1. Output shape is (batch, 39).
2. The model can overfit one tiny batch: loss must fall sharply.

Run from the repo root:
    python scripts/sanity_check_cnn.py
    python scripts/sanity_check_cnn.py --config configs/default.yaml --steps 100
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plantdisease.models import build_model, count_parameters  # noqa: E402
from plantdisease.utils import load_config, set_seed  # noqa: E402


def overfit_tiny_batch(cfg, batch_size, steps, lr):
    """Train on ONE fixed random batch; return (first_loss, last_loss)."""
    m = cfg["model"]
    model = build_model(cfg)

    # Random images + random labels: the only way to fit them is to memorise.
    x = torch.randn(batch_size, m["in_channels"], m["input_size"], m["input_size"])
    y = torch.randint(0, m["num_classes"], (batch_size,))

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    first = last = 0.0
    for step in range(steps):
        optimiser.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimiser.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:3d}  loss {loss.item():.4f}")
    return first, last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    m = cfg["model"]

    model = build_model(cfg)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Check 1: output shape
    out = model.eval()(torch.randn(4, m["in_channels"], m["input_size"], m["input_size"]))
    assert out.shape == (4, m["num_classes"]), f"bad output shape {tuple(out.shape)}"
    print(f"Output shape OK: {tuple(out.shape)}")

    # Check 2: overfit a tiny batch
    first, last = overfit_tiny_batch(cfg, args.batch_size, args.steps, args.lr)
    print(f"\nloss {first:.4f} -> {last:.4f}")
    if last < 0.1 * first:
        print("PASS: model overfits a tiny batch, safe to scale up.")
    else:
        print("FAIL: loss did not drop enough. Check the model/lr before scaling.")
        sys.exit(1)


if __name__ == "__main__":
    main()