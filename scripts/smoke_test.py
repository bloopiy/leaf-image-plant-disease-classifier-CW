"""End-to-end smoke test on the REAL PlantVillage data.

Runs the whole pipeline on a small slice (10% of the development pool by
default), with no ClearML involved:
    load_and_split -> loaders -> build_model -> fit -> evaluate

It checks that the pieces fit together (config keys, class mapping, tensor
shapes, training loop), not that the model is any good. Accuracy after a
couple of epochs on a small slice is only a sanity number.

Do not go much below 10%: the stratified 70/15/15 split needs several images
of the rarest class in every split, and fails on tiny fractions.

The permanent test set is deliberately NOT used here; it is reserved for the
final evaluation.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --fraction 0.15 --epochs 3 --config configs/default.yaml
"""
import argparse
import math
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plantdisease.data import load_and_split
from plantdisease.models import build_model, count_parameters
from plantdisease.train import fit
from plantdisease.utils import get_device, load_config, set_seed

REQUIRED_MODEL_KEYS = (
    "num_classes", "in_channels", "input_size", "channels",
    "dense_units", "pooling", "batch_norm", "conv_dropout", "dense_dropout",
)
TRAIN_DEFAULTS = {"lr": 1e-3, "weight_decay": 0.0, "patience": 2}


class SmokeTestError(Exception):
    pass


def check(condition, message):
    if not condition:
        raise SmokeTestError(message)


def step(text):
    print(f"\n[{text}]")


class HFImageDataset(Dataset):
    """Wraps a Hugging Face split for PyTorch.

    TODO: replace with the Data lead's transforms / make_loaders once
    `data/transforms.py` is implemented. This is a minimal stand-in.
    """

    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.transform(item["image"].convert("RGB")), int(item["label"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--fraction", type=float, default=0.10,
                   help="prototype_fraction of the development pool "
                        "(default 0.10; very small values break stratified splitting)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def build_config(args, checkpoint_path):
    cfg = load_config(args.config)
    check("model" in cfg, "config has no 'model' section")
    check("data" in cfg, "config has no 'data' section")
    missing = [k for k in REQUIRED_MODEL_KEYS if k not in cfg["model"]]
    check(not missing, f"config model section is missing keys: {missing}")

    cfg["data"]["prototype_fraction"] = args.fraction
    cfg["data"]["use_full_development_pool"] = False

    if "train" not in cfg:
        print("  note: config has no 'train' section yet, using smoke-test defaults")
    train_cfg = cfg.setdefault("train", {})
    for key, value in TRAIN_DEFAULTS.items():
        train_cfg.setdefault(key, value)
    train_cfg["epochs"] = args.epochs
    train_cfg["checkpoint_path"] = str(checkpoint_path)
    cfg.setdefault("seed", 42)
    return cfg


def make_loader(hf_split, transform, args, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        HFImageDataset(hf_split, transform),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def run(args, checkpoint_path):
    cfg = build_config(args, checkpoint_path)
    set_seed(cfg["seed"])
    device = get_device()
    m = cfg["model"]
    print(f"Device: {device} | prototype_fraction={args.fraction} | epochs={args.epochs}")

    step("1/6 load and split the real dataset")
    try:
        data = load_and_split(cfg, cfg["seed"])
    except ValueError as error:
        raise SmokeTestError(
            f"load_and_split failed: {error}\n"
            f"  The stratified split needs several images of the rarest class in every "
            f"split. --fraction {args.fraction} is probably too small; try 0.10 or more."
        ) from error
    expected = {"train", "validation", "prototype_test", "permanent_test"}
    check(set(data) == expected, f"load_and_split returned {set(data)}, expected {expected}")
    for name, split in data.items():
        print(f"  {name:15s} {len(split):6d} images")
        check(len(split) > 0, f"split '{name}' is empty")

    step("2/6 class mapping")
    class_names = data["train"].features["label"].names
    check(
        len(class_names) == m["num_classes"],
        f"dataset has {len(class_names)} classes but config model.num_classes={m['num_classes']}",
    )
    top_label = max(data["train"]["label"])
    check(top_label < m["num_classes"], f"label {top_label} out of range")
    print(f"  {len(class_names)} classes, e.g. {class_names[:3]}")

    step("3/6 loaders (temporary transform: resize + ToTensor)")
    size = m["input_size"]
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    train_loader = make_loader(data["train"], tf, args, True, cfg["seed"])
    val_loader = make_loader(data["validation"], tf, args, False, cfg["seed"])
    test_loader = make_loader(data["prototype_test"], tf, args, False, cfg["seed"])
    images, labels = next(iter(train_loader))
    check(
        tuple(images.shape[1:]) == (m["in_channels"], size, size),
        f"batch shape {tuple(images.shape)} != (B, {m['in_channels']}, {size}, {size})",
    )
    check(images.dtype == torch.float32, f"images dtype is {images.dtype}")
    check(0.0 <= images.min() and images.max() <= 1.0, "pixels not in [0, 1]")
    check(labels.dtype == torch.int64, f"labels dtype is {labels.dtype}")
    print(f"  batch OK: {tuple(images.shape)}")

    step("4/6 model")
    model = build_model(cfg).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
    check(
        tuple(logits.shape) == (images.size(0), m["num_classes"]),
        f"logits shape {tuple(logits.shape)}",
    )
    check(bool(torch.isfinite(logits).all()), "model produced non-finite logits")
    print(f"  logits OK: {tuple(logits.shape)} | trainable parameters: {count_parameters(model):,}")

    step("5/6 training loop")
    result = fit(model, train_loader, val_loader, cfg, device)
    history = result["history"]
    check(len(history) >= 1, "fit returned an empty history")
    for row in history:
        check(
            all(math.isfinite(row[k]) for k in ("train_loss", "val_loss")),
            f"non-finite loss at epoch {row['epoch']}",
        )
    check(Path(result["checkpoint_path"]).is_file(), "no checkpoint was saved")
    first, last = history[0]["train_loss"], history[-1]["train_loss"]
    if last >= first:
        print(f"  WARNING: train loss did not fall ({first:.3f} -> {last:.3f}); "
              "try more epochs or data before worrying")
    else:
        print(f"  train loss fell: {first:.3f} -> {last:.3f}")

    step("6/6 evaluate on prototype_test (permanent_test stays untouched)")
    acc = accuracy(model, test_loader, device)
    chance = 1 / m["num_classes"]
    print(f"  accuracy {acc:.3f} (chance is {chance:.3f})")
    if acc <= chance:
        print("  WARNING: accuracy is not above chance; expected after so little training")


def main():
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run(args, Path(tmp) / "best.pt")
        except SmokeTestError as error:
            print(f"\nSMOKE TEST FAILED: {error}")
            sys.exit(1)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
