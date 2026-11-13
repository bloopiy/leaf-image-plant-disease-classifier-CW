"""End-to-end smoke test on the REAL PlantVillage data, for either model.

Runs the whole pipeline on a small slice, with no ClearML involved:
    load_and_split -> loaders -> build_model -> fit -> evaluate

It checks that the pieces fit together (config keys, class mapping, tensor
shapes, training loop, evaluation), not that the model is any good.

The permanent test set is deliberately NOT used here; it is reserved for the
final evaluation.

Do not go much below 10%: the stratified 70/15/15 split needs several images
of the rarest class in every split, and fails on tiny fractions.

Usage:
    python scripts/smoke_test.py                                   # CNN
    python scripts/smoke_test.py --config configs/transfer.yaml    # ResNet18
    python scripts/smoke_test.py --fraction 0.15 --epochs 3 --checkpoint-out best.pt
"""
import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plantdisease.data import load_and_split, make_loaders
from plantdisease.evaluate import evaluate
from plantdisease.models import build_model, count_parameters
from plantdisease.train import fit
from plantdisease.utils import get_device, load_config, set_seed

CNN_KEYS = (
    "num_classes", "in_channels", "input_size", "channels",
    "dense_units", "pooling", "batch_norm", "conv_dropout", "dense_dropout",
)
RESNET_KEYS = ("num_classes", "input_size", "pretrained", "freeze_backbone")
TRAIN_DEFAULTS = {"lr": 1e-3, "weight_decay": 0.0, "patience": 2}


class SmokeTestError(Exception):
    pass


def check(condition, message):
    if not condition:
        raise SmokeTestError(message)


def step(text):
    print(f"\n[{text}]")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--fraction", type=float, default=0.10,
                   help="prototype_fraction of the development pool "
                        "(default 0.10; very small values break stratified splitting)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true",
                   help="skip the ResNet weight download (transfer config only)")
    p.add_argument("--checkpoint-out", default=None,
                   help="copy the trained checkpoint here (e.g. to try it in the app)")
    return p.parse_args()


def model_name(cfg):
    return cfg["model"].get("name", "cnn")


def build_config(args, checkpoint_path):
    cfg = load_config(args.config)
    check("model" in cfg, "config has no 'model' section")
    check("data" in cfg, "config has no 'data' section")

    required = RESNET_KEYS if model_name(cfg) == "resnet18" else CNN_KEYS
    missing = [k for k in required if k not in cfg["model"]]
    if missing and "input_size" in missing and "input_size" in cfg["data"]:
        raise SmokeTestError(
            "input_size is under data: but must be under model: "
            "(default.yaml and the app both read model.input_size)"
        )
    check(not missing, f"config model section is missing keys: {missing}")
    check("dataset" in cfg["data"] and "id" in cfg["data"]["dataset"],
          "config is missing data.dataset.id, so load_and_split cannot run")

    cfg["data"]["prototype_fraction"] = args.fraction
    cfg["data"]["use_full_development_pool"] = False
    if args.no_pretrained and "pretrained" in cfg["model"]:
        cfg["model"]["pretrained"] = False

    train_cfg = cfg.setdefault("train", {})
    if "train" not in load_config(args.config):
        print("  note: config has no 'train' section yet, using smoke-test defaults")
    if "early_stopping_patience" in train_cfg and "patience" not in train_cfg:
        print("  WARNING: config has 'early_stopping_patience' but fit() reads 'patience'")
    for key, value in TRAIN_DEFAULTS.items():
        train_cfg.setdefault(key, value)
    train_cfg["epochs"] = args.epochs
    train_cfg["batch_size"] = args.batch_size
    train_cfg["checkpoint_path"] = str(checkpoint_path)
    cfg.setdefault("seed", 42)
    return cfg


def run(args, checkpoint_path):
    cfg = build_config(args, checkpoint_path)
    set_seed(cfg["seed"])
    device = get_device()
    m = cfg["model"]
    print(f"Model: {model_name(cfg)} | device: {device} | "
          f"prototype_fraction={args.fraction} | epochs={args.epochs}")

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

    step("3/6 loaders (shared transforms and make_loaders)")
    normalised = "normalize_mean" in m
    size = m["input_size"]
    loaders = make_loaders(cfg, cfg["seed"], num_workers=args.num_workers, splits=data)
    train_loader, val_loader = loaders["train"], loaders["validation"]
    test_loader = loaders["prototype_test"]
    images, labels = next(iter(train_loader))
    in_channels = m.get("in_channels", 3)
    check(
        tuple(images.shape[1:]) == (in_channels, size, size),
        f"batch shape {tuple(images.shape)} != (B, {in_channels}, {size}, {size})",
    )
    check(images.dtype == torch.float32, f"images dtype is {images.dtype}")
    if not normalised:
        check(0.0 <= images.min() and images.max() <= 1.0, "pixels not in [0, 1]")
    check(labels.dtype == torch.int64, f"labels dtype is {labels.dtype}")
    print(f"  batch OK: {tuple(images.shape)} ({'normalised' if normalised else 'ToTensor only'})")

    step("4/6 model")
    try:
        model = build_model(cfg).to(device)
    except KeyError as error:
        raise SmokeTestError(
            f"build_model(cfg) failed with missing key {error} for model "
            f"'{model_name(cfg)}'. If this is the transfer config, build_model "
            "needs a resnet18 case that calls build_transfer_model_from_cfg."
        ) from error
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
    if args.checkpoint_out:
        shutil.copy(result["checkpoint_path"], args.checkpoint_out)
        print(f"  checkpoint copied to {args.checkpoint_out}")

    step("6/6 evaluate on prototype_test (permanent_test stays untouched)")
    report = evaluate(model, test_loader, device,
                      num_classes=m["num_classes"], class_names=class_names)
    cm = report["confusion_matrix"]
    check(cm.shape == (m["num_classes"],) * 2, f"confusion matrix shape {cm.shape}")
    check(int(cm.sum()) == report["num_samples"] == len(data["prototype_test"]),
          "confusion matrix does not cover every test image")
    chance = 1 / m["num_classes"]
    print(f"  accuracy {report['accuracy']:.3f} | top-5 {report['top5_accuracy']:.3f} "
          f"(chance is {chance:.3f})")
    if report["accuracy"] <= chance:
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
