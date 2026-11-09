"""ClearML training entry point for the real PlantVillage data.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/transfer.yaml --epochs 3
    python scripts/train.py --offline
"""
import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from clearml import Task
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plantdisease.data import load_and_split
from plantdisease.models import build_model, count_parameters
from plantdisease.train import fit
from plantdisease.utils import get_device, load_config, set_seed

REQUIRED_SECTIONS = ("model", "data", "train", "clearml")


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
    p.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="override data.prototype_fraction",
    )
    p.add_argument(
        "--full-pool",
        action="store_true",
        help="use the full development pool (data.use_full_development_pool)",
    )
    p.add_argument(
        "--epochs", type=int, default=None, help="override train.epochs"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override train.batch_size",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="override train.checkpoint_path",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="run ClearML offline, without credentials",
    )
    return p.parse_args()


def build_config(args):
    """Load the YAML config and apply CLI overrides."""
    cfg = load_config(args.config)
    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            raise ValueError(
                f"config has no '{section}' section yet: {args.config}"
            )

    cfg.setdefault("seed", 42)

    if args.fraction is not None:
        cfg["data"]["prototype_fraction"] = args.fraction
    if args.full_pool:
        cfg["data"]["use_full_development_pool"] = True
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.checkpoint is not None:
        cfg["train"]["checkpoint_path"] = args.checkpoint

    checkpoint_path = Path(cfg["train"]["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        cfg["train"]["checkpoint_path"] = str(ROOT / checkpoint_path)

    return cfg


def make_loader(hf_split, transform, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        HFImageDataset(hf_split, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
    )


def main():
    args = parse_args()
    try:
        cfg = build_config(args)
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    if args.offline or cfg["clearml"].get("offline", False):
        Task.set_offline(True)

    set_seed(cfg["seed"])
    device = get_device()
    gpu_name = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else None
    )
    print(f"Device: {device} | GPU: {gpu_name}")
    print(
        f"Config: {args.config} | "
        f"epochs={cfg['train']['epochs']} | "
        f"batch_size={cfg['train']['batch_size']}"
    )

    data = load_and_split(cfg, cfg["seed"])
    for name, split in data.items():
        print(f"  {name:15s} {len(split):6d} images")

    class_names = data["train"].features["label"].names
    if len(class_names) != cfg["model"]["num_classes"]:
        raise ValueError(
            f"dataset has {len(class_names)} classes but config "
            f"model.num_classes={cfg['model']['num_classes']}"
        )
    print(f"Classes: {len(class_names)}")

    size = cfg["model"]["input_size"]
    # TODO: replace with the Data lead's transforms once data/transforms.py
    # exists. This resize + ToTensor pair is a temporary stand-in.
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    # The permanent test set is deliberately NOT used here; it is reserved
    # for the final evaluation.
    batch_size = cfg["train"]["batch_size"]
    train_loader = make_loader(
        data["train"], transform, batch_size, True, cfg["seed"],
        args.num_workers,
    )
    val_loader = make_loader(
        data["validation"], transform, batch_size, False, cfg["seed"],
        args.num_workers,
    )

    model = build_model(cfg)
    print(f"Trainable parameters: {count_parameters(model):,}")

    config_stem = Path(args.config).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task = Task.init(
        project_name=cfg["clearml"]["project_name"],
        task_name=(
            f"{cfg['clearml']['task_name_prefix']}-{config_stem}-{timestamp}"
        ),
        tags=["train", "real-data", config_stem],
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
    )

    try:
        task.connect(copy.deepcopy(cfg))

        result = fit(
            model,
            train_loader,
            val_loader,
            cfg,
            device,
            logger=task.get_logger(),
        )

        output_dir = ROOT / "outputs" / "train"
        output_dir.mkdir(parents=True, exist_ok=True)

        history_path = output_dir / "history.json"
        history_path.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

        config_path = output_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8",
        )

        (output_dir / "task_id.txt").write_text(
            task.id, encoding="utf-8"
        )

        artifacts = {
            "best_checkpoint": result["checkpoint_path"],
            "training_history": str(history_path),
            "run_config": str(config_path),
        }

        for name, path in artifacts.items():
            uploaded = task.upload_artifact(
                name=name,
                artifact_object=path,
                wait_on_upload=True,
            )
            if not uploaded:
                raise RuntimeError(f"Artifact upload failed: {name}")

        print(f"Task ID: {task.id}")
        print(f"Best epoch: {result['best_epoch']}")
        print(f"Best val accuracy: {result['best_val_accuracy']:.3f}")

    except BaseException:
        try:
            task.mark_failed()
        except Exception:
            pass
        raise
    finally:
        task.close()


if __name__ == "__main__":
    main()
