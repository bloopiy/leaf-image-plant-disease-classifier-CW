"""ClearML training entry point for the real PlantVillage data.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/transfer.yaml --epochs 3
    python scripts/train.py --offline
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from clearml import Task

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plantdisease.data import load_and_split, make_loaders
from plantdisease.models import build_model, count_parameters
from plantdisease.train import fit
from plantdisease.utils import get_device, load_config, set_seed

REQUIRED_SECTIONS = ("model", "data", "train", "clearml")


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


def init_task(cfg, config_path):
    """Create the ClearML task, named and tagged from the config."""
    config_stem = Path(config_path).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # default.yaml uses task_name_prefix; transfer.yaml uses task_name.
    clearml_cfg = cfg["clearml"]
    prefix = (
        clearml_cfg.get("task_name_prefix")
        or clearml_cfg.get("task_name")
        or "train"
    )
    tags = ["train", "real-data", config_stem] + list(
        clearml_cfg.get("tags", [])
    )
    return Task.init(
        project_name=clearml_cfg.get("project_name", "PlantDisease-CW"),
        task_name=f"{prefix}-{config_stem}-{timestamp}",
        tags=tags,
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
    )


def to_plain(config):
    """Copy a config through JSON, giving plain dicts and lists.

    ClearML hands back its own dict wrapper, which yaml.safe_dump rejects.
    """
    return json.loads(json.dumps(config))


def save_outputs(task, cfg, result, class_names):
    """Write the run's files and upload them to ClearML as artifacts."""
    output_dir = ROOT / "outputs" / "train" / task.id
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    class_names_path = output_dir / "class_names.json"
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    artifacts = {
        "best_checkpoint": result["checkpoint_path"],
        "training_history": str(history_path),
        "run_config": str(config_path),
        "class_names": str(class_names_path),
    }
    for name, path in artifacts.items():
        uploaded = task.upload_artifact(
            name=name,
            artifact_object=path,
            wait_on_upload=True,
        )
        if not uploaded:
            raise RuntimeError(f"Artifact upload failed: {name}")


def run_training(task, cfg, num_workers):
    """Load the data, train the model and record the results."""
    set_seed(cfg["seed"])
    device = get_device()
    gpu_name = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else None
    )
    print(f"Device: {device} | GPU: {gpu_name}")
    print(
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

    # The permanent test set is deliberately NOT used here; it is reserved
    # for the final evaluation. make_loaders leaves it out by default.
    loaders = make_loaders(cfg, cfg["seed"], num_workers=num_workers, splits=data)

    model = build_model(cfg)
    print(f"Trainable parameters: {count_parameters(model):,}")

    result = fit(
        model,
        loaders["train"],
        loaders["validation"],
        cfg,
        device,
        logger=task.get_logger(),
    )
    save_outputs(task, cfg, result, class_names)

    print(f"Task ID: {task.id}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best val accuracy: {result['best_val_accuracy']:.3f}")


def main():
    args = parse_args()
    try:
        cfg = build_config(args)
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    if args.offline or cfg["clearml"].get("offline", False):
        Task.set_offline(True)

    task = init_task(cfg, args.config)
    try:
        # Use the connected config: values edited in ClearML (a cloned task or a
        # hyperparameter sweep) come back here and take effect.
        cfg = to_plain(task.connect(cfg, name="config"))
        run_training(task, cfg, args.num_workers)
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
