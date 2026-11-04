import copy
import json
import platform
import sys
from pathlib import Path

import torch
import yaml
from clearml import Task
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plantdisease.train import fit
from plantdisease.utils import get_device, set_seed


def main():
    seed = 42
    set_seed(seed)
    device = get_device()

    gpu_name = (
        torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else None
    )

    cfg = {
        "seed": seed,
        "data": {
            "kind": "synthetic_dark_bright",
            "image_size": 32,
            "class_names": ["dark", "bright"],
            "train_examples": 64,
            "val_examples": 32,
            "batch_size": 16,
        },
        "model": {
            "kind": "tiny_smoke_cnn",
            "feature_channels": 8,
        },
        "train": {
            "epochs": 3,
            "lr": 0.01,
            "patience": 3,
            "checkpoint_path": str(
                ROOT / "checkpoints" / "smoke" / "best.pt"
            ),
        },
        "runtime": {
            "device": str(device),
            "gpu": gpu_name,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }

    # Separate examples for training and validation.
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(
        96, 3, 32, 32, generator=generator
    ) * 0.2
    labels = torch.arange(96) % 2

    # An intentionally easy artificial signal:
    # class 0 is darker; class 1 is brighter.
    images += (
        (labels.float() * 2 - 1).view(-1, 1, 1, 1) * 0.5
    )

    train_loader = DataLoader(
        TensorDataset(images[:64], labels[:64]),
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(images[64:], labels[64:]),
        batch_size=16,
        shuffle=False,
    )

    # Independent of the unfinished project model module.
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 2),
    )

    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")

    task = Task.init(
        project_name="PlantDisease-CW",
        task_name=f"synthetic-smoke-{device.type}",
        tags=["smoke-test", "synthetic", "not-project-results"],
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
    )

    try:
        # Track a copy; keep the trainer configuration plain.
        task.connect(copy.deepcopy(cfg))

        result = fit(
            model,
            train_loader,
            val_loader,
            cfg,
            device,
            logger=task.get_logger(),
        )

        output_dir = ROOT / "outputs" / "smoke"
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
            "smoke_script": str(Path(__file__).resolve()),
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
        print(f"Checkpoint: {result['checkpoint_path']}")

    except Exception:
        task.mark_failed()
        raise
    finally:
        task.close()


if __name__ == "__main__":
    main()
