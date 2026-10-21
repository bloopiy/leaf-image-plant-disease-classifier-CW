"""ClearML task 2: train a model with hyperparameters exposed from the config."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clearml import Task

from plantdisease.data import make_loaders
from plantdisease.models import build_model
from plantdisease.train import fit
from plantdisease.utils import ensure_dir, get_device, load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="override model.name (cnn | resnet18)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model

    task = Task.init(project_name=cfg["project_name"],
                     task_name=f"train-{cfg['model']['name']}",
                     task_type=Task.TaskTypes.training)
    task.add_tags(cfg["clearml"]["tags"] + [cfg["model"]["name"]])
    for section in ("data", "model", "train"):  # exposes hyperparameters for comparison
        task.connect(cfg[section], name=section)

    set_seed(cfg["seed"])
    device = get_device()
    train_loader, val_loader, _ = make_loaders(cfg["data"] | {"batch_size": cfg["train"]["batch_size"]},
                                               cfg["seed"])
    model = build_model(cfg).to(device)

    ensure_dir("checkpoints")
    best = fit(model, train_loader, val_loader, cfg["train"], device,
               logger=task.get_logger(), ckpt_path="checkpoints/best.pt")
    task.upload_artifact("best_model", "checkpoints/best.pt")  # final model lives on ClearML
    print(f"Best val accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
