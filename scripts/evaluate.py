"""ClearML task 3: evaluate a trained model on the held-out test set."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clearml import Task

from plantdisease.data import make_loaders
from plantdisease.evaluate import evaluate
from plantdisease.models import build_model
from plantdisease.utils import get_device, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task-id", required=True, help="ClearML training task that produced the model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = Task.init(project_name=cfg["project_name"], task_name="evaluate",
                     task_type=Task.TaskTypes.testing)
    device = get_device()

    weights = Task.get_task(task_id=args.task_id).artifacts["best_model"].get_local_copy()
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))

    _, _, test_loader = make_loaders(cfg["data"] | {"batch_size": cfg["train"]["batch_size"]},
                                     cfg["seed"])
    results = evaluate(model, test_loader, device)
    print(results["report"])
    logger = task.get_logger()
    logger.report_single_value("test_accuracy", results["accuracy"])
    logger.report_single_value("test_top5_accuracy", results["top5_accuracy"])
    logger.report_confusion_matrix("confusion_matrix", "test",
                                   iteration=0, matrix=results["confusion_matrix"])


if __name__ == "__main__":
    main()
