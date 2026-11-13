"""ClearML task 3: evaluate a trained model on a held-out split.

The permanent test set is reserved for the final evaluation: it is only used
when --final is passed, and should be run once, for the chosen model.

Usage:
    python scripts/evaluate.py --task-id <training task id>                    # prototype_test
    python scripts/evaluate.py --task-id <id> --split validation
    python scripts/evaluate.py --task-id <id> --split permanent_test --final
    python scripts/evaluate.py --checkpoint checkpoints/simple-cnn/best.pt --offline
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from clearml import Task

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plantdisease.checkpoint import load_trained_model
from plantdisease.data import load_and_split, make_loaders
from plantdisease.evaluate import evaluate
from plantdisease.utils import get_device, load_config, set_seed

SPLITS = ("validation", "prototype_test", "permanent_test")


def check_split_allowed(split, final):
    """Refuse to touch the permanent test set unless this is the final run."""
    if split == "permanent_test" and not final:
        raise ValueError(
            "permanent_test is reserved for the final evaluation of the chosen "
            "model. Add --final if this is that run."
        )


def summarise(report, class_names):
    """The evaluation report as plain, JSON-friendly data."""
    support = report["confusion_matrix"].sum(axis=1)
    per_class = [
        {
            "class": name,
            "precision": float(report["precision"][i]),
            "recall": float(report["recall"][i]),
            "f1": float(report["f1"][i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    ]
    return {
        "num_samples": report["num_samples"],
        "accuracy": report["accuracy"],
        "top5_accuracy": report["top5_accuracy"],
        "per_class": per_class,
        "most_confused": [list(pair) for pair in report["most_confused"]],
        "misclassified_indices": report["misclassified"].tolist(),
    }


def plot_confusion_matrix(matrix, class_names):
    """Row-normalised confusion matrix: each row shows where a true class goes."""
    row_totals = matrix.sum(axis=1, keepdims=True).clip(min=1)
    figure, axis = plt.subplots(figsize=(12, 11))
    image = axis.imshow(matrix / row_totals, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=90, fontsize=6)
    axis.set_yticklabels(class_names, fontsize=6)
    axis.set_xlabel("predicted class")
    axis.set_ylabel("true class")
    figure.colorbar(image, ax=axis, fraction=0.04, label="share of the true class")
    figure.tight_layout()
    return figure


def load_run(args):
    """Return (checkpoint path, config or None) from a ClearML task or a local file."""
    if args.task_id:
        train_task = Task.get_task(task_id=args.task_id)
        checkpoint = train_task.artifacts["best_checkpoint"].get_local_copy()
        config_text = Path(train_task.artifacts["run_config"].get_local_copy()).read_text()
        return checkpoint, yaml.safe_load(config_text)
    cfg = load_config(args.config) if args.config else None
    return args.checkpoint, cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-id", help="ClearML training task that produced the model")
    source.add_argument("--checkpoint", help="local checkpoint file instead of a task")
    p.add_argument("--config", default=None,
                   help="config for --checkpoint (defaults to the one saved in the checkpoint)")
    p.add_argument("--split", choices=SPLITS, default="prototype_test")
    p.add_argument("--final", action="store_true",
                   help="allow evaluating on permanent_test (use once, for the chosen model)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--offline", action="store_true", help="run ClearML offline")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        check_split_allowed(args.split, args.final)
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    checkpoint, cfg = load_run(args)
    device = get_device()
    model, cfg = load_trained_model(cfg, checkpoint, device)
    cfg.setdefault("seed", 42)

    if args.offline:
        Task.set_offline(True)
    source = args.task_id[:8] if args.task_id else Path(checkpoint).stem
    task = Task.init(
        project_name=cfg.get("clearml", {}).get("project_name", "PlantDisease-CW"),
        task_name=f"evaluate-{args.split}-{source}",
        task_type=Task.TaskTypes.testing,
        tags=["evaluation", args.split] + (["final"] if args.final else []),
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
    )
    try:
        task.connect(
            {"trained_task_id": args.task_id, "checkpoint": str(checkpoint),
             "split": args.split, "final": args.final},
            name="evaluation",
        )
        set_seed(cfg["seed"])

        # Same data config and seed as training, so the splits are identical.
        splits = load_and_split(cfg, cfg["seed"])
        class_names = splits["train"].features["label"].names
        loaders = make_loaders(
            cfg, cfg["seed"], num_workers=args.num_workers, splits=splits,
            include_permanent_test=args.split == "permanent_test",
        )
        report = evaluate(
            model, loaders[args.split], device,
            num_classes=cfg["model"]["num_classes"], class_names=class_names,
        )
        summary = summarise(report, class_names)

        output_dir = ROOT / "outputs" / "evaluate" / task.id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        figure = plot_confusion_matrix(report["confusion_matrix"], class_names)
        figure.savefig(output_dir / "confusion_matrix.png", dpi=150)

        logger = task.get_logger()
        logger.report_single_value("accuracy", summary["accuracy"])
        logger.report_single_value("top5_accuracy", summary["top5_accuracy"])
        logger.report_table(
            "per_class_metrics", args.split, iteration=0,
            table_plot=pd.DataFrame(summary["per_class"]),
        )
        logger.report_confusion_matrix(
            "confusion_matrix", args.split, iteration=0,
            matrix=report["confusion_matrix"], xlabels=class_names, ylabels=class_names,
        )
        logger.report_matplotlib_figure(
            title="confusion matrix (row-normalised)", series=args.split,
            figure=figure, iteration=0, report_image=True,
        )
        for name, path in {"metrics": output_dir / "metrics.json",
                           "confusion_matrix_png": output_dir / "confusion_matrix.png"}.items():
            task.upload_artifact(name=name, artifact_object=str(path), wait_on_upload=True)

        print(f"{args.split}: {summary['num_samples']} images")
        print(f"accuracy {summary['accuracy']:.4f} | top-5 {summary['top5_accuracy']:.4f}")
        worst = sorted(summary["per_class"], key=lambda row: row["f1"])[:5]
        print("lowest F1:", ", ".join(f"{r['class']} ({r['f1']:.2f})" for r in worst))
        for true_name, predicted, count in summary["most_confused"][:5]:
            print(f"  {true_name} -> {predicted}: {count}")
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
