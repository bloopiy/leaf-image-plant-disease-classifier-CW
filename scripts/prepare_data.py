"""ClearML task 1: check the data splits and register them as a ClearML Dataset.

Logs the size and label distribution of every split, so the report can show that
each split keeps the original class balance, and stores a split summary as a
versioned ClearML Dataset. The splits themselves are reproducible from the
config and the seed (the split tests check that).

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/transfer.yaml
    python scripts/prepare_data.py --offline        # no ClearML account, no dataset registration
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
from clearml import Dataset, Task

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plantdisease.data import load_and_split, load_plant_dataset
from plantdisease.utils import load_config, set_seed


def class_counts(labels, num_classes):
    """Number of images per class, as a list indexed by class id."""
    counts = [0] * num_classes
    for label in labels:
        counts[int(label)] += 1
    return counts


def shares(counts):
    """Each class's fraction of the total."""
    total = sum(counts)
    return [c / total for c in counts]


def max_share_deviation(split_counts, full_counts):
    """Largest difference between a class's share in a split and in the full dataset."""
    return max(abs(a - b) for a, b in zip(shares(split_counts), shares(full_counts)))


def build_summary(cfg, class_names, full_counts, split_counts):
    """Everything needed to understand (and re-create) the splits, as plain data."""
    return {
        "dataset": cfg["data"]["dataset"]["id"],
        "seed": cfg["seed"],
        "data_config": cfg["data"],
        "class_names": class_names,
        "sizes": {name: sum(counts) for name, counts in split_counts.items()},
        "class_counts": {"full_dataset": full_counts, **split_counts},
        "max_share_deviation": {
            name: max_share_deviation(counts, full_counts)
            for name, counts in split_counts.items()
        },
    }


def plot_distribution(class_names, full_counts, split_counts):
    """Top: images per class. Bottom: each split's class shares, which should overlap."""
    positions = range(len(class_names))
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    top.bar(positions, full_counts, color="#4c78a8")
    top.set_ylabel("images per class (full dataset)")
    for name, counts in split_counts.items():
        bottom.plot(positions, shares(counts), marker="o", markersize=3, label=name)
    bottom.set_ylabel("share of the split")
    bottom.legend()
    bottom.set_xticks(list(positions))
    bottom.set_xticklabels(class_names, rotation=90, fontsize=6)
    figure.tight_layout()
    return figure


def register_dataset(project_name, folder):
    """Upload the summary folder as a finalized ClearML Dataset and return its id."""
    dataset = Dataset.create(
        dataset_name="plant-village-splits",
        dataset_project=project_name,
        dataset_tags=["splits"],
    )
    dataset.add_files(str(folder))
    dataset.upload()
    dataset.finalize()
    return dataset.id


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--offline", action="store_true",
                   help="run ClearML offline; the dataset is not registered")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("seed", 42)
    project_name = cfg.get("clearml", {}).get("project_name", "PlantDisease-CW")

    if args.offline:
        Task.set_offline(True)
    task = Task.init(
        project_name=project_name,
        task_name="data-preparation",
        task_type=Task.TaskTypes.data_processing,
        tags=["data", "splits"],
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
    )
    try:
        task.connect(cfg["data"], name="data")
        set_seed(cfg["seed"])

        full = load_plant_dataset(cfg)
        class_names = full.features["label"].names
        full_counts = class_counts(full["label"], len(class_names))
        splits = load_and_split(cfg, cfg["seed"])
        split_counts = {
            name: class_counts(split["label"], len(class_names))
            for name, split in splits.items()
        }
        missing = [(n, class_names[i]) for n, c in split_counts.items()
                   for i, count in enumerate(c) if count == 0]
        if missing:
            raise ValueError(f"classes missing from splits: {missing[:5]}")

        summary = build_summary(cfg, class_names, full_counts, split_counts)
        output_dir = ROOT / "outputs" / "prepare_data" / task.id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "split_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        figure = plot_distribution(class_names, full_counts, split_counts)
        figure.savefig(output_dir / "label_distribution.png", dpi=150)

        logger = task.get_logger()
        logger.report_matplotlib_figure(
            title="label distribution", series="splits",
            figure=figure, iteration=0, report_image=True,
        )
        for name, size in summary["sizes"].items():
            logger.report_single_value(f"size/{name}", size)
            print(f"  {name:15s} {size:6d} images")
        for name, deviation in summary["max_share_deviation"].items():
            logger.report_single_value(f"max_share_deviation/{name}", deviation)
            print(f"  {name:15s} max class-share deviation {deviation:.4f}")

        if args.offline:
            print("Offline run: ClearML Dataset not registered.")
        else:
            dataset_id = register_dataset(project_name, output_dir)
            task.set_parameter("dataset_id", dataset_id)
            print(f"ClearML Dataset id: {dataset_id}")
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
