"""ClearML task 1: load data, split, and register the processed split as a ClearML Dataset.

TODO (Data subteam): save the split indices/files to data/processed and upload
them with clearml.Dataset so training tasks can fetch a fixed, versioned split.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clearml import Task

from plantdisease.data import load_and_split
from plantdisease.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = Task.init(project_name=cfg["project_name"], task_name="data-preparation",
                     task_type=Task.TaskTypes.data_processing)
    task.connect(cfg["data"], name="data")
    set_seed(cfg["seed"])

    train, val, test = load_and_split(cfg["data"], cfg["seed"])
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    # TODO: log label-distribution plots per split and create the ClearML Dataset


if __name__ == "__main__":
    main()
