"""Tests for the pure helpers in scripts/prepare_data.py and scripts/evaluate.py.

The scripts import clearml, so these tests are skipped if it is not installed.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("clearml")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_data = load_script("prepare_data")
evaluate_script = load_script("evaluate")


def test_class_counts_counts_each_label():
    assert prepare_data.class_counts([0, 2, 2, 1, 2], 4) == [1, 1, 3, 0]


def test_shares_sum_to_one():
    assert sum(prepare_data.shares([1, 1, 2])) == pytest.approx(1.0)


def test_identical_distribution_has_zero_deviation():
    assert prepare_data.max_share_deviation([5, 10], [50, 100]) == pytest.approx(0.0)


def test_skewed_split_has_positive_deviation():
    assert prepare_data.max_share_deviation([9, 1], [5, 5]) == pytest.approx(0.4)


def test_build_summary_reports_sizes_and_deviation():
    cfg = {"seed": 1, "data": {"dataset": {"id": "x/y"}}}
    summary = prepare_data.build_summary(cfg, ["a", "b"], [50, 50], {"train": [5, 5]})
    assert summary["sizes"] == {"train": 10}
    assert summary["max_share_deviation"]["train"] == pytest.approx(0.0)


def test_permanent_test_needs_final_flag():
    with pytest.raises(ValueError, match="--final"):
        evaluate_script.check_split_allowed("permanent_test", final=False)
    evaluate_script.check_split_allowed("permanent_test", final=True)
    evaluate_script.check_split_allowed("prototype_test", final=False)


def test_summarise_is_json_friendly():
    import json
    cm = np.array([[3, 1], [0, 4]])
    report = {
        "num_samples": 8, "accuracy": 0.875, "top5_accuracy": 1.0,
        "precision": np.array([1.0, 0.8]), "recall": np.array([0.75, 1.0]),
        "f1": np.array([0.86, 0.89]), "confusion_matrix": cm,
        "most_confused": [("a", "b", 1)], "misclassified": np.array([2]),
    }
    summary = evaluate_script.summarise(report, ["a", "b"])
    json.dumps(summary)
    assert summary["per_class"][0]["support"] == 4
