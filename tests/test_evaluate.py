"""Hand-computed checks for plantdisease.evaluate.

Worked example: targets = [0,0,1,1,2,2], preds = [0,1,1,1,2,0]
Confusion matrix (rows true, cols predicted):
    [[1, 1, 0],
     [0, 2, 0],
     [1, 0, 1]]
accuracy 4/6; precision [1/2, 2/3, 1]; recall [1/2, 1, 1/2]
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from plantdisease.evaluate import (
    confusion_matrix,
    evaluate,
    misclassified_indices,
    most_confused_pairs,
    per_class_metrics,
    topk_accuracy,
)

TARGETS = np.array([0, 0, 1, 1, 2, 2])
PREDS = np.array([0, 1, 1, 1, 2, 0])


def test_confusion_matrix_hand_computed():
    cm = confusion_matrix(TARGETS, PREDS, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_confusion_matrix_keeps_full_size_when_class_absent():
    cm = confusion_matrix(TARGETS, PREDS, 4)
    assert cm.shape == (4, 4)
    assert cm[3].sum() == 0


def test_per_class_precision_recall_hand_computed():
    precision, recall, _ = per_class_metrics(confusion_matrix(TARGETS, PREDS, 3))
    np.testing.assert_allclose(precision, [1 / 2, 2 / 3, 1.0])
    np.testing.assert_allclose(recall, [1 / 2, 1.0, 1 / 2])


def test_absent_class_gives_zero_not_nan():
    precision, recall, f1 = per_class_metrics(confusion_matrix(TARGETS, PREDS, 4))
    assert precision[3] == 0 and recall[3] == 0 and f1[3] == 0


def test_topk_accuracy_hand_computed():
    topk = np.array([[0, 1], [2, 0], [1, 0], [1, 2], [2, 1], [1, 0]])
    assert topk_accuracy(topk, TARGETS, 1) == pytest.approx(4 / 6)
    assert topk_accuracy(topk, TARGETS, 2) == pytest.approx(5 / 6)


def test_most_confused_pairs_ordered_and_named():
    cm = np.array([[5, 3, 0], [1, 4, 0], [0, 2, 6]])
    names = ["a", "b", "c"]
    assert most_confused_pairs(cm, names, top_n=2) == [("a", "b", 3), ("c", "b", 2)]


def test_misclassified_indices():
    assert misclassified_indices(PREDS, TARGETS).tolist() == [1, 5]


def test_evaluate_end_to_end_with_identity_model():
    # Identity model: the input rows are the logits, so predictions are known exactly
    x = torch.eye(3)[torch.tensor(PREDS)]
    y = torch.tensor(TARGETS)
    res = evaluate(nn.Identity(), [(x, y)], "cpu", num_classes=3, k=3)
    assert res["accuracy"] == pytest.approx(4 / 6)
    assert res["top3_accuracy"] == 1.0
    assert res["confusion_matrix"].shape == (3, 3)
    assert res["misclassified"].tolist() == [1, 5]
