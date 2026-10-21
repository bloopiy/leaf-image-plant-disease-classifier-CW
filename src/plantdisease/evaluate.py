"""Test-set evaluation: accuracy, top-k, per-class metrics, confusion matrix."""
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


def evaluate(model, loader, device, class_names=None):
    probs, labels = predict_all(model, loader, device)
    preds = probs.argmax(1)
    return {
        "accuracy": float((preds == labels).mean()),
        "top5_accuracy": topk_accuracy(probs, labels, 5),
        "confusion_matrix": confusion_matrix(labels, preds),
        "report": classification_report(labels, preds, target_names=class_names, zero_division=0),
    }
