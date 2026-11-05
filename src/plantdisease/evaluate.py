"""Test-set evaluation for plant disease classifiers.

Only run evaluate() on the TEST loader, once, after the final model is chosen.
Pure torch/numpy (no scikit-learn), so there is no extra dependency.
Confusion matrix convention: rows = true class, columns = predicted class.
"""
import numpy as np
import torch


@torch.no_grad()
def collect_predictions(model, loader, device, k=5):
    """Run the model over a loader.

    Returns (topk_indices, top1_confidence, targets) as numpy arrays.
    The loader must not shuffle, so row i is sample i of the dataset.
    """
    model.eval()
    model.to(device)
    topk_all, conf_all, targets_all = [], [], []
    for inputs, targets in loader:
        probs = torch.softmax(model(inputs.to(device)), dim=1)
        top_probs, top_idx = probs.topk(min(k, probs.shape[1]), dim=1)
        topk_all.append(top_idx.cpu())
        conf_all.append(top_probs[:, 0].cpu())
        targets_all.append(targets.cpu())
    return (
        torch.cat(topk_all).numpy(),
        torch.cat(conf_all).numpy(),
        torch.cat(targets_all).numpy(),
    )


def topk_accuracy(topk, targets, k):
    """Fraction of samples whose true class is in the first k predictions."""
    return float((topk[:, :k] == targets[:, None]).any(axis=1).mean())


def confusion_matrix(targets, preds, num_classes):
    """(num_classes, num_classes) counts, always full size even if a class is absent."""
    flat = np.bincount(targets * num_classes + preds, minlength=num_classes**2)
    return flat.reshape(num_classes, num_classes)


def per_class_metrics(cm):
    """Per-class precision, recall and F1 from a confusion matrix (0 when undefined)."""
    tp = np.diag(cm).astype(float)
    pred_totals = cm.sum(axis=0).astype(float)
    true_totals = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_totals, out=np.zeros_like(tp), where=pred_totals > 0)
    recall = np.divide(tp, true_totals, out=np.zeros_like(tp), where=true_totals > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def most_confused_pairs(cm, class_names=None, top_n=10):
    """Largest off-diagonal cells as (true_class, predicted_class, count), biggest first."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    name = (lambda i: class_names[i]) if class_names is not None else (lambda i: i)
    pairs = []
    for idx in np.argsort(off, axis=None)[::-1][:top_n]:
        true_i, pred_i = divmod(int(idx), off.shape[1])
        count = int(off[true_i, pred_i])
        if count == 0:
            break
        pairs.append((name(true_i), name(pred_i), count))
    return pairs


def misclassified_indices(preds, targets):
    """Dataset positions of wrong predictions, for looking at the actual images."""
    return np.flatnonzero(preds != targets)


def evaluate(model, loader, device, num_classes=39, class_names=None, k=5):
    """Full test-set evaluation. Returns a dict of metrics and error-analysis data."""
    topk, conf, targets = collect_predictions(model, loader, device, k=k)
    preds = topk[:, 0]
    cm = confusion_matrix(targets, preds, num_classes)
    precision, recall, f1 = per_class_metrics(cm)
    return {
        "num_samples": int(len(targets)),
        "accuracy": float((preds == targets).mean()),
        f"top{k}_accuracy": topk_accuracy(topk, targets, k),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "most_confused": most_confused_pairs(cm, class_names),
        "misclassified": misclassified_indices(preds, targets),
        "predictions": {"preds": preds, "targets": targets, "confidences": conf},
    }
