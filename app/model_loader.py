"""Build the classifier and load its weights for inference."""
import json

from plantdisease.checkpoint import load_trained_model


def load_class_names(path):
    """Read the ordered list of class labels."""
    with open(path) as f:
        return json.load(f)


def load_model(cfg, weights=None):
    """Build the model and return (model in eval mode, config it was built from).

    `weights` is a checkpoint saved by the trainer or a plain state dict. The
    config stored in the checkpoint wins over `cfg`, so the architecture
    always matches the weights. Without weights the model is untrained, so
    predictions are random. Pretrained weights are never downloaded.
    """
    return load_trained_model(cfg, weights)
