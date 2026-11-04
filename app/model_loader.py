"""Build the classifier and load its weights for inference."""
import json

import torch

from plantdisease.models import build_model


def load_class_names(path):
    """Read the ordered list of class labels."""
    with open(path) as f:
        return json.load(f)


def load_model(cfg, weights=None):
    """Build the model and return (model in eval mode, config it was built from).

    `weights` is a checkpoint saved by the trainer (a dict with
    "model_state_dict" and the "config" it trained with) or a plain state
    dict. A saved config wins over `cfg` so the architecture always matches
    the weights. Without weights the model is untrained, so predictions are
    random.
    """
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True) if weights else {}
    cfg = checkpoint.get("config", cfg)
    model = build_model(cfg)
    if checkpoint:
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model, cfg
