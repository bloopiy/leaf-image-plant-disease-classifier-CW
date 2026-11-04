"""Build the classifier and load its weights for inference."""
import json

import torch

from plantdisease.models import build_model


def load_class_names(path):
    """Read the ordered list of class labels."""
    with open(path) as f:
        return json.load(f)


def load_model(cfg, weights=None):
    """Build the model from the project config and return it in eval mode.

    `weights` is a checkpoint saved by the trainer (a dict with
    "model_state_dict") or a plain state dict. Without it the model is
    untrained, so predictions are random.
    """
    model = build_model(cfg)
    if weights:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model
