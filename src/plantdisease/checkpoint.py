"""Loading trained models, shared by evaluation and the app.

The trainer stores the config it trained with as a JSON string under
"config_json". A string keeps the checkpoint loadable with weights_only=True,
and it means the weights always travel with the architecture they belong to.
"""
import copy
import json

import torch

from .models import build_model


def saved_config(checkpoint):
    """Return the config stored in a checkpoint dict, or None if there is none."""
    if "config_json" in checkpoint:
        return json.loads(checkpoint["config_json"])
    return checkpoint.get("config")  # older checkpoints stored a plain dict


def load_trained_model(cfg, weights=None, device="cpu"):
    """Build the model and return (model in eval mode, config it was built from).

    `weights` is a checkpoint saved by the trainer or a plain state dict. A
    config saved in the checkpoint wins over `cfg`, so the architecture always
    matches the weights. Without weights the model is untrained.

    Never downloads pretrained weights: they would be replaced straight away
    by the checkpoint (or are not wanted for an untrained model).
    """
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True) if weights else {}
    cfg = copy.deepcopy(saved_config(checkpoint) or cfg)
    if "pretrained" in cfg["model"]:
        cfg["model"]["pretrained"] = False

    model = build_model(cfg)
    if checkpoint:
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return model.to(device).eval(), cfg
