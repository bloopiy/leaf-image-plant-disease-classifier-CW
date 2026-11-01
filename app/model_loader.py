"""Build the classifier and load its weights for inference."""
import json

import torch
import torch.nn as nn


class DummyModel(nn.Module):
    """Untrained stand-in so the UI works before a real model exists."""

    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_class_names(path):
    """Read the ordered list of class labels."""
    with open(path) as f:
        return json.load(f)


def load_model(weights, num_classes):
    """Return the model in eval mode, loading a state dict if a path is given."""
    model = DummyModel(num_classes)  # TODO: swap for plantdisease.models.build_model(cfg)
    if weights:
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model
