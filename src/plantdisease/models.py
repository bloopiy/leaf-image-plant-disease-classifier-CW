"""Model definitions: a CNN from scratch (required) and a transfer-learning model."""
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """From-scratch CNN: conv blocks (resolution down, channels up) + dense head."""

    def __init__(self, channels, num_classes, dropout=0.3, batch_norm=True):
        super().__init__()
        layers, in_ch = [], 3
        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)  # keeps the head small
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_resnet18(num_classes: int, freeze_backbone: bool = True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(cfg: dict):
    m = cfg["model"]
    if m["name"] == "cnn":
        return SimpleCNN(m["channels"], m["num_classes"], m["dropout"], m["batch_norm"])
    if m["name"] == "resnet18":
        return build_resnet18(m["num_classes"])
    raise ValueError(f"Unknown model: {m['name']}")
