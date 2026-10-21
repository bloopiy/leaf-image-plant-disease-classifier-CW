import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plantdisease.models import SimpleCNN


def test_cnn_output_shape():
    model = SimpleCNN([8, 16], num_classes=39).eval()
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 39)
