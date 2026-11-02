from pathlib import Path

import pytest
import torch

from plantdisease.models import SimpleCNN, build_model, count_parameters
from plantdisease.utils import load_config

DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.mark.parametrize("pooling", ["flatten", "global_avg"])
@pytest.mark.parametrize("channels", [[16, 32], [32, 64, 128], [16, 32, 64, 128]])
def test_output_shape_is_batch_by_39(pooling, channels):
    model = SimpleCNN(num_classes=39, input_size=128, channels=channels, pooling=pooling).eval()
    out = model(torch.randn(5, 3, 128, 128))
    assert out.shape == (5, 39)


def test_build_model_from_default_config():
    cfg = load_config(DEFAULT_CFG)
    size = cfg["model"]["input_size"]
    model = build_model(cfg).eval()
    assert model(torch.randn(2, 3, size, size)).shape == (2, 39)


def test_build_model_from_custom_config():
    cfg = {"model": {
        "num_classes": 39, "in_channels": 3, "input_size": 64, "channels": [8, 16],
        "dense_units": [], "pooling": "global_avg", "batch_norm": False,
        "conv_dropout": 0.1, "dense_dropout": 0.0,
    }}
    model = build_model(cfg).eval()
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 39)


def test_global_avg_is_smaller_than_flatten():
    flat = SimpleCNN(pooling="flatten")
    gap = SimpleCNN(pooling="global_avg")
    assert count_parameters(gap) < count_parameters(flat)


def test_rejects_input_too_small_for_depth():
    with pytest.raises(ValueError):
        SimpleCNN(input_size=8, channels=[8, 8, 8, 8])


def test_rejects_unknown_pooling():
    with pytest.raises(ValueError):
        SimpleCNN(pooling="max")