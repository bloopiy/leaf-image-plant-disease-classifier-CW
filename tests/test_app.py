from pathlib import Path

import pytest
from PIL import Image

from inference import build_preprocess, predict_topk
from model_loader import DummyModel, load_class_names

CLASS_NAMES_PATH = Path(__file__).resolve().parents[1] / "app" / "class_names.json"
NAMES = [f"class_{i}" for i in range(39)]


@pytest.fixture
def model():
    return DummyModel(len(NAMES)).eval()


@pytest.mark.parametrize("mode,size", [("RGB", (256, 256)), ("RGBA", (300, 200)), ("L", (64, 64))])
def test_predict_returns_five_probabilities(model, mode, size):
    out = predict_topk(Image.new(mode, size), model, build_preprocess(128), NAMES, k=5)
    assert len(out) == 5
    assert sum(out.values()) <= 1.0 + 1e-6
    assert all(name in NAMES for name in out)


def test_predictions_sorted_by_probability(model):
    out = predict_topk(Image.new("RGB", (128, 128)), model, build_preprocess(128), NAMES, k=5)
    probs = list(out.values())
    assert probs == sorted(probs, reverse=True)


def test_class_names_file_has_39_unique_labels():
    names = load_class_names(CLASS_NAMES_PATH)
    assert len(names) == 39
    assert len(set(names)) == 39
