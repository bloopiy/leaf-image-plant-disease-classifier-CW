import copy
from pathlib import Path

import pytest
import torch
from PIL import Image

from inference import build_preprocess, format_label, predict_topk
from model_loader import load_class_names, load_model
from plantdisease.models import SimpleCNN

CLASS_NAMES_PATH = Path(__file__).resolve().parents[1] / "app" / "class_names.json"
NAMES = [f"class_{i}" for i in range(39)]


@pytest.fixture
def model(project_cfg):
    return load_model(project_cfg)[0]


@pytest.fixture
def preprocess(project_cfg):
    return build_preprocess(project_cfg["model"]["input_size"])


def test_load_model_builds_simple_cnn_in_eval_mode(model):
    assert isinstance(model, SimpleCNN)
    assert not model.training


@pytest.mark.parametrize("key", ["model_state_dict", None])
def test_load_model_restores_weights(project_cfg, tmp_path, key):
    trained, _ = load_model(project_cfg)
    state = trained.state_dict()
    path = tmp_path / "best.pt"
    torch.save({key: state, "epoch": 3} if key else state, path)

    restored, _ = load_model(project_cfg, path)
    for name, tensor in restored.state_dict().items():
        assert torch.equal(tensor, state[name])


def test_load_model_uses_config_saved_in_checkpoint(project_cfg, tmp_path):
    trained_cfg = copy.deepcopy(project_cfg)
    trained_cfg["model"]["channels"] = [8, 16]
    trained, _ = load_model(trained_cfg)
    path = tmp_path / "best.pt"
    torch.save({"model_state_dict": trained.state_dict(), "config": trained_cfg}, path)

    restored, cfg = load_model(project_cfg, path)
    assert cfg["model"]["channels"] == [8, 16]
    assert restored.state_dict().keys() == trained.state_dict().keys()


@pytest.mark.parametrize("mode,size", [("RGB", (256, 256)), ("RGBA", (300, 200)), ("L", (64, 64))])
def test_predict_returns_five_probabilities(model, preprocess, mode, size):
    out = predict_topk(Image.new(mode, size), model, preprocess, NAMES, k=5)
    assert len(out) == 5
    assert sum(out.values()) <= 1.0 + 1e-6
    assert all(name in NAMES for name in out)


def test_predictions_sorted_by_probability(model, preprocess):
    out = predict_topk(Image.new("RGB", (128, 128)), model, preprocess, NAMES, k=5)
    probs = list(out.values())
    assert probs == sorted(probs, reverse=True)


def test_class_names_match_model_outputs(project_cfg):
    names = load_class_names(CLASS_NAMES_PATH)
    assert len(names) == project_cfg["model"]["num_classes"]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("raw,shown", [
    ("Peach___healthy", "Peach · Healthy"),
    ("Tomato___Septoria_leaf_spot", "Tomato · Septoria leaf spot"),
    ("Pepper,_bell___Bacterial_spot", "Pepper, bell · Bacterial spot"),
    ("Background_without_leaves", "Background without leaves"),
])
def test_format_label(raw, shown):
    assert format_label(raw) == shown


def test_formatted_labels_stay_unique():
    names = load_class_names(CLASS_NAMES_PATH)
    assert len({format_label(name) for name in names}) == len(names)
