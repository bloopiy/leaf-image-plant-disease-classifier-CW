"""Tests for the shared image transforms (offline, synthetic images)."""
import pytest
import torch
from PIL import Image

from plantdisease.data.transforms import build_eval_transform, build_train_transform

MODEL_CFG = {"input_size": 32}
NORMALISED_CFG = {"input_size": 32, "normalize_mean": [0.5, 0.5, 0.5], "normalize_std": [0.5, 0.5, 0.5]}
AUGMENTATION = {"enabled": True, "rotation_degrees": 15, "brightness": 0.2,
                "saturation": 0.2, "blur_probability": 0.5}


def noise_image(size=(64, 48)):
    torch.manual_seed(1)
    data = torch.randint(0, 256, (size[1], size[0], 3), dtype=torch.uint8).numpy()
    return Image.fromarray(data)


def test_eval_transform_resizes_and_scales_to_unit_range():
    out = build_eval_transform(MODEL_CFG)(noise_image())
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_eval_transform_is_deterministic():
    tf = build_eval_transform(MODEL_CFG)
    assert torch.equal(tf(noise_image()), tf(noise_image()))


def test_normalisation_is_applied_when_configured():
    tf = build_eval_transform(NORMALISED_CFG)
    white = tf(Image.new("RGB", (40, 40), (255, 255, 255)))
    black = tf(Image.new("RGB", (40, 40), (0, 0, 0)))
    assert torch.allclose(white, torch.ones_like(white))        # (1 - 0.5) / 0.5
    assert torch.allclose(black, -torch.ones_like(black))       # (0 - 0.5) / 0.5


def test_mean_without_std_is_rejected():
    with pytest.raises(ValueError, match="together"):
        build_eval_transform({"input_size": 32, "normalize_mean": [0.5, 0.5, 0.5]})


def test_train_transform_without_augmentation_equals_eval():
    tf_train = build_train_transform(MODEL_CFG, {"enabled": False})
    tf_eval = build_eval_transform(MODEL_CFG)
    assert torch.equal(tf_train(noise_image()), tf_eval(noise_image()))
    assert torch.equal(build_train_transform(MODEL_CFG)(noise_image()), tf_eval(noise_image()))


def test_augmentation_changes_the_image_but_keeps_shape_and_range():
    tf = build_train_transform(MODEL_CFG, AUGMENTATION)
    image = noise_image()
    outputs = [tf(image) for _ in range(8)]
    assert all(o.shape == (3, 32, 32) for o in outputs)
    assert all(0.0 <= o.min() and o.max() <= 1.0 for o in outputs)
    assert any(not torch.equal(outputs[0], o) for o in outputs[1:])


def test_augmentation_keeps_normalisation():
    tf = build_train_transform(NORMALISED_CFG, AUGMENTATION)
    out = tf(noise_image())
    assert out.min() < 0  # values were mapped to [-1, 1], not left in [0, 1]
