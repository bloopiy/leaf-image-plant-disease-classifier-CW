"""Tests for the datasets and DataLoaders (offline: fake rows instead of Hugging Face)."""
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from plantdisease.data.loaders import HFImageDataset, make_loaders
from plantdisease.data.transforms import build_eval_transform

CFG = {
    "model": {"input_size": 16},
    "train": {"batch_size": 4},
    "data": {"transformations": {"augment": {"enabled": True}}},
}


def fake_rows(n, label_offset=0):
    return [
        {"image": Image.new("RGB", (20, 20), (i * 10 % 256, 50, 90)), "label": (i + label_offset) % 39}
        for i in range(n)
    ]


@pytest.fixture
def splits():
    return {
        "train": fake_rows(10),
        "validation": fake_rows(6, 1),
        "prototype_test": fake_rows(6, 2),
        "permanent_test": fake_rows(6, 3),
    }


def test_dataset_returns_a_tensor_and_an_int_label():
    dataset = HFImageDataset(fake_rows(3), build_eval_transform(CFG["model"]))
    image, label = dataset[1]
    assert len(dataset) == 3
    assert image.shape == (3, 16, 16)
    assert isinstance(label, int)


def test_dataset_converts_other_image_modes_to_rgb():
    rows = [{"image": Image.new("L", (20, 20), 128), "label": 4}]
    image, _ = HFImageDataset(rows, build_eval_transform(CFG["model"]))[0]
    assert image.shape[0] == 3


def test_loaders_keep_the_permanent_test_set_out_by_default(splits):
    loaders = make_loaders(CFG, seed=0, splits=splits)
    assert set(loaders) == {"train", "validation", "prototype_test"}


def test_loaders_include_the_permanent_test_set_only_when_asked(splits):
    loaders = make_loaders(CFG, seed=0, splits=splits, include_permanent_test=True)
    assert set(loaders) == {"train", "validation", "prototype_test", "permanent_test"}


def test_only_the_training_loader_shuffles(splits):
    loaders = make_loaders(CFG, seed=0, splits=splits, include_permanent_test=True)
    assert isinstance(loaders["train"].sampler, RandomSampler)
    for name in ("validation", "prototype_test", "permanent_test"):
        assert isinstance(loaders[name].sampler, SequentialSampler)


def test_batch_size_and_shapes_come_from_the_config(splits):
    loaders = make_loaders(CFG, seed=0, splits=splits)
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (4, 3, 16, 16)
    assert labels.dtype == torch.int64


def test_evaluation_loaders_keep_dataset_order(splits):
    loaders = make_loaders(CFG, seed=0, splits=splits)
    labels = torch.cat([y for _, y in loaders["validation"]]).tolist()
    assert labels == [row["label"] for row in splits["validation"]]


def test_training_order_is_repeatable_for_the_same_seed(splits):
    first = [y.tolist() for _, y in make_loaders(CFG, seed=3, splits=splits)["train"]]
    second = [y.tolist() for _, y in make_loaders(CFG, seed=3, splits=splits)["train"]]
    assert first == second
