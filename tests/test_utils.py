import random

import numpy as np
import pytest
import torch

from plantdisease.utils import get_device, load_config, set_seed


def test_load_config_reads_yaml(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("seed: 7\ntrain:\n  lr: 0.001\n")
    cfg = load_config(cfg_file)
    assert cfg["seed"] == 7
    assert cfg["train"]["lr"] == 0.001


def test_load_config_empty_file_returns_empty_dict(tmp_path):
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text("")
    assert load_config(cfg_file) == {}


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_set_seed_makes_random_numbers_repeatable():
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second


def test_get_device_returns_cpu_or_cuda():
    assert get_device().type in ("cpu", "cuda")
