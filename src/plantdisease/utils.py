"""Shared helpers: config loading, seeding, device selection."""
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    """Read a YAML config file and return it as a dict."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def set_seed(seed):
    """Seed Python, NumPy and PyTorch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the GPU if one is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
