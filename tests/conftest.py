import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Make `plantdisease` importable without installing the package.
sys.path.insert(0, str(ROOT / "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_data: uses the real PlantVillage dataset from Hugging Face "
        "(slow on the first run while it downloads)",
    )


@pytest.fixture(scope="session")
def project_cfg():
    from plantdisease.utils import load_config

    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def raw_dataset(project_cfg):
    """Full dataset with a unique `_row_id` column, so tests can check that
    no image ends up in two splits."""
    from plantdisease.data.dataset import load_plant_dataset

    dataset = load_plant_dataset(project_cfg)
    return dataset.add_column("_row_id", list(range(len(dataset))))
