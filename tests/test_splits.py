"""Integrity tests for the data splits, run on the real PlantVillage data.

The first run downloads the dataset from Hugging Face; later runs use the
local cache. Skip these tests with:  pytest -m "not real_data"

Overlap is checked through a `_row_id` column added in conftest.py. The split
functions keep every column, so each split still carries the ids.
"""
import copy
from collections import Counter

import pytest

from plantdisease.data import splits
from plantdisease.data.splits import (
    create_development_splits,
    create_permanent_holdout,
    select_prototype_subset,
)

pytestmark = pytest.mark.real_data

SEED = 42
EXPECTED_SAMPLES = 55_447  # change if the dataset version changes
EXPECTED_CLASSES = 39
FRACTION_TOL = 0.005  # allowed error in split sizes (as a share of the data)
SHARE_TOL = 0.005     # allowed error in a class's share of a split


# ---------- helpers ----------

def id_set(dataset):
    values = list(dataset["_row_id"])
    assert len(values) == len(set(values)), "an image appears twice in one split"
    return set(values)


def class_shares(dataset):
    counts = Counter(dataset["label"])
    total = sum(counts.values())
    return {label: n / total for label, n in counts.items()}


def assert_stratified(dataset, reference_shares, name):
    shares = class_shares(dataset)
    assert set(shares) == set(reference_shares), f"{name}: a class is missing"
    worst = max(abs(shares[c] - reference_shares[c]) for c in reference_shares)
    assert worst < SHARE_TOL, f"{name}: class share off by {worst:.4f}"


def assert_disjoint(named_splits):
    names = list(named_splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = id_set(named_splits[a]) & id_set(named_splits[b])
            assert not overlap, f"{len(overlap)} images in both {a} and {b}"


def run_load_and_split(raw_dataset, cfg, seed, **data_overrides):
    """Run the real `load_and_split`, feeding it the id-tagged dataset."""
    cfg = copy.deepcopy(cfg)
    cfg["data"].update(data_overrides)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(splits, "load_plant_dataset", lambda _cfg: raw_dataset)
        return splits.load_and_split(cfg, seed)


# ---------- fixtures (built once per file) ----------

@pytest.fixture(scope="module")
def full_shares(raw_dataset):
    return class_shares(raw_dataset)


@pytest.fixture(scope="module")
def holdout(raw_dataset):
    return create_permanent_holdout(raw_dataset, SEED)  # (dev_pool, test)


@pytest.fixture(scope="module")
def prototype_result(raw_dataset, project_cfg):
    return run_load_and_split(
        raw_dataset, project_cfg, SEED,
        prototype_fraction=0.25, use_full_development_pool=False,
    )


@pytest.fixture(scope="module")
def full_result(raw_dataset, project_cfg):
    return run_load_and_split(
        raw_dataset, project_cfg, SEED, use_full_development_pool=True,
    )


# ---------- dataset ----------

def test_dataset_size_and_classes(raw_dataset):
    assert len(raw_dataset) == EXPECTED_SAMPLES
    assert len(raw_dataset.features["label"].names) == EXPECTED_CLASSES


# ---------- permanent holdout ----------

def test_holdout_is_a_complete_disjoint_partition(raw_dataset, holdout):
    dev, test = holdout
    assert_disjoint({"development": dev, "permanent_test": test})
    assert id_set(dev) | id_set(test) == id_set(raw_dataset)


def test_holdout_is_15_percent(raw_dataset, holdout):
    _, test = holdout
    assert abs(len(test) / len(raw_dataset) - 0.15) < FRACTION_TOL


def test_holdout_and_dev_pool_are_stratified(holdout, full_shares):
    dev, test = holdout
    assert_stratified(test, full_shares, "permanent_test")
    assert_stratified(dev, full_shares, "development pool")


def test_holdout_is_reproducible(raw_dataset, holdout):
    _, test = holdout
    _, test_again = create_permanent_holdout(raw_dataset, SEED)
    assert id_set(test) == id_set(test_again)


# ---------- prototype subset ----------

@pytest.mark.parametrize("bad_fraction", [0, -0.1, 1.5])
def test_prototype_fraction_rejects_invalid_values(holdout, bad_fraction):
    dev, _ = holdout
    with pytest.raises(ValueError):
        select_prototype_subset(dev, bad_fraction, SEED)


def test_prototype_fraction_one_keeps_everything(holdout):
    dev, _ = holdout
    assert len(select_prototype_subset(dev, 1, SEED)) == len(dev)


def test_prototype_subset_is_25_percent_stratified_and_inside_dev_pool(
    holdout, full_shares
):
    dev, _ = holdout
    subset = select_prototype_subset(dev, 0.25, SEED + 1)
    assert abs(len(subset) / len(dev) - 0.25) < FRACTION_TOL
    assert id_set(subset) <= id_set(dev)
    assert_stratified(subset, full_shares, "prototype subset")


def test_prototype_subset_is_reproducible(holdout):
    dev, _ = holdout
    first = select_prototype_subset(dev, 0.25, SEED + 1)
    second = select_prototype_subset(dev, 0.25, SEED + 1)
    assert id_set(first) == id_set(second)


# ---------- 70/15/15 development splits ----------

def test_development_splits_are_70_15_15_and_disjoint(holdout):
    dev, _ = holdout
    train, val, test = create_development_splits(dev, SEED)
    total = len(dev)
    assert abs(len(train) / total - 0.70) < FRACTION_TOL
    assert abs(len(val) / total - 0.15) < FRACTION_TOL
    assert abs(len(test) / total - 0.15) < FRACTION_TOL
    assert_disjoint({"train": train, "validation": val, "prototype_test": test})
    assert id_set(train) | id_set(val) | id_set(test) == id_set(dev)


def test_development_splits_are_stratified(holdout, full_shares):
    dev, _ = holdout
    train, val, test = create_development_splits(dev, SEED)
    assert_stratified(train, full_shares, "train")
    assert_stratified(val, full_shares, "validation")
    assert_stratified(test, full_shares, "prototype_test")


# ---------- load_and_split (the public API) ----------

def test_load_and_split_returns_the_four_expected_splits(prototype_result):
    assert set(prototype_result) == {
        "train", "validation", "prototype_test", "permanent_test",
    }


def test_no_image_is_shared_between_any_two_splits(prototype_result):
    assert_disjoint(prototype_result)


def test_permanent_test_is_15_percent_of_all_data(raw_dataset, prototype_result):
    share = len(prototype_result["permanent_test"]) / len(raw_dataset)
    assert abs(share - 0.15) < FRACTION_TOL


def test_prototype_mode_uses_about_25_percent_of_dev_pool(
    raw_dataset, prototype_result
):
    r = prototype_result
    used = len(r["train"]) + len(r["validation"]) + len(r["prototype_test"])
    dev_size = len(raw_dataset) - len(r["permanent_test"])
    assert abs(used / dev_size - 0.25) < FRACTION_TOL
    assert abs(len(r["train"]) / used - 0.70) < FRACTION_TOL
    assert abs(len(r["validation"]) / used - 0.15) < FRACTION_TOL
    assert abs(len(r["prototype_test"]) / used - 0.15) < FRACTION_TOL


def test_full_development_mode_uses_the_whole_dev_pool(raw_dataset, full_result):
    r = full_result
    used = len(r["train"]) + len(r["validation"]) + len(r["prototype_test"])
    assert used == len(raw_dataset) - len(r["permanent_test"])
    assert_disjoint(r)


def test_every_split_contains_every_class(prototype_result, full_shares):
    for name, split in prototype_result.items():
        assert_stratified(split, full_shares, name)


def test_permanent_test_does_not_change_with_subset_settings(
    prototype_result, full_result
):
    """The holdout must be identical in prototype and full mode."""
    assert id_set(prototype_result["permanent_test"]) == id_set(
        full_result["permanent_test"]
    )


def test_same_seed_gives_identical_splits(raw_dataset, project_cfg, prototype_result):
    again = run_load_and_split(
        raw_dataset, project_cfg, SEED,
        prototype_fraction=0.25, use_full_development_pool=False,
    )
    for name in prototype_result:
        assert id_set(prototype_result[name]) == id_set(again[name]), name


def test_different_seed_gives_different_splits(
    raw_dataset, project_cfg, prototype_result
):
    other = run_load_and_split(
        raw_dataset, project_cfg, SEED + 100,
        prototype_fraction=0.25, use_full_development_pool=False,
    )
    assert id_set(prototype_result["train"]) != id_set(other["train"])


def test_class_labels_are_consistent_across_splits(raw_dataset, prototype_result):
    """Class index -> name mapping must be the same everywhere (the UI and
    the evaluation code rely on it)."""
    names = raw_dataset.features["label"].names
    for split_name, split in prototype_result.items():
        assert split.features["label"].names == names, split_name
