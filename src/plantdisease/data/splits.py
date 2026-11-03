from datasets import Dataset, DatasetDict

from .dataset import load_plant_dataset

def create_permanent_holdout(dataset: Dataset, seed: int) -> tuple[Dataset, Dataset]:
    """Return the 85% development pool and stratified 15% permanent test holdout."""

    outer_split = dataset.train_test_split(
        test_size=0.15,
        seed=seed,
        stratify_by_column="label",
    )

    return outer_split["train"], outer_split["test"]


def select_prototype_subset(    dataset: Dataset,fraction: float,seed: int) -> Dataset:
    """Return a seeded, stratified fraction of the development pool."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1.")

    if fraction == 1:
        return dataset

    subset_split = dataset.train_test_split(
        test_size=1 - fraction,
        seed=seed,
        stratify_by_column="label",
    )
    return subset_split["train"]


def create_development_splits(dataset: Dataset, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    """Return stratified 70/15/15 train, validation, and prototype-test splits."""

    train_and_holdout = dataset.train_test_split(
        test_size=0.30,
        seed=seed,
        stratify_by_column="label",
    )
    validation_and_test = train_and_holdout["test"].train_test_split(
        test_size=0.5,
        seed=seed + 1,
        stratify_by_column="label",
    )
    return (
        train_and_holdout["train"],
        validation_and_test["train"],
        validation_and_test["test"],
    )


def load_and_split(cfg, seed):
    """Load data and return prototype splits plus the untouched permanent holdout."""

    dataset = load_plant_dataset(cfg)

    development_pool, permanent_test = create_permanent_holdout(dataset, seed)
    data_cfg = cfg["data"]
    fraction = 1.0 if data_cfg.get("use_full_development_pool", False) else data_cfg.get(
        "prototype_fraction", 0.25
    )
    prototype_subset = select_prototype_subset(development_pool, fraction, seed + 1)
    train, validation, prototype_test = create_development_splits(
        prototype_subset,
        seed + 2,
    )

    return {
        "train": train,
        "validation": validation,
        "prototype_test": prototype_test,
        "permanent_test": permanent_test,
    }