import math

from datasets import Dataset

from .dataset import load_plant_dataset


def _resolve_split_config(cfg):
    """Return the configured train/validation/test split ratios with safe defaults."""

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    split_cfg = data_cfg.get("splits", {})
    train_frac = float(split_cfg.get("train", 0.70))
    val_frac = float(split_cfg.get("val", 0.15))
    test_frac = float(split_cfg.get("test", 0.15))
    total = train_frac + val_frac + test_frac
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "data.splits must sum to 1.0; "
            f"got train={train_frac}, val={val_frac}, test={test_frac}"
        )

    return {"train": train_frac, "val": val_frac, "test": test_frac}


def create_permanent_holdout(dataset: Dataset, seed: int, test_fraction: float = 0.15) -> tuple[Dataset, Dataset]:
    """Return the development pool and a stratified permanent test holdout."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in the range (0, 1).")

    outer_split = dataset.train_test_split(
        test_size=test_fraction,
        seed=seed,
        stratify_by_column="label",
    )

    return outer_split["train"], outer_split["test"]


def select_prototype_subset(dataset: Dataset, fraction: float, seed: int) -> Dataset:
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


def create_development_splits(dataset: Dataset, seed: int, split_cfg: dict | None = None) -> tuple[Dataset, Dataset, Dataset]:
    """Return train, validation, and test splits using configured ratios."""

    if split_cfg is None:
        split_cfg = {"train": 0.70, "val": 0.15, "test": 0.15}

    train_frac = float(split_cfg["train"])
    val_frac = float(split_cfg["val"])
    test_frac = float(split_cfg["test"])
    total = train_frac + val_frac + test_frac
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "split_cfg must sum to 1.0; "
            f"got train={train_frac}, val={val_frac}, test={test_frac}"
        )

    train_and_holdout = dataset.train_test_split(
        test_size=1.0 - train_frac,
        seed=seed,
        stratify_by_column="label",
    )
    remaining = train_and_holdout["test"]
    remaining_test_size = test_frac / (val_frac + test_frac)
    validation_and_test = remaining.train_test_split(
        test_size=remaining_test_size,
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
    split_cfg = _resolve_split_config(cfg)

    development_pool, permanent_test = create_permanent_holdout(dataset, seed)
    data_cfg = cfg["data"]
    fraction = 1.0 if data_cfg.get("use_full_development_pool", False) else data_cfg.get(
        "prototype_fraction", 0.25
    )
    prototype_subset = select_prototype_subset(development_pool, fraction, seed + 1)
    train, validation, prototype_test = create_development_splits(
        prototype_subset,
        seed + 2,
        split_cfg,
    )

    return {
        "train": train,
        "validation": validation,
        "prototype_test": prototype_test,
        "permanent_test": permanent_test,
    }