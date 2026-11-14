"""Data loaders for the CNN model"""

from torch.utils.data import DataLoader
from .splits import load_and_split
from .transforms import build_transform_pipeline

def apply_transforms(dataset, transform):
    """Apply the specified transform to the dataset."""

    dataset.set_transform(transform)

    return dataset

def make_loaders(cfg, seed):
    """Create data loaders for the CNN model based on the configuration and seed."""

    train, validation, prototype_test, permanent_test = load_and_split(cfg, seed)
    pipelines = build_transform_pipeline(cfg)

    train_dataset = apply_transforms(train, pipelines["train"])
    validation_dataset = apply_transforms(validation, pipelines["eval"])
    prototype_test_dataset = apply_transforms(prototype_test, pipelines["eval"])
    permanent_test_dataset = apply_transforms(permanent_test, pipelines["eval"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["loaders"]["batch_size"],
        num_workers=cfg["data"]["loaders"]["num_workers"],
        pin_memory=cfg["data"]["loaders"]["pin_memory"],
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg["data"]["loaders"]["batch_size"],
        num_workers=cfg["data"]["loaders"]["num_workers"],
        pin_memory=cfg["data"]["loaders"]["pin_memory"],
        shuffle=False
    )

    prototype_test_loader = DataLoader(
        prototype_test_dataset,
        batch_size=cfg["data"]["loaders"]["batch_size"],
        num_workers=cfg["data"]["loaders"]["num_workers"],
        pin_memory=cfg["data"]["loaders"]["pin_memory"],
        shuffle=False
    )

    permanent_test_loader = DataLoader(
        permanent_test_dataset,
        batch_size=cfg["data"]["loaders"]["batch_size"],
        num_workers=cfg["data"]["loaders"]["num_workers"],
        pin_memory=cfg["data"]["loaders"]["pin_memory"],
        shuffle=False
    )

    return {
        "train": train_loader,
        "validation": validation_loader,
        "prototype_test": prototype_test_loader,
        "permanent_test": permanent_test_loader
    }

