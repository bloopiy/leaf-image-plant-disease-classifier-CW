"""PyTorch datasets and DataLoaders built on the Hugging Face splits."""
import torch
from torch.utils.data import DataLoader, Dataset

from .splits import load_and_split
from .transforms import build_eval_transform, build_train_transform


class HFImageDataset(Dataset):
    """Wraps a Hugging Face split (or any list of {"image", "label"} rows)."""

    def __init__(self, split, transform):
        self.data = split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.transform(item["image"].convert("RGB")), int(item["label"])


def make_loaders(cfg, seed, num_workers=None, splits=None, include_permanent_test=False):
    """Return {"train", "validation", "prototype_test"} DataLoaders.

    Training uses augmentation (data.transformations.augment); every other
    loader uses the plain evaluation transform and keeps the dataset order.

    Batch size comes from train.batch_size (the one setting the CLI and ClearML
    override). num_workers and pin_memory come from data.loaders unless
    `num_workers` is passed explicitly.

    Pass `splits` (the result of load_and_split) to avoid loading twice.
    `permanent_test` is only included when `include_permanent_test` is true:
    it is reserved for the final evaluation.
    """
    splits = splits if splits is not None else load_and_split(cfg, seed)
    model_cfg = cfg["model"]
    batch_size = cfg["train"]["batch_size"]
    data_cfg = cfg.get("data", {})
    loader_cfg = data_cfg.get("loaders", {})
    if num_workers is None:
        num_workers = loader_cfg.get("num_workers", 0)
    pin_memory = bool(loader_cfg.get("pin_memory", False)) and torch.cuda.is_available()
    augment_cfg = data_cfg.get("transformations", {}).get("augment")
    train_tf = build_train_transform(model_cfg, augment_cfg)
    eval_tf = build_eval_transform(model_cfg)

    def loader(name, transform, shuffle):
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(
            HFImageDataset(splits[name], transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )

    loaders = {
        "train": loader("train", train_tf, shuffle=True),
        "validation": loader("validation", eval_tf, shuffle=False),
        "prototype_test": loader("prototype_test", eval_tf, shuffle=False),
    }
    if include_permanent_test:
        loaders["permanent_test"] = loader("permanent_test", eval_tf, shuffle=False)
    return loaders
