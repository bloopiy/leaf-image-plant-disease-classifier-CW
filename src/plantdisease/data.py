"""Data loading, splitting, transforms.

Owner: Data subteam. Keep functions small and config-driven.
"""
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def build_transforms(image_size: int, augment: bool):
    """Return (train_tf, eval_tf). Augmentation is applied to training only."""
    normalise = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    eval_tf = transforms.Compose(normalise)
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    return train_tf, eval_tf


class HFImageDataset(Dataset):
    """Wraps a Hugging Face dataset split with a torchvision transform."""

    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.transform(item["image"].convert("RGB")), item["label"]


def load_and_split(cfg: dict, seed: int):
    """Load the dataset, take a subset, and return train/val/test HF splits.

    TODO (Data subteam): verify the column names, use a stratified split if
    available, and check the label distribution is preserved after splitting.
    """
    ds = load_dataset(cfg["hf_dataset"], split="train")
    ds = ds.shuffle(seed=seed)
    frac = cfg["subset_fraction"]
    if frac < 1.0:
        ds = ds.select(range(int(len(ds) * frac)))

    val_test = cfg["split"]["val"] + cfg["split"]["test"]
    first = ds.train_test_split(test_size=val_test, seed=seed)
    second = first["test"].train_test_split(
        test_size=cfg["split"]["test"] / val_test, seed=seed
    )
    return first["train"], second["train"], second["test"]


def make_loaders(cfg: dict, seed: int):
    train_s, val_s, test_s = load_and_split(cfg, seed)
    train_tf, eval_tf = build_transforms(cfg["image_size"], cfg["augment"])
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    return (
        DataLoader(HFImageDataset(train_s, train_tf), shuffle=True, **kw),
        DataLoader(HFImageDataset(val_s, eval_tf), **kw),
        DataLoader(HFImageDataset(test_s, eval_tf), **kw),
    )
