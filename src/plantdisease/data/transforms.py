"""Image preprocessing and augmentation transforms"""

from datasets import load_dataset, Dataset
import torchvision.transforms as transforms
import torch

def resize_image(size):
    """Resize the image to the specified size."""

    return transforms.Resize(size)

def augment_image():
    """Apply a series of augmentations to the image."""

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    return transform

def normalise_image():
    """Normalise the image to have pixel values in the range [0, 1]."""

    return transforms.ToTensor()

def build_transform_pipeline(cfg):
    """Build a complete transform pipeline for image preprocessing and augmentation."""

    resize = resize_image(cfg["data"]["image_size"])
    augment = augment_image()
    tensor = normalise_image()

    train_transform = transforms.Compose([resize, augment, tensor])
    eval_transform = transforms.Compose([resize, tensor])

    return {"train": train_transform, "eval": eval_transform}




















