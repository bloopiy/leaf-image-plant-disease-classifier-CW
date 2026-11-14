"""Image preprocessing and augmentation transforms"""

import torchvision.transforms as transforms

def resize_image(size):
    """Resize the image to the specified size."""

    return transforms.Resize((size, size))

def augment_image(cfg):
    """Apply a series of augmentations to the image."""

    augment_cfg = cfg["data"]["transformations"]["augment"]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=augment_cfg["flip_prob"]),
        transforms.RandomRotation(degrees=augment_cfg["degrees"]),
        transforms.ColorJitter(
            brightness=augment_cfg["brightness"],
            contrast=augment_cfg["contrast"],
            saturation=augment_cfg["saturation"],
            hue=augment_cfg["hue"],
        ),
    ])

    return transform

def normalise_image():
    """Normalise the image to have pixel values in the range [0, 1]."""

    return transforms.ToTensor()

def build_transform_pipeline(cfg):
    """Build a complete transform pipeline for image preprocessing and augmentation."""

    if cfg["data"]["transformations"].get("resize", True):
        resize = resize_image(cfg["data"]["image_size"])
    else:
        resize = transforms.Lambda(lambda x: x)

    augment = augment_image(cfg)
    tensor = normalise_image()

    train_transform = transforms.Compose([resize, augment, tensor])
    eval_transform = transforms.Compose([resize, tensor])

    return {"train": train_transform, "eval": eval_transform}




















