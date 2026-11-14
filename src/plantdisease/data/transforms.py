"""Image transforms shared by training, evaluation, the smoke test and the app.

One place defines how an image becomes a tensor, so the app preprocesses
exactly like training did. Settings come from the config:

    model.input_size                      resize target (square)
    model.normalize_mean / normalize_std  optional, set both or neither
    data.transformations.augment          training-only, see build_train_transform
"""
from torchvision import transforms


def _normalisation(model_cfg):
    """Return (mean, std) from the model config, or None if not configured."""
    mean = model_cfg.get("normalize_mean")
    std = model_cfg.get("normalize_std")
    if (mean is None) != (std is None):
        raise ValueError("model.normalize_mean and model.normalize_std must be set together")
    return None if mean is None else (mean, std)


def _finish(steps, model_cfg):
    """Append ToTensor and, if configured, Normalize, then compose."""
    steps = list(steps) + [transforms.ToTensor()]
    normalisation = _normalisation(model_cfg)
    if normalisation is not None:
        steps.append(transforms.Normalize(*normalisation))
    return transforms.Compose(steps)


def build_eval_transform(model_cfg):
    """Resize, convert to a tensor, and normalise if configured. No augmentation.

    Use this for validation, test and the app.
    """
    size = model_cfg["input_size"]
    return _finish([transforms.Resize((size, size))], model_cfg)


def build_train_transform(model_cfg, augment_cfg=None):
    """The evaluation transform with random augmentation added. Training only.

    `augment_cfg` is the data.transformations.augment section, for example:

        flip_prob: 0.5
        degrees: 15
        brightness: 0.2
        contrast: 0.2
        saturation: 0.2
        hue: 0.1
        blur_probability: 0.2     # optional, default 0 (off)
        enabled: true             # optional, default true when the section exists

    With no section, or `enabled: false`, this equals the evaluation transform,
    which is what the "augmentation off" experiments use.
    """
    aug = augment_cfg
    if not aug or not aug.get("enabled", True):
        return build_eval_transform(model_cfg)

    size = model_cfg["input_size"]
    steps = [
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=aug.get("flip_prob", 0.5)),
        transforms.RandomRotation(aug.get("degrees", 15)),
        transforms.ColorJitter(
            brightness=aug.get("brightness", 0.2),
            contrast=aug.get("contrast", 0.2),
            saturation=aug.get("saturation", 0.2),
            hue=aug.get("hue", 0.0),
        ),
    ]
    blur = aug.get("blur_probability", 0.0)
    if blur > 0:
        steps.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=blur))
    return _finish(steps, model_cfg)
