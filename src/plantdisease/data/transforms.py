"""Image transforms shared by training, evaluation, the smoke test and the app.

One place defines how an image becomes a tensor, so the app preprocesses
exactly like training did. Settings come from the config:

    model.input_size                      resize target (square)
    model.normalize_mean / normalize_std  optional, set both or neither
    data.augmentation                     training-only, see build_train_transform
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


def build_train_transform(model_cfg, augmentation_cfg=None):
    """The evaluation transform with random augmentation added. Training only.

    `augmentation_cfg` is the data.augmentation section, for example:

        enabled: true
        rotation_degrees: 15
        brightness: 0.2
        saturation: 0.2
        blur_probability: 0.2

    With `enabled` false or no section this equals the evaluation transform,
    which is what the "augmentation off" experiments use.
    """
    aug = augmentation_cfg or {}
    if not aug.get("enabled", False):
        return build_eval_transform(model_cfg)

    size = model_cfg["input_size"]
    steps = [
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(aug.get("rotation_degrees", 15)),
        transforms.ColorJitter(
            brightness=aug.get("brightness", 0.2),
            saturation=aug.get("saturation", 0.2),
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
            p=aug.get("blur_probability", 0.2),
        ),
    ]
    return _finish(steps, model_cfg)
