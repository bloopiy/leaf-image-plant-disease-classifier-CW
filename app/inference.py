"""Preprocess an image and return the top-k class probabilities."""
import torch
from torchvision import transforms


def build_preprocess(image_size):
    """Resize and normalise the same way as at training time."""
    # TODO: replace with plantdisease.data.build_transforms(image_size, augment=False)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def format_label(name):
    """Turn a dataset label like 'Tomato___Leaf_Mold' into 'Tomato · Leaf Mold'."""
    crop, _, condition = name.partition("___")
    parts = [part.replace("_", " ").strip() for part in (crop, condition) if part]
    return " · ".join(part[0].upper() + part[1:] for part in parts)


def predict_topk(image, model, preprocess, class_names, k=5):
    """Return {class_name: probability} for the k most likely classes."""
    x = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, k)
    return {class_names[i]: p.item() for p, i in zip(top.values, top.indices)}
