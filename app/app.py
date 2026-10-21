"""Gradio app deployed to the Hugging Face Space.

Minimal requirements: image upload, resize to model resolution, top-5 probabilities.
Extras to consider: example gallery, model selector, flag button, batch upload.

TODO (App subteam): replace CLASS_NAMES, load real weights (HF hub / space file /
ClearML), and match IMAGE_SIZE and model config to the trained model.
"""
import sys
from pathlib import Path

import gradio as gr
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plantdisease.models import SimpleCNN

IMAGE_SIZE = 128
CLASS_NAMES = [f"class_{i}" for i in range(39)]  # TODO: real 39 label names
WEIGHTS_PATH = "model.pt"                          # TODO: fetch from HF hub / ClearML

preprocess = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])


def load_model():
    model = SimpleCNN([32, 64, 128, 256], len(CLASS_NAMES))  # TODO: match trained config
    if Path(WEIGHTS_PATH).exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    return model.eval()


MODEL = load_model()


def predict(image):
    x = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1)[0]
    top = torch.topk(probs, 5)
    return {CLASS_NAMES[i]: float(p) for p, i in zip(top.values, top.indices)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Leaf image"),
    outputs=gr.Label(num_top_classes=5, label="Top predictions"),
    title="Plant Disease Classifier",
)

if __name__ == "__main__":
    demo.launch()
