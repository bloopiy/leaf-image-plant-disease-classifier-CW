"""Gradio app: upload a leaf photo and see the top predicted diseases."""
from pathlib import Path

import gradio as gr
import yaml

from inference import build_preprocess, predict_topk
from model_loader import load_class_names, load_model

APP_DIR = Path(__file__).resolve().parent


def find_config():
    """Locate app.yaml in the repo (configs/ beside app/) or in a deployed Space."""
    for root in (APP_DIR.parent, APP_DIR):
        path = root / "configs" / "app.yaml"
        if path.is_file():
            return path
    raise FileNotFoundError("configs/app.yaml not found")


CFG = yaml.safe_load(find_config().read_text())["app"]
CLASS_NAMES = load_class_names(APP_DIR / CFG["class_names"])
MODEL = load_model(CFG["weights"], len(CLASS_NAMES))
PREPROCESS = build_preprocess(CFG["image_size"])


def classify(image):
    """Gradio callback: run the model on one uploaded image."""
    if image is None:
        return None
    return predict_topk(image, MODEL, PREPROCESS, CLASS_NAMES, k=CFG["top_k"])


def build_ui():
    """Lay out the upload box, button and results."""
    with gr.Blocks(title="Plant Disease Classifier") as demo:
        gr.Markdown("# Plant Disease Classifier\nUpload a photo of a leaf to see the most likely diseases.")
        with gr.Row():
            image = gr.Image(type="pil", label="Leaf image")
            result = gr.Label(num_top_classes=CFG["top_k"], label="Top predictions")
        gr.Button("Classify", variant="primary").click(classify, inputs=image, outputs=result)
    return demo


if __name__ == "__main__":
    build_ui().launch()
