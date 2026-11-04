"""Gradio app: upload a leaf photo and see the top predicted diseases."""
import sys
from pathlib import Path

import gradio as gr
import yaml

APP_DIR = Path(__file__).resolve().parent


def find_root():
    """Return the folder holding configs/ and src/: the repo, or a deployed Space."""
    for root in (APP_DIR.parent, APP_DIR):
        if (root / "configs" / "app.yaml").is_file():
            return root
    raise FileNotFoundError("configs/app.yaml not found")


ROOT = find_root()
# Make `plantdisease` importable without installing the package.
sys.path.insert(0, str(ROOT / "src"))

from inference import build_preprocess, predict_topk  # noqa: E402
from model_loader import load_class_names, load_model  # noqa: E402


def load_settings():
    """Read the app settings and the model config they point to."""
    app_cfg = yaml.safe_load((ROOT / "configs" / "app.yaml").read_text())["app"]
    model_cfg = yaml.safe_load((ROOT / "configs" / app_cfg["model_config"]).read_text())
    return app_cfg, model_cfg


CFG, MODEL_CFG = load_settings()
CLASS_NAMES = load_class_names(APP_DIR / CFG["class_names"])
if len(CLASS_NAMES) != MODEL_CFG["model"]["num_classes"]:
    raise ValueError(f"{len(CLASS_NAMES)} class names but the model has {MODEL_CFG['model']['num_classes']} outputs")
MODEL = load_model(MODEL_CFG, ROOT / CFG["weights"] if CFG["weights"] else None)
PREPROCESS = build_preprocess(MODEL_CFG["model"]["input_size"])


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
