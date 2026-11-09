---
title: Plant Disease Classifier
sdk: gradio
app_file: app.py
pinned: false
---

# Plant Disease Classifier

Upload a photo of a leaf and the app shows the five most likely diseases (or healthy) across 39 PlantVillage classes.

## Run locally

From the repo root:

```bash
pip install -r app/requirements.txt
python app/app.py
```

Then open http://127.0.0.1:7860. Settings such as input size and the weights path live in `configs/app.yaml`.
