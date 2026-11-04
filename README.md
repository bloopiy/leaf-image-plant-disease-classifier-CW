# Plant Disease Classifier

Plant disease classification from leaf images (PlantVillage). PyTorch, ClearML, Gradio, Hugging Face Spaces.

## Team
- Team: TODO
- Members: Kaveen Ovitigala (K25103019), Zaid Mukthar (K25091896), Aravindhakshan Sriram (k25038743), Amir Hoque (k23157382), Swaraj Singh (k24028164), Adam Abdelmeguid (k25055046)

## Links
- Hugging Face Space: TODO
- Final model (ClearML model/task ID): TODO

## Setup

Verified locally on Windows with Python 3.12 and an NVIDIA GPU.

Create an environment:

```bash
py -3.12 -m venv .venv
```

Install the verified CUDA 12.6 PyTorch pair, then project dependencies:

```bash
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For other hardware, select the appropriate build using the official
PyTorch installation instructions.

Configure ClearML using a personal account:

```bash
.\.venv\Scripts\clearml-init.exe
```

Enter credentials locally. Never commit credentials or clearml.conf.

## Usage

Run the local tests:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

Run the synthetic training-and-tracking smoke test:

```bash
.\.venv\Scripts\python.exe scripts/smoke_train.py
```

This creates a ClearML experiment with configuration, training/validation
curves, checkpoint, history and script artifacts.

The smoke test uses artificial dark/bright images. Its accuracy is not
PlantVillage performance. Real-data training still requires integration
with the project model and data-loading modules.
