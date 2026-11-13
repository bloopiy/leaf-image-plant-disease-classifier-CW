# Plant Disease Classifier

Plant disease classification from leaf images (PlantVillage). PyTorch, ClearML, Gradio, Hugging Face Spaces.

## Team
- Team: TODO
- Members: TODO (add as the assignment requires)

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

Run the offline tests (no network, no ClearML account needed):

```bash
python -m pytest -m "not real_data" -q
```

Run the full suite, which also downloads the Hugging Face dataset:

```bash
python -m pytest -q
```

Check the whole pipeline on a small slice of the real data:

```bash
python scripts/smoke_test.py --config configs/default.yaml
python scripts/smoke_test.py --config configs/transfer.yaml
```

### Experiments (ClearML)

```bash
python scripts/prepare_data.py                                  # split sizes, label plots, ClearML Dataset
python scripts/train.py --config configs/default.yaml           # CNN from scratch
python scripts/train.py --config configs/transfer.yaml          # ResNet18 transfer
python scripts/evaluate.py --task-id <training task id>         # prototype_test
python scripts/evaluate.py --task-id <id> --split permanent_test --final   # once, for the chosen model
```

Add `--offline` to any script to run without a ClearML account. The permanent
test set is only used with `--final`.

### App

```bash
python app/app.py
```

The app reads the configuration saved inside the checkpoint, so it works for
both models without extra settings.
