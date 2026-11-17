# Plant Disease Classifier

Plant disease classification from leaf images (PlantVillage). PyTorch, ClearML, Gradio, Hugging Face Spaces.

## Team
- Team: TODO
- Members: TODO (add as the assignment requires)

## Links
- Hugging Face Space: TODO
- Final model (ClearML task ID): pending — retraining on the dev pipeline (see the experiment log)
- Permanent-test accuracy: pending (one-shot evaluation, run once on the final model)

### Experiment log (ClearML project `PlantDisease-CW`)

| Run | Task ID | Status |
| --- | --- | --- |
| Comparison 25% pool — SimpleCNN 0.894 / ResNet18 0.993 val acc | [`11919653…`](https://app.clear.ml/projects/2e8c7318aa8d41ff89fce07bd7a4c819/tasks/1191965337af49f7b18f646ffdda738d) / [`ebae7a60…`](https://app.clear.ml/projects/2e8c7318aa8d41ff89fce07bd7a4c819/tasks/ebae7a60c2524cc4b73c60b22632b9f5) | master pipeline — superseded, retraining |
| Full pool — SimpleCNN, 0.960 val acc | [`76fe7fa9…`](https://app.clear.ml/projects/2e8c7318aa8d41ff89fce07bd7a4c819/tasks/76fe7fa9ff2f4906a628136775bc4522) | master pipeline — superseded, retraining |
| Full pool — ResNet18, 0.994 val acc | [`54df76bd…`](https://app.clear.ml/projects/2e8c7318aa8d41ff89fce07bd7a4c819/tasks/54df76bd8eb0441eaf0268912d117b66) | master pipeline — superseded, retraining |

The 15 Nov Kaggle runs were trained from a zip of the master branch by
mistake (master has no normalisation / augmentation pipeline). This was
caught in PR review before the permanent-test evaluation. The dev-pipeline
retrained runs will replace this table; the superseded tasks are tagged
`scratch-ignore` in ClearML.

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

No local GPU? `kaggle-training.ipynb` runs the same training commands on a
free Kaggle T4 and tracks to the same ClearML project.

### App

```bash
python app/app.py
```

The app reads the configuration saved inside the checkpoint, so it works for
both models without extra settings.
