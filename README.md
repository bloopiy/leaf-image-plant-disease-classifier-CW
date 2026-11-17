# Plant Disease Classifier

Plant disease classification from leaf images (PlantVillage). PyTorch, ClearML, Gradio, Hugging Face Spaces.

## Team
- Team: TODO
- Members: TODO (add as the assignment requires)

## Links
- Hugging Face Space: TODO
- Final model (ClearML task ID): [`9d0570880da2487b9dab08bf211f6834`](https://app.clear.ml/projects/2e8c7318aa8d41ff89fce07bd7a4c819/tasks/9d0570880da2487b9dab08bf211f6834/output/log) — ResNet18 fine-tune, 0.994 validation accuracy on the full development pool
- Permanent-test accuracy: pending (one-shot evaluation, run once on the final model)

### Experiment log (ClearML project `PlantDisease-CW`)

| Run | Task ID | Status |
| --- | --- | --- |
| Comparison 25% pool — SimpleCNN 0.912 / ResNet18 0.989 val acc | [`b1786fd7…`](https://app.clear.ml/projects/2e8c7318aa8d41ff89fce07bd7a4c819/tasks/b1786fd7739c47a7b0d9ac464425db83) / [`e72a6a4c…`](https://app.clear.ml/projects/2e8c7318aa8d41ff89fce07bd7a4c819/tasks/e72a6a4c4a444707aa99061253892626) | dev pipeline |
| Full pool — SimpleCNN, 0.975 val acc | [`0c098d52…`](https://app.clear.ml/projects/2e8c7318aa8d41ff89fce07bd7a4c819/tasks/0c098d520b754f3e8e24d70b66915b31) | dev pipeline, `final-benchmark` |
| Full pool — ResNet18, 0.994 val acc | [`9d057088…`](https://app.clear.ml/projects/2e8c7318aa8d41ff89fce07bd7a4c819/tasks/9d0570880da2487b9dab08bf211f6834) | dev pipeline, `final-benchmark`, **final model** |

All runs above were trained on the dev pipeline; the notebook's provenance
guard verifies the zip (fingerprint matched dev @ 1d04e41). The 15 Nov runs
that used a master-branch zip by mistake are kept in ClearML tagged
`scratch-ignore` — caught in PR review before the permanent-test
evaluation and retrained the same day.

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
