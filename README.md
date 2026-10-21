# Plant Disease Classifier

Image classification of plant diseases from leaf pictures (PlantVillage, 39 classes).
Stack: PyTorch, ClearML (KCL server), Gradio, Hugging Face Spaces.

## Team
- Team name/number: TODO
- Members: TODO (6 names)

## Links (fill in before submission)
- Git repo: TODO (KCL GitHub, not github.com)
- Hugging Face Space (must be public): TODO
- Final model: ClearML model ID / task ID: TODO

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # never commit real credentials
clearml-init                # point at https://5ccsagap.er.kcl.ac.uk/ (KCL server, NOT public ClearML)
```

## Pipeline
```bash
python scripts/prepare_data.py --config configs/default.yaml      # subset + splits + ClearML dataset
python scripts/train.py --config configs/default.yaml --model cnn # from-scratch CNN
python scripts/train.py --config configs/transfer.yaml --model resnet18
python scripts/evaluate.py --config configs/default.yaml --task-id <clearml_task_id>
python app/app.py                                                  # local Gradio UI
```

## Repo layout
- `src/plantdisease/` reusable code (data, models, training, evaluation)
- `scripts/` thin CLI entry points, each maps to one ClearML task
- `configs/` hyperparameters (no hardcoded values in code)
- `app/` Gradio app deployed to the HF Space
- `notebooks/` EDA only
- `docs/` plan, meeting notes template, report notes
- `tests/` basic tests

## Workflow
See `docs/CONTRIBUTING.md` and `docs/PLAN.md`.
