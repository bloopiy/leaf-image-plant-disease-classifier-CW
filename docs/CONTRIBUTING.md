# Contributing

## Branches
- `main`: stable, always runs. No direct pushes.
- `feature/<name>-<topic>` for new work, e.g. `feature/sam-augmentation`.
- `fix/<topic>` for bug fixes.

## Commits
- Small and focused, one idea per commit.
- Imperative messages: `Add Dropout2d option to SimpleCNN`.
- Commit regularly throughout the week.

## Pull requests
1. Pull the latest `main`, create your branch.
2. Push and open a PR with a short description of what and why.
3. At least one teammate reviews before merging. Squash or merge, then delete the branch.

## Code standards
- Hyperparameters live in `configs/*.yaml`, never hardcoded.
- Reusable logic goes in `src/plantdisease/`; `scripts/` stay thin.
- Short functions, docstrings for non-obvious code, avoid repetition and deep nesting.
- Run `pytest` before opening a PR.

## Do not commit
- Datasets, model weights (`.pt`, `.pth`, `.ckpt`), `.env`, or any credentials.

## ClearML
- Always tag runs. Use `scratch-ignore` for throwaway experiments and `final-benchmark` for runs that go in the report.
- Use the KCL server only.
