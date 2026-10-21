# Project Plan

Deadline: **Thursday 3 December 2026**. Plan written Wednesday 21 October 2026, which leaves about six weeks. Aim to freeze features by **26 Nov** so the last week is only report, video and polish.

## 0. Admin checklist (do in the first 48 hours)

- [ ] Confirm every member appears in the team allocations file and is registered on Team Feedback
- [ ] Read the Team Feedback handbook and the project management approach guide (linked from the course page)
- [ ] Hold the **kick-off meeting**, then the **project initiation meeting**; log minutes in Team Feedback
- [ ] One person creates the repo on **https://github.kcl.ac.uk/** (NOT github.com)
- [ ] Invite **k21226001 (Tom Kuipers)** to the repo (needed for the ClearML integration)
- [ ] Register the repo on Team Feedback so commits are attributed to each member
- [ ] Everyone gets ClearML credentials from the KCL server (https://5ccsagap.er.kcl.ac.uk/), runs `clearml-init`, and confirms they can see the shared workspace
- [ ] Agree a weekly meeting slot (all 6 attend; attendance is monitored)

## 1. Roles (6 members)

Every member must commit code every week and attend every meeting. These are monitored and feed the individual mark adjustment, so even report/video roles should own some code.

| # | Role | Main ownership | Also owns |
|---|------|----------------|-----------|
| 1 | Project lead / integration | Repo setup, branch protection, PR reviews, meeting minutes, risk log, Gantt chart | `utils.py`, tests, CI of README/requirements |
| 2 | Data lead | `data.py`, `prepare_data.py`, EDA notebook, splits, augmentation, ClearML Dataset | Report section 2 |
| 3 | CNN lead | `SimpleCNN` in `models.py`, architecture ablations | Report section 3 (architecture) |
| 4 | Training and tracking lead | `train.py` loop, early stopping, scheduler, ClearML logging, hyperparameter sweeps | Report section 4 (experiments) |
| 5 | Transfer learning and evaluation lead | ResNet18 fine-tuning, `evaluate.py`, confusion matrix, per-class metrics, top-k | Report section 4 (test results) |
| 6 | UI and deployment lead | `app/app.py`, HF Space, model upload, advanced UI features | Demo video script and recording |

Report sections 1, 5, 6, 7 are split: lead writes 1 and 7, the UI lead writes 5, everyone contributes to 6. Pair people up on tasks (data with CNN, tracking with transfer) so nobody is blocked.

## 2. Timeline

| Week | Dates | Goals | Done when |
|------|-------|-------|-----------|
| 1 | 21 to 27 Oct | Admin checklist, repo skeleton pushed, everyone can run the pipeline locally, EDA started | Each member has one merged PR |
| 2 | 28 Oct to 3 Nov | Data pipeline: subset (about 25%), 70/15/15 split, label distribution check, normalisation and augmentation; first CNN trains end to end; first ClearML task logged | Baseline accuracy visible in ClearML |
| 3 | 4 to 10 Nov | CNN iterations (depth, channels, dropout vs batch norm); ResNet18 transfer baseline; Gradio prototype running locally | At least 6 tagged experiments compared |
| 4 | 11 to 17 Nov | Hyperparameter sweeps (optimiser, lr, batch size, augmentation on/off); pick candidate models; start full-dataset runs; first HF Space deploy | Public Space reachable with a working model |
| 5 | 18 to 24 Nov | Full-data final training, test-set evaluation (confusion matrix, per-class precision/recall, top-5); advanced UI features; first report draft | Final model ID recorded in README |
| 6 | 25 Nov to 1 Dec | **Feature freeze 26 Nov.** Finish report (5 to 7 pages, LNCS template), record and edit demo video, clean code, final README, AI use statement | Full draft reviewed by everyone |
| Final | 2 to 3 Dec | Submission checks only, no new features | Submitted before 3 Dec, ideally 2 Dec |

## 3. Technical approach

### Data
- PlantVillage from the Hugging Face Hub (55,400 images, 256x256, 39 classes). Use the link in the assignment to fill `hf_dataset` in the config.
- Prototype on 20 to 30% of the data, then retrain the best candidates on 100%.
- EDA: class counts plot, sample images per class. Class imbalance fixes are optional.
- Split 70/15/15 using a shuffled, seeded split. Verify the label distribution is close to the original in all three splits and include this check in the report.
- Normalise with `ToTensor`; augment training images only (brightness, saturation, rotation, blur, flip). Never augment val/test.

### Models
- **Required:** CNN from scratch extending `nn.Module`. Start small (about 3 to 4 conv blocks), reduce resolution and increase channels with depth, finish with a linear layer to 39 outputs. Try Dropout2d/Dropout and BatchNorm2d/BatchNorm1d, and compare. Set `model.train()` / `model.eval()` explicitly.
- **Optional but recommended:** fine-tune pretrained ResNet18 and compare against the scratch CNN.
- Loss: cross-entropy. Optimiser: Adam or AdamW. Optional LR scheduler.
- Early stopping: stop when validation accuracy has not improved for 2 to 3 epochs.

### ClearML
- Use the KCL server only (not the public ClearML cloud).
- Several narrow tasks rather than one monolith: data preparation, training, evaluation.
- Log train/val loss and accuracy every epoch, upload the best checkpoint, connect all hyperparameters via `task.connect`.
- Tag every run (architecture, `final-benchmark`, `scratch-ignore` for throwaway runs) so comparisons are clean.
- Version the processed dataset with ClearML Datasets.
- Use ClearML's compare tool to produce the plots and tables for the report.

### Evaluation
- Validation accuracy during training; on the test set report accuracy, top-5 accuracy, per-class precision/recall and a confusion matrix, plus a short error analysis (which classes get confused and why).

### UI and deployment
- Gradio: image upload, resize to model input size, top-5 (or top-10) probabilities as a table or bar chart. This gets the pass mark.
- Extras for more marks (pick a few and do them well): example image gallery, model selector (CNN vs ResNet), flag-wrong-prediction button, batch upload, show training examples of the predicted class.
- Deploy to a **public** Hugging Face Space. Load the model from the HF Hub, from a weights file in the Space repo, or from ClearML via Space secrets.
- Test locally first; Space builds are slow.

## 4. Git workflow (this is marked)

- `main` is always working. Nobody pushes directly to `main`.
- One branch per feature (`feature/<name>-<topic>`), small focused commits, a pull request for each, reviewed by at least one other member.
- Commit steadily through the weeks, not in a burst before the deadline.
- Never commit data, model files or `.env`. Real code only; no fake commit activity (that is treated as misconduct).
- See `CONTRIBUTING.md` for details.

## 5. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| A member disengages | Medium | Weekly commitments in the minutes, early check-in, pair tasks, escalate to the module organiser early |
| Uneven commit history | Medium | Give every member a code task each week; track in Team Feedback |
| ClearML/GPU queue congestion near the deadline | High | Finish heavy training by 24 Nov, use the 25% subset for iteration |
| Model does not converge | Medium | Start simple, check the loss on a tiny batch first, lower the LR, verify normalisation |
| Data leakage or bad splits | Low | Seeded split done once in one place, check label distributions, keep test set untouched until the end |
| HF Space build or memory problems | Medium | Deploy a stub early (Week 3 to 4), keep the model small, test locally |
| Merge conflicts | Medium | Small PRs, clear file ownership, frequent pulls |
| Report/video squeezed at the end | High | Draft sections as work completes, feature freeze 26 Nov, script the video in Week 5 |
| Credentials leaked | Low | `.env` ignored, use HF Space secrets |

Add any unforeseen risks (and how you handled them) as they occur. The report's project management section needs this.

## 6. Submission checklist

- [ ] Code archive without dataset or model files, plus `requirements.txt`
- [ ] README: team name and members, repo link, HF Space link, final model ID or task ID, setup instructions
- [ ] Report PDF, 5 to 7 pages, Springer LNCS template, all 7 sections (including Gantt and risk list)
- [ ] Demo video: about 2 to 3 min for the core app, up to 5 to 6 min if feature-rich, with narration
- [ ] AI use statement
- [ ] HF Space is public and working
- [ ] Peer feedback questionnaire completed on Team Feedback (every member, individually)
- [ ] Meeting minutes all logged
