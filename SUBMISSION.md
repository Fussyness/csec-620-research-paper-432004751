# Submission Manifest

Fill this in before submitting. The grading agent reads it first.

## Author
- Name (RIT ID): sole author (individual graduate research paper)

## Course level
- [ ] 520 (undergraduate)
- [x] 620 (graduate): research paper, Track B (reproduction / extension)

## Problem
Image-based (byteplot) malware family classification. This work reproduces Nataraj et
al. (2011) GIST + k-NN on Malimg, then tests whether its ~98% accuracy survives a
duplicate-aware split, a second dataset (BIG 2015), and functionality-preserving append
attacks, with a fine-tuned ResNet as the stronger baseline.

## Dataset
- Malimg: version / source / PNG count / SHA-256: see `data/README.md` (Recorded line)
- Microsoft BIG 2015 (Kaggle): see `data/README.md`
- How to obtain: manual download, steps in `data/README.md`
- Target: malware family (25 classes Malimg, 9 classes BIG 2015)

## How to reproduce
```bash
make setup
make reproduce      # full run, config.yaml
```
Anything non-default the grader must know:
- Needs both datasets in `data/` (manual download).
- Full run takes ____ hours on a ____ GPU. `make quick` (config.quick.yaml, 1 seed) takes ____.
- Reported numbers come from: `config.yaml` / `config.quick.yaml` (circle one).
- ImageNet weights for ResNet-50 are downloaded by torchvision on first run.

## Claimed results
Headline = E1 (stratified 10-fold, Malimg), GIST + k-NN, mean over seeds; must match
`results/metrics.json`. Metrics are macro-averaged over families; ROC-AUC is one-vs-rest.
- Accuracy:
- Precision / Recall / F1 (macro):
- ROC-AUC (macro OvR):

## AI-use acknowledgment
- Milestone 2: Claude (Anthropic) helped find and verify references, draft first-pass
  annotations, and propose the outline and experiment plan.
- Final draft (2 Nov 2026): Claude (Anthropic) scaffolded the experiment pipeline in this
  repo (data loading, dedup and folds, ResNet/k-NN/LR wrappers, evaluation, tables,
  tests, configs) and the LaTeX skeleton and `.bib`. GIST (`src/features.py: gist`) and the
  attacks (`src/attack.py`) were written by the author. The paper's prose, analysis, and
  argument are the author's own.
