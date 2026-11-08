# Submission Manifest

Fill this in before submitting. The grading agent reads it first.

## Author
- Theodore DellaVilla (432004751)

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
make reproduce              # seconds, no data: rebuilds metrics.json + paper tables
make reproduce-scratch      # hours, needs both datasets: recomputes everything
```
Anything non-default the grader must know:
- `make reproduce` (what `grading/grade.py` runs) reuses the committed per-seed raw
  results in `results/<exp>/runs/`. Each records the config fingerprint and git commit
  it was produced with; a mismatching file stops the run instead of being reused. It
  then recomputes every summary, `results/metrics.json` and the paper's tables from
  them. It shows that the paper's numbers follow from the raw results, not that the
  raw results can be recreated; that is what `make reproduce-scratch` does. See README
  "Reproducing".
- `make reproduce-scratch` needs both datasets in `data/` (manual download, see
  `data/README.md`) and a GPU.
- Full run: 2 h 12 min wall-clock (8 Nov 2026, 14:25:14-16:37:20, including the
  cache build and the merge) on lovegood.cs.rit.edu with 10 x Tesla T4 (15 jobs in
  parallel, one GPU each; the longest, an E2 seed with the E4 attacks, took 2.05 h).
  12.9 GPU-hours in total, so about 13 h on a single T4. Mean time per job: E1 27.5 min, E2 + E4 121 min, E3 6 min
  (the `seconds` field of each `results/<exp>/runs/seed*.json`).
- `make quick` (config.quick.yaml, 1 seed, 5 folds) was not timed end to end; from
  the per-job times above it is roughly 1-1.5 h on one T4.
- Reported numbers come from: `config.yaml` (5 seeds x 10 folds).
- ImageNet weights for ResNet-50 are downloaded by torchvision on first run.

## Claimed results
Headline = E1 (stratified 10-fold, Malimg), GIST + k-NN, mean over seeds; must match
`results/metrics.json`. Metrics are macro-averaged over families; ROC-AUC is one-vs-rest.
Values are mean ± std over the 5 seeds, in %.
- Accuracy: 97.46 ± 0.04 (Nataraj et al. report 98%)
- Precision / Recall / F1 (macro): 93.67 ± 0.12 / 93.66 ± 0.08 / 93.56 ± 0.09
- ROC-AUC (macro OvR): 98.70 ± 0.06

Every other number in the paper (E2-E4, the CNN and byte-histogram baselines, the
per-family table, the threshold sensitivity table) is generated from `results/` into
`report/generated/` by `make reproduce`.

## AI-use acknowledgment
- Milestone 2: Claude (Anthropic) helped find and verify references, draft first-pass
  annotations, and propose the outline and experiment plan.
- Final draft (2 Nov 2026): Claude (Anthropic) scaffolded the experiment pipeline in this
  repo (data loading, dedup and folds, ResNet/k-NN/LR wrappers, evaluation, tables,
  tests, configs) and the LaTeX skeleton and `.bib`. GIST (`src/features.py: gist`) and the
  attacks (`src/attack.py`) were written by the author.
- Later (Nov 2026), also by Claude: the lovegood parallel-run scripts
  (`scripts/lovegood/`), the FGSM step-size pilot (`scripts/fgsm_eps_pilot.py`), the
  near-duplicate threshold analysis (`scripts/dedup_threshold.py`), the testable-family
  metrics, and the resume-from-committed-results path of `make reproduce`. It also
  walked the author through implementing GIST and the attacks step by step, reviewing
  and testing each piece; the code in those two files was typed and written by the
  author.
- Paper (Nov 2026): Claude drafted the body text of `report/main.tex` from the
  author's results, leaving the result values as marked gaps for the author to fill
  in and check. [Author: describe your own edits here.]
