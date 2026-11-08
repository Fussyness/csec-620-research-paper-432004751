# Image-Based Malware Classification: Reproducing Nataraj et al. (2011)

CSEC 620 graduate research paper, Track B. Reproduces GIST + k-NN malware
classification on Malimg [1] and extends it with a CNN baseline, a duplicate-aware
split, a cross-dataset test on BIG 2015, and an append-attack robustness study.
Built on the course project template.

| Exp. | RQ | What | Output |
|---|---|---|---|
| E1 | Reproduction | [1]'s stratified 10-fold on Malimg | `results/e1/` |
| E2 | Leakage | grouped folds: exact (+ near) duplicates kept in one fold | `results/e2/` |
| E3 | Generalization | same pipeline on BIG 2015 | `results/e3/` |
| E4 | Robustness | append attacks on E2's test folds | `results/e4/` |

## Quick start

```bash
make setup                 # .venv + requirements (Windows or Unix)
make reproduce             # rebuild results/metrics.json + report/generated/ (see below)
make reproduce-scratch     # recompute everything from the datasets (hours, GPU)
make quick                 # 1 seed, 5 folds -> results_quick/ (needs the datasets)
make ci                    # synthetic sanity run, no data needed -> results_ci/
make test
```

## Reproducing

The full experiment cannot be recomputed inside a short grading window: both datasets
need a manual download behind a Kaggle login and cannot be committed (data/README.md),
and the full run (5 seeds x 10 folds, ResNet-50, E1-E4) takes about 2.5 h on 10 Tesla T4
GPUs (about 14 h on one). The repository therefore separates two levels:

| Command | Needs data | Time | What it shows |
|---|---|---|---|
| `make reproduce` | no | seconds | every number in the paper is computed from the committed raw results |
| `make reproduce-scratch` | yes | hours | the raw results themselves can be recreated |

- **Committed raw results.** `results/<exp>/runs/seed<s>.json` holds each seed's
  out-of-fold metrics, per-family recall, attack results and McNemar counts, written by
  the final run. Each file records the settings fingerprint of the config it was
  produced with and the git commit. `results/dedup_threshold/summary.json` holds the
  threshold sensitivity table. Figures from the run are committed next to them.
- **`make reproduce` resumes.** `src.train` reuses a seed's file only if its fingerprint
  matches the current `config.yaml`, computes any seed that is missing (which needs the
  data), then merges everything into `results/metrics.json`; `src/tables.py` writes the
  paper's tables from it. A file made with different settings stops the run with an
  error rather than being reused.
- **`make reproduce-scratch`** passes `--fresh`, ignores the committed files and
  recomputes everything. The 10-GPU version is `scripts/lovegood/run_parallel.sh`
  ([scripts/lovegood/README.md](scripts/lovegood/README.md)).

Fold assignments (`results/splits/`) are not committed; any run regenerates them
deterministically from the seeds.

## Who wrote what (AI-use policy)

The from-scratch parts of this reproduction are written by the student:

- `src/features.py: gist()`: the GIST descriptor [3] with [1]'s parameters
- `src/attack.py`: the E4 append and FGSM attacks

The surrounding pipeline (data loading, folds, models, evaluation, tables, tests) was
scaffolded with AI assistance and is acknowledged in the paper's AI-use appendix. Until
`gist()` and the attacks are implemented, `make reproduce` skips the parts that need
them and lists each skip under `"skipped"` in `results/metrics.json`. The contract tests
in `tests/test_smoke.py` skip until then too, and pass once the implementations meet
the contract.

## Repository layout

```
├── config.yaml          # full run (every knob lives here)
├── config.quick.yaml    # 1-seed run, inherits config.yaml
├── config.ci.yaml       # synthetic run for CI/tests, inherits config.yaml
├── src/
│   ├── data.py          # width table, byteplots, Malimg/BIG 2015/synthetic, dedup, folds
│   ├── features.py      # GIST (student), byte histogram, thumbnail
│   ├── model.py         # GIST+kNN, byte-histogram LR, ResNet
│   ├── attack.py        # E4 attacks (student)
│   ├── evaluate.py      # macro metrics, OvR ROC-AUC, McNemar, figures
│   ├── train.py         # entry point: experiments x seeds x folds
│   └── tables.py        # results/ -> report/generated/*.tex
├── data/                # NOT committed — see data/README.md
├── results/             # committed raw results of the final run; rebuilt by `make reproduce`
├── report/              # IEEE LaTeX paper (main.tex, refs.bib)
├── tests/               # pytest: width table, leakage, metrics, contracts, end-to-end
└── grading/             # course grading harness
```

## Grading (by an AI agent or a human)

This repo is designed to be graded by an agent. See `grading/`:

- `grading/rubric.yaml` — machine-readable rubric (weights sum to 100).
- `grading/grade.py` — objective check harness. Run `python grading/grade.py`
  to produce `grading/auto_report.json` (structure, tests, reproduce, metrics,
  leakage scan, report presence, git hygiene).
- `grading/AGENT_GRADING.md` — the protocol an agent follows to turn those checks
  plus qualitative judgment into `grading/grade.json` + `grading/grade_report.md`.

Before submitting, fill in `SUBMISSION.md` and make sure `python grading/grade.py`
looks clean.

## Requirements checklist (per the project handout)

- [ ] `make reproduce` runs clean from a fresh environment (uses committed results)
- [ ] Results (`results/`) match what your report claims
- [ ] Evaluation reports precision / recall / F1 / ROC-AUC, not just accuracy
- [ ] No data leakage (scaling/fitting on train only)
- [ ] Report in IEEE format (`report/`), code referenced in the appendix
- [ ] Author listed below

## Author

- Name (RIT ID)

## Academic integrity & AI use

Follow the syllabus AI-use policy: assistants are allowed as aids, but acknowledge
substantive AI use and do not submit AI-generated implementations for from-scratch tasks.
