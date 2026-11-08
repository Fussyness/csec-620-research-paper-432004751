# Running on lovegood.cs.rit.edu

`lovegood` is one shared server with 10 Tesla T4 GPUs (16 GB each), 48 cores and
376 GB RAM. It has **no job scheduler**: anyone can use any GPU, so check
`nvidia-smi` and use only idle GPUs. Your home directory is network storage
(`/home/stu4/...`). `/tmp` is a RAM disk, so keep data out of it.

Replace `USER` below with your RIT username.

**Measured (final run, 8 Nov 2026, `config.yaml`, all 10 GPUs):** 2 h 12 min wall-clock
(14:25:14 to 16:37:20), including the cache build and the merge. 15 jobs, 12.9
GPU-hours in total; mean per job E1 27.5 min, E2 + E4 attacks 121 min (longest 2.05 h),
E3 6 min. When the Malimg caches already exist, the cache stage only renders BIG 2015
(about 5-10 min).

## 1. One-time setup

On the server:

```bash
git clone https://github.com/Fussyness/csec-620-research-paper-432004751.git
cd csec-620-research-paper-432004751
bash scripts/lovegood/setup.sh       # venv, CUDA PyTorch, ResNet weights, tests
```

From your laptop (Git Bash, repo root), copy the data, which is not in git:

```bash
scp data/archive.zip USER@lovegood.cs.rit.edu:csec-620-research-paper-432004751/data/
scp -r data/extracted USER@lovegood.cs.rit.edu:csec-620-research-paper-432004751/data/   # 6.8 GB
```

Back on the server, unzip Malimg so the layout matches `data/README.md`
(`data/malimg_paper_dataset_imgs/<Family>/*.png`), then check the counts:

```bash
cd data && unzip -q archive.zip && cd ..
find data/malimg_paper_dataset_imgs -name '*.png' | wc -l     # 9339
ls data/extracted/train | wc -l                               # 2042
```

## 2. Run

```bash
git pull                                  # pick up your latest code (e.g. gist())
nvidia-smi                                # which GPUs are idle?
tmux new -s nataraj                       # survives logging out; detach: Ctrl-b d
bash scripts/lovegood/run_parallel.sh config.yaml "0 1 2 3 4"
```

The script:
1. builds the descriptor caches once (`--prepare`);
2. queues one job per (experiment, seed), 15 for `config.yaml`, and the GPUs you
   list each take the next job when free. E2 jobs go first because they also run
   the E4 attacks;
3. merges the results (`--merge`) and regenerates `report/generated/`.

Reattach with `tmux attach -t nataraj`. Progress lines show which GPU runs what;
each job logs to `logs/<config>/<exp>_seed<s>.log`. If `tmux` is missing, use
`nohup bash scripts/lovegood/run_parallel.sh config.yaml "0 1 2 3 4" > run.log 2>&1 &`.

A failed job is reported and the rest continue. To redo one job and remerge:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.train --config config.yaml --experiments e2 --seeds 3
.venv/bin/python -m src.train --config config.yaml --merge && .venv/bin/python -m src.tables --results results
```

## 3. Bring results back and commit them

From your laptop (repo root), replace the local results with the server's:

```bash
scp -r USER@lovegood.cs.rit.edu:csec-620-research-paper-432004751/results .
make reproduce                   # reuses them: rebuilds metrics.json + report/generated/
git add results report && git commit -m "Results of the full run" && git push
```

`.gitignore` admits the small result files (`results/metrics.json`, `results/e*/`,
`results/dedup_threshold/`) but not `results/splits/`. Committing them is what lets
`make reproduce` finish in seconds for a grader without the datasets (README
"Reproducing"). State in the paper and `SUBMISSION.md` that the reported numbers came
from this server (10x T4, N GPUs used) and give the wall-clock time from the run log.

## Notes

- Each job uses about 48 / (number of GPUs) CPU threads, so the jobs don't fight
  over the cores.
- Courtesy: if others are using the server, use fewer GPUs. The queue adapts to
  any number.
- The same flags work on your own PC: `python -m src.train --config config.yaml
  --experiments e1 --seeds 0` runs one slice, and `--merge` combines the slices.
