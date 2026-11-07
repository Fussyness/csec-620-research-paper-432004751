#!/usr/bin/env bash
# Run every (experiment x seed) job of a config in parallel, one job per GPU at a
# time, then merge the results and regenerate the paper tables.
#
#   bash scripts/lovegood/run_parallel.sh CONFIG "GPU IDS"
#   e.g. bash scripts/lovegood/run_parallel.sh config.yaml "0 1 2 3 4"
#
# lovegood is shared and has no scheduler: check `nvidia-smi` first and list only
# idle GPUs. Start this inside tmux (or with nohup) so it survives logging out.
# Logs go to logs/<config>/; per-job results to <output.dir>/<exp>/runs/.
set -euo pipefail

CONFIG=${1:?usage: run_parallel.sh CONFIG "GPU IDS"}
GPUS=(${2:?usage: run_parallel.sh CONFIG "GPU IDS"})
PY=.venv/bin/python
LOGS="logs/$(basename "$CONFIG" .yaml)"
mkdir -p "$LOGS"

# Share the 48 CPU cores between the jobs instead of each job grabbing all of them.
export OMP_NUM_THREADS=$(( $(nproc) / ${#GPUS[@]} > 1 ? $(nproc) / ${#GPUS[@]} : 1 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# 1. Build descriptor caches once (rendering BIG 2015, GIST, resized CNN inputs).
echo "[$(date +%T)] preparing caches"
$PY -m src.train --config "$CONFIG" --prepare > "$LOGS/prepare.log" 2>&1

# 2. Job list: one line "EXP SEED" per job; experiments that carry attacks
#    (the longest jobs) go first so they don't end up last in the queue.
JOBS="$LOGS/jobs.txt"
$PY - "$CONFIG" > "$JOBS" <<'EOF'
import sys
from src.train import load_config
cfg = load_config(sys.argv[1])
specs, wanted = cfg["experiments"], cfg["run"]["experiments"]
bases = [specs[e]["attack_on"] for e in wanted if "attack_on" in specs[e]]
trained = [e for e in wanted if "attack_on" not in specs[e]] + [b for b in bases if b not in wanted]
for exp in sorted(trained, key=lambda e: e not in bases):
    for seed in cfg["seeds"]:
        print(exp, seed)
EOF
echo "[$(date +%T)] $(wc -l < "$JOBS") jobs on GPUs ${GPUS[*]} (OMP threads per job: $OMP_NUM_THREADS)"

# 3. Workers: each GPU repeatedly takes the next unclaimed job from the list.
COUNTER="$LOGS/.next"; echo 0 > "$COUNTER"
next_job() {
    flock "$COUNTER.lock" bash -c '
        n=$(cat "$0"); echo $((n + 1)) > "$0"
        sed -n "$((n + 1))p" "$1"' "$COUNTER" "$JOBS"
}
worker() {
    local gpu=$1 exp seed
    while read -r exp seed < <(next_job) && [ -n "${exp:-}" ]; do
        echo "[$(date +%T)] GPU $gpu: start $exp seed $seed"
        if CUDA_VISIBLE_DEVICES=$gpu $PY -m src.train --config "$CONFIG" \
               --experiments "$exp" --seeds "$seed" > "$LOGS/${exp}_seed${seed}.log" 2>&1; then
            echo "[$(date +%T)] GPU $gpu: done  $exp seed $seed"
        else
            echo "[$(date +%T)] GPU $gpu: FAILED $exp seed $seed (see $LOGS/${exp}_seed${seed}.log)"
        fi
    done
}
for g in "${GPUS[@]}"; do worker "$g" & done
wait

# 4. Merge whatever finished (missing seeds are reported), then build the tables.
echo "[$(date +%T)] merging"
$PY -m src.train --config "$CONFIG" --merge | tee "$LOGS/merge.log"
OUT=$($PY -c "import sys; from src.train import load_config; print(load_config(sys.argv[1])['output']['dir'])" "$CONFIG")
$PY -m src.tables --results "$OUT"
echo "[$(date +%T)] all done"
