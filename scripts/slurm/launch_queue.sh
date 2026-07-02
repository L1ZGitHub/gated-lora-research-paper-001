#!/bin/bash
# Multi-run orchestrator: runs every (config × seed) from experiments/queue.txt
# through chain_jobs.sh, with a cap on CONCURRENT runs.
#
# Each run occupies exactly 1 GPU at a time (train.sbatch asks --gres=gpu:1),
# so --max-concurrent IS the GPU footprint of the whole campaign. Default 4
# → at most 4 of the 18 pinned rtx6000 GPUs (turing-[4-9] × 3) are ours.
#
# The 4h MaxTime is handled INSIDE each chain: the trainer exits at 3h30 with
# a pushed "latest" checkpoint, chain_jobs.sh resubmits until TRAINING_DONE
# appears on HF Hub. Nothing is stored on the NFS home except tiny text logs.
#
# Usage (from the repo root on the Ensimag frontale):
#   # See what would run, without launching anything:
#   bash scripts/slurm/launch_queue.sh --dry-run
#
#   # Launch for real, surviving SSH disconnects:
#   nohup bash scripts/slurm/launch_queue.sh --max-concurrent 4 \
#       >> logs/launch_queue.log 2>&1 &
#
#   # Follow progress:
#   tail -f logs/launch_queue.log logs/chains/*.log ; squeue -u $USER
#
# Options:
#   --queue <file>          default experiments/queue.txt
#   --max-concurrent <N>    default 4 (concurrent runs = GPUs used)
#   --partition <name>      default rtx6000
#   --nodelist <spec>       default: chain_jobs.sh's per-partition default
#   --max-jobs <N>          per-run chain budget, default 10 (= 10×4h slices)
#   --dry-run               print the plan and exit — LAUNCHES NOTHING

set -uo pipefail

QUEUE_FILE="experiments/queue.txt"
MAX_CONCURRENT=4
PARTITION="rtx6000"
NODELIST=""
MAX_JOBS=10
DRY_RUN=0
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue)          QUEUE_FILE="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --partition)      PARTITION="$2"; shift 2 ;;
        --nodelist)       NODELIST="$2"; shift 2 ;;
        --max-jobs)       MAX_JOBS="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"
mkdir -p logs/chains

if [[ ! -f "$QUEUE_FILE" ]]; then
    echo "ERROR: queue file not found: $QUEUE_FILE" >&2
    exit 1
fi

# ---- parse queue.txt: "<config> <seed1> [<seed2> ...]" -------------------
RUNS=()  # entries: "config|seed"
while IFS= read -r line; do
    line="${line%%#*}"                      # strip comments
    line="$(echo "$line" | xargs 2>/dev/null || true)"  # trim
    [[ -z "$line" ]] && continue
    read -r config seeds <<<"$line"
    if [[ ! -f "$config" ]]; then
        echo "ERROR: config listed in queue does not exist: $config" >&2
        exit 1
    fi
    for seed in $seeds; do
        RUNS+=("${config}|${seed}")
    done
done < "$QUEUE_FILE"

if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "Queue is empty (all lines commented?) — nothing to do."
    exit 0
fi

echo "==================================================================="
echo "Queue launcher — $(date -Iseconds)"
echo "  Queue file:      $QUEUE_FILE"
echo "  Runs:            ${#RUNS[@]}"
echo "  Max concurrent:  $MAX_CONCURRENT (= max GPUs used at once)"
echo "  Partition:       $PARTITION"
echo "  Nodelist:        ${NODELIST:-<per-partition default>}"
echo "  Chain budget:    $MAX_JOBS jobs/run"
echo "==================================================================="
for r in "${RUNS[@]}"; do
    echo "  - config=${r%%|*} seed=${r##*|}"
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo "--dry-run: nothing launched."
    exit 0
fi

# ---- run with a concurrency cap ------------------------------------------
declare -A CHAIN_PID_TO_NAME=()
FAILED=0

launch_one() {
    local config="$1" seed="$2"
    local run_name
    run_name="$(basename "$config" .yaml)_seed${seed}"
    local log="logs/chains/${run_name}.log"
    local args=(--config "$config" --seed "$seed" --max-jobs "$MAX_JOBS" --partition "$PARTITION")
    if [[ -n "$NODELIST" ]]; then
        args+=(--nodelist "$NODELIST")
    fi
    echo "[queue] starting chain: $run_name (log: $log)"
    bash scripts/slurm/chain_jobs.sh "${args[@]}" >> "$log" 2>&1 &
    CHAIN_PID_TO_NAME[$!]="$run_name"
}

reap_one() {
    # Wait for ANY chain to finish; report its status.
    # Portable (bash 4+): poll the tracked pids instead of `wait -n -p`
    # (which needs bash >= 5.1 — not guaranteed on the frontale).
    while true; do
        local pid
        for pid in "${!CHAIN_PID_TO_NAME[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                local rc=0
                wait "$pid" || rc=$?
                local name="${CHAIN_PID_TO_NAME[$pid]:-unknown}"
                unset "CHAIN_PID_TO_NAME[$pid]"
                if [[ $rc -eq 0 ]]; then
                    echo "[queue] DONE: $name"
                else
                    echo "[queue] FAILED (rc=$rc): $name — see logs/chains/${name}.log" >&2
                    FAILED=$((FAILED + 1))
                fi
                return
            fi
        done
        sleep 30
    done
}

for r in "${RUNS[@]}"; do
    while (( ${#CHAIN_PID_TO_NAME[@]} >= MAX_CONCURRENT )); do
        reap_one
    done
    launch_one "${r%%|*}" "${r##*|}"
done
while (( ${#CHAIN_PID_TO_NAME[@]} > 0 )); do
    reap_one
done

echo "==================================================================="
echo "Queue finished at $(date -Iseconds) — failures: $FAILED"
echo "==================================================================="
exit $(( FAILED > 0 ? 1 : 0 ))
