#!/bin/bash
# Orchestrate SLURM job chaining until TRAINING_DONE marker appears.
#
# Usage:
#   bash scripts/slurm/chain_jobs.sh \
#       --config configs/experiments/phi2_harder_multitask.yaml \
#       --seed 42 \
#       [--max-jobs 10] \
#       [--partition rtx6000] \
#       [--output-dir ./outputs/phi2_harder_multitask_seed42]
#
# Behavior:
#   1. Submit job N via sbatch (uses train.sbatch).
#   2. Wait for job to complete (polls squeue every 60s).
#   3. Check $output_dir/TRAINING_DONE.
#      - If exists → success, exit 0.
#      - If absent and N < max_jobs → resubmit with same args (resume=auto).
#      - If N == max_jobs → exit with error code 2 (didn't converge in budget).
#
# Designed to be invoked from VPS via SSH or interactively on Ensimag login node.

set -euo pipefail

CONFIG=""
SEED=""
MAX_JOBS=10
PARTITION="rtx6000"
OUTPUT_DIR=""
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --max-jobs)    MAX_JOBS="$2"; shift 2 ;;
        --partition)   PARTITION="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$CONFIG" ]] && { echo "ERROR: --config required" >&2; exit 1; }
[[ -z "$SEED"   ]] && { echo "ERROR: --seed required"   >&2; exit 1; }

# Default output dir from config name + seed
if [[ -z "$OUTPUT_DIR" ]]; then
    config_basename=$(basename "$CONFIG" .yaml)
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/${config_basename}_seed${SEED}"
fi

DONE_MARKER="${OUTPUT_DIR}/TRAINING_DONE"

echo "==================================================================="
echo "Chain orchestrator"
echo "  Config:    $CONFIG"
echo "  Seed:      $SEED"
echo "  Output:    $OUTPUT_DIR"
echo "  Partition: $PARTITION"
echo "  Max jobs:  $MAX_JOBS"
echo "  Marker:    $DONE_MARKER"
echo "==================================================================="

mkdir -p "$OUTPUT_DIR" "${PROJECT_ROOT}/logs"

for ((i=1; i<=MAX_JOBS; i++)); do
    if [[ -f "$DONE_MARKER" ]]; then
        echo "[chain] TRAINING_DONE found — exiting after $((i-1)) job(s)"
        exit 0
    fi

    echo "[chain] Submitting job $i / $MAX_JOBS at $(date -Iseconds)"

    # Submit and capture job id
    job_output=$(sbatch \
        --partition="$PARTITION" \
        --export=ALL,GLR_CONFIG="$CONFIG",GLR_SEED="$SEED",GLR_OUTPUT_DIR="$OUTPUT_DIR",GLR_PROJECT_ROOT="$PROJECT_ROOT",GLR_RESUME="auto" \
        "${PROJECT_ROOT}/scripts/slurm/train.sbatch")

    job_id=$(echo "$job_output" | grep -oP '\d+$')
    if [[ -z "$job_id" ]]; then
        echo "[chain] ERROR: could not parse job id from sbatch output:" >&2
        echo "$job_output" >&2
        exit 1
    fi

    echo "[chain] Submitted job $job_id, polling..."

    # Poll squeue until job leaves the queue
    while squeue -j "$job_id" -h 2>/dev/null | grep -q "^"; do
        sleep 60
    done

    echo "[chain] Job $job_id finished at $(date -Iseconds)"
done

if [[ -f "$DONE_MARKER" ]]; then
    echo "[chain] TRAINING_DONE found in final iteration — success"
    exit 0
fi

echo "[chain] ERROR: hit max-jobs limit ($MAX_JOBS) without TRAINING_DONE marker" >&2
exit 2
