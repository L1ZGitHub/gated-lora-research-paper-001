#!/bin/bash
# Submit SLURM jobs in a loop until TRAINING_DONE appears on HF Hub.
#
# State of truth: Hugging Face Hub (Helain/gated-lora-experiments).
# Login node has no visibility into compute /tmp, so we poll HF instead.
#
# Usage:
#   bash scripts/slurm/chain_jobs.sh \
#       --config configs/experiments/phi2_harder_multitask.yaml \
#       --seed 42 \
#       [--max-jobs 10] \
#       [--partition rtx6000] \
#       [--run-name <override>]
#
# Exit codes: 0 done, 1 bad args, 2 budget exhausted.

set -euo pipefail

CONFIG=""
SEED=""
MAX_JOBS=10
PARTITION="rtx6000"
RUN_NAME=""
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_REPO="${GLR_HF_REPO:-Helain/gated-lora-experiments}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --max-jobs)   MAX_JOBS="$2"; shift 2 ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --run-name)   RUN_NAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$CONFIG" ]] && { echo "ERROR: --config required" >&2; exit 1; }
[[ -z "$SEED"   ]] && { echo "ERROR: --seed required"   >&2; exit 1; }

# Load HF_TOKEN from .env at repo root
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set (looked in $PROJECT_ROOT/.env)" >&2
    exit 1
fi

# Default run name
if [[ -z "$RUN_NAME" ]]; then
    config_basename=$(basename "$CONFIG" .yaml)
    RUN_NAME="${config_basename}_seed${SEED}"
fi

# Pick the right Python — prefer the project venv if it exists, else system.
PY=""
for candidate in "$PROJECT_ROOT/.venv/bin/python" "$(command -v python3 || true)"; do
    if [[ -x "$candidate" ]]; then
        if "$candidate" -c "import huggingface_hub" 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done
if [[ -z "$PY" ]]; then
    echo "ERROR: no Python with huggingface_hub found." >&2
    echo "  Try: cd $PROJECT_ROOT && uv sync --no-dev" >&2
    exit 1
fi

echo "==================================================================="
echo "Chain orchestrator"
echo "  Config:    $CONFIG"
echo "  Seed:      $SEED"
echo "  Run name:  $RUN_NAME"
echo "  HF repo:   $HF_REPO"
echo "  Partition: $PARTITION"
echo "  Max jobs:  $MAX_JOBS"
echo "  Python:    $PY"
echo "==================================================================="

# Check HF Hub for TRAINING_DONE on this run.
check_done() {
    HF_TOKEN="$HF_TOKEN" "$PY" - <<EOF
import os, sys
from huggingface_hub import HfApi, login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
try:
    files = HfApi().list_repo_files(repo_id="$HF_REPO", repo_type="dataset")
    sys.exit(0 if any(f == "$RUN_NAME/TRAINING_DONE" for f in files) else 1)
except Exception as e:
    print(f"  HF check failed: {e}", file=sys.stderr)
    sys.exit(2)  # treat as "unknown" — let the loop retry next iteration
EOF
}

for ((i=1; i<=MAX_JOBS; i++)); do
    if check_done; then
        echo "[chain] HF Hub has $RUN_NAME/TRAINING_DONE — done after $((i-1)) job(s)"
        exit 0
    fi

    echo "[chain] Submitting job $i / $MAX_JOBS at $(date -Iseconds)"
    # Partition-specific nodelist:
    #   rtx6000 = restrict to 6 of 9 turing nodes (per agreement with the user)
    #   a40     = only the "ampere" node exists, let SLURM pick automatically
    NODELIST_ARG=()
    case "$PARTITION" in
        rtx6000) NODELIST_ARG=(--nodelist=turing-[4-9]) ;;
    esac
    job_output=$(sbatch \
        --partition="$PARTITION" \
        "${NODELIST_ARG[@]}" \
        --export=ALL,GLR_CONFIG="$CONFIG",GLR_SEED="$SEED",GLR_RUN_NAME="$RUN_NAME",HF_TOKEN="$HF_TOKEN",GLR_HF_REPO="$HF_REPO",GLR_RESUME="auto" \
        "${PROJECT_ROOT}/scripts/slurm/train.sbatch")

    job_id=$(echo "$job_output" | grep -oP '\d+$')
    if [[ -z "$job_id" ]]; then
        echo "[chain] ERROR: could not parse job id from: $job_output" >&2
        exit 1
    fi
    echo "[chain] Submitted job $job_id, polling..."

    while squeue -j "$job_id" -h 2>/dev/null | grep -q "^"; do
        sleep 60
    done
    echo "[chain] Job $job_id left the queue at $(date -Iseconds)"
done

if check_done; then
    echo "[chain] Final TRAINING_DONE check OK — success"
    exit 0
fi
echo "[chain] ERROR: budget exhausted ($MAX_JOBS jobs) without TRAINING_DONE" >&2
exit 2
