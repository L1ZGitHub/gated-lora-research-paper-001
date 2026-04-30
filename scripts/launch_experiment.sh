#!/bin/bash
# User-friendly wrapper to launch one experiment over multiple seeds.
#
# Usage:
#   ./scripts/launch_experiment.sh configs/experiments/phi2_harder_multitask.yaml [seed1 seed2 ...]
#
# If no seeds given, defaults to: 42 1337 2024 (3 seeds for paper).

set -euo pipefail

CONFIG="${1:-}"
shift || true
SEEDS=("$@")
[[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(42 1337 2024)

if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 <config.yaml> [seed1 seed2 ...]" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARTITION="${SLURM_PARTITION:-rtx6000}"

echo "Launching experiment $CONFIG over ${#SEEDS[@]} seed(s) on partition $PARTITION"
echo "Seeds: ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Seed $seed"
    bash "${PROJECT_ROOT}/scripts/slurm/chain_jobs.sh" \
        --config "$CONFIG" \
        --seed "$seed" \
        --partition "$PARTITION" \
        &
done

echo ""
echo "All ${#SEEDS[@]} chains launched in background."
echo "Wait for them with: wait"
echo "Or monitor: squeue -u \$USER"
