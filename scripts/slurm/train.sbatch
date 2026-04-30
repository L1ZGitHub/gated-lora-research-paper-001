#!/bin/bash
# SLURM training template — parameterized via env vars (set by chain_jobs.sh).
#
# Required env vars:
#   GLR_CONFIG       — path to experiment YAML (e.g. configs/experiments/phi2_harder_multitask.yaml)
#   GLR_SEED         — int, training seed
#   GLR_OUTPUT_DIR   — output dir (overrides config's output.output_dir)
#   GLR_PROJECT_ROOT — absolute path to repo root (where pyproject.toml lives)
#
# Optional:
#   GLR_RESUME       — "auto" (default) or explicit checkpoint path
#   GLR_HF_TOKEN     — HuggingFace token for gated models
#   GLR_WANDB_MODE   — "online" / "offline" / "disabled" (default offline on Ensimag)
#
#SBATCH --job-name=glr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --nodelist=turing-[4-9]   # 6 of 9 turing nodes (rtx6000 partition)

set -euo pipefail

# --- Sanity checks ---
: "${GLR_CONFIG:?must be set}"
: "${GLR_SEED:?must be set}"
: "${GLR_OUTPUT_DIR:?must be set}"
: "${GLR_PROJECT_ROOT:?must be set}"

cd "$GLR_PROJECT_ROOT"

mkdir -p logs "$GLR_OUTPUT_DIR"

echo "==================================================================="
echo "SLURM job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Config:    $GLR_CONFIG"
echo "Seed:      $GLR_SEED"
echo "Output:    $GLR_OUTPUT_DIR"
echo "Resume:    ${GLR_RESUME:-auto}"
echo "Started:   $(date -Iseconds)"
echo "==================================================================="

# --- HF auth ---
if [[ -n "${GLR_HF_TOKEN:-}" ]]; then
    echo "HF token present — logging in"
    huggingface-cli login --token "$GLR_HF_TOKEN" --add-to-git-credential 2>/dev/null \
        || python -c "from huggingface_hub import login; login(token='$GLR_HF_TOKEN')"
fi

# --- Wandb mode (offline by default on compute nodes — no internet) ---
export WANDB_MODE="${GLR_WANDB_MODE:-offline}"

# --- Run training ---
uv run python -m gated_lora.cli \
    --config "$GLR_CONFIG" \
    --seed "$GLR_SEED" \
    --output-dir "$GLR_OUTPUT_DIR" \
    --resume "${GLR_RESUME:-auto}"

EXIT_CODE=$?
echo "==================================================================="
echo "Job ${SLURM_JOB_ID:-local} exited with code $EXIT_CODE at $(date -Iseconds)"
echo "==================================================================="
exit $EXIT_CODE
