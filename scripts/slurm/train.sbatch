#!/bin/bash
# SLURM training template — parameterized via env vars (set by chain_jobs.sh).
#
# Required env vars:
#   GLR_CONFIG       — path to experiment YAML (e.g. configs/experiments/phi2_harder_multitask.yaml)
#   GLR_SEED         — int, training seed
#   GLR_OUTPUT_DIR   — output dir (overrides config's output.output_dir)
#   GLR_PROJECT_ROOT — absolute path to repo root (where pyproject.toml lives)
#
# Auth:
#   HF_TOKEN is loaded from $GLR_PROJECT_ROOT/.env at the start of every
#   job (compute nodes don't persist auth state between SLURM allocations).
#   Used both for downloading gated models AND for the trainer's post-save
#   HF Hub push (gated_lora.training.gated_trainer.save_checkpoint).
#
# Optional:
#   GLR_RESUME       — "auto" (default) or explicit checkpoint path
#   GLR_WANDB_MODE   — "online" / "offline" / "disabled" (default offline on Ensimag)
#
#SBATCH --job-name=glr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --nodelist=turing-[4-9]   # restrict to 6 of 9 turing nodes (rtx6000)

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

# --- Re-auth on every job (compute nodes don't keep state) ---
if [[ -f "${GLR_PROJECT_ROOT}/.env" ]]; then
    set -a; source "${GLR_PROJECT_ROOT}/.env"; set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set — gated model downloads + HF Hub push will fail" >&2
else
    echo "HF token loaded — logging in"
    # Programmatic login is the canonical path; CLI is a no-op fallback for visibility
    .venv/bin/python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
fi

# --- Wandb mode (offline by default on compute nodes — no internet) ---
export WANDB_MODE="${GLR_WANDB_MODE:-offline}"

# --- Run training ---
.venv/bin/python -m gated_lora.cli \
    --config "$GLR_CONFIG" \
    --seed "$GLR_SEED" \
    --output-dir "$GLR_OUTPUT_DIR" \
    --resume "${GLR_RESUME:-auto}"

EXIT_CODE=$?
echo "==================================================================="
echo "Job ${SLURM_JOB_ID:-local} exited with code $EXIT_CODE at $(date -Iseconds)"
echo "==================================================================="
exit $EXIT_CODE
