# SLURM scripts

## Files

| Script | Purpose |
|--------|---------|
| `train.sbatch` | Single-job training template. Receives config/seed/output_dir via env vars. 4h time limit. Pinned to `turing-[4-9]` (6 of 9 turing nodes). |
| `chain_jobs.sh` | Loops `sbatch train.sbatch` until `TRAINING_DONE` marker appears. Auto-resumes via the SLURM-aware trainer. |
| `../launch_experiment.sh` | High-level wrapper. Launches one experiment across multiple seeds in parallel. |

## Constraints baked in

- **4h time limit** per job (max on all Ensimag partitions)
- **6 of 9 turing nodes** used (`turing-[4-9]`) — leaves 3 for other users
- **Auto-resume** via `--resume auto` (trainer's `find_latest_checkpoint()` does the discovery)
- **TRAINING_DONE marker** signals "no more jobs needed" to chain orchestrator
- **48 GB RAM** per job (set in `#SBATCH --mem=48GB`)

## Usage examples

### One experiment, default seeds (42, 1337, 2024)
```bash
./scripts/launch_experiment.sh configs/experiments/phi2_harder_multitask.yaml
```

### One experiment, custom seeds
```bash
./scripts/launch_experiment.sh configs/experiments/gemma2_all_8.yaml 7 13 21
```

### Direct chain invocation (one seed)
```bash
bash scripts/slurm/chain_jobs.sh \
    --config configs/experiments/phi2_harder_multitask.yaml \
    --seed 42 \
    --partition rtx6000
```

### Switch partition (use a40 for big models)
```bash
SLURM_PARTITION=a40 ./scripts/launch_experiment.sh configs/experiments/llama32.yaml
```

## Required env vars (set automatically by `launch_experiment.sh` / `chain_jobs.sh`)

- `GLR_CONFIG` — path to YAML
- `GLR_SEED` — int
- `GLR_OUTPUT_DIR` — absolute path
- `GLR_PROJECT_ROOT` — absolute path to repo root
- `GLR_RESUME` — `auto` (default), or explicit checkpoint path
- `GLR_HF_TOKEN` — required for gated models (Llama, Gemma)
- `GLR_WANDB_MODE` — defaults to `offline` (no internet on compute nodes)

## Disk hygiene

The trainer rotates `optimizer.pt` (only the LATEST checkpoint keeps it).
Earlier checkpoints keep `expert_pools.pt` only — these are pulled by the
transfer pipeline (Phase K) and uploaded to Hugging Face Hub.

Target: **less than 3 GB total** on Ensimag at any moment, even with
6 concurrent runs.
