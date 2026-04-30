# gated-lora-research-paper-001

Unified codebase for the **Gated LoRA** paper: per-layer expert routing for parameter-efficient
multi-task adaptation. Refactored from 11 fragmented experiment folders into a single, clean,
config-driven pipeline.

## Status

> **Work in progress.** Repo scaffolded 2026-04-30 from 11 fragmented
> `ensicompute_*` folders. See [`experiments/legacy/merge_notes.md`](experiments/legacy/merge_notes.md)
> for the refactor plan and provenance of each module.

Phases A–I complete: scaffold, stable code, unified `LoRAExpert` /
`GatedLoRAModelV2` / `GatedLoRATrainer`, YAML-driven configs, SLURM chain
templates, CLI entrypoint, and 36 passing smoke tests.

## Supported models

- Phi-2 (2.7B) — baseline, no HF auth required
- Gemma-2-2B — gated, requires HF auth
- Llama-3.2-3B — gated, requires HF auth
- Pythia-410M — EleutherAI, GPT-NeoX architecture
- Qwen2.5-0.5B — small, fast
- SmolLM-360M — smallest

All architectures share a single `LoRAExpert` / `GatedLoRAModelV2` implementation
with automatic GQA detection and architecture-specific hooks.

## Repo layout

```
src/gated_lora/         # Single, unified package (no more 11 copies of src/)
  models/               # Base model + Gated LoRA (V1 legacy + V2)
  data/                 # Multi-task dataset (YAML-driven)
  training/             # Trainer + Gated trainer (with SLURM chaining)
  analysis/             # Routing analysis + snapshots
configs/
  models/               # Per-model hyperparams (batch_size, grad_accum, ...)
  tasks/                # original_4, harder_4, all_8, ...
  ablations/            # no_layer_embedding, partial_gating, per_layer_only
  experiments/          # Combinations: phi2_harder.yaml, gemma2_all8.yaml, ...
scripts/
  slurm/                # Parameterized SLURM templates (4h-aware chaining)
  transfer/             # Ensimag → VPS → HF Hub pipeline
  supervisor/           # Optional Claude-based anomaly handler
experiments/            # Tracked metrics/figures (no checkpoints)
paper/                  # LaTeX sources + figures
tests/                  # Smoke tests
```

## Quickstart

```bash
# Setup (one-time)
uv sync --extra dev

# Validate a config without launching training
uv run python -m gated_lora.cli \
    --config configs/experiments/phi2_harder_multitask.yaml \
    --seed 42 --dry-run

# Run a single training (locally, requires GPU)
uv run python -m gated_lora.cli \
    --config configs/experiments/phi2_harder_multitask.yaml \
    --seed 42

# Run on Ensimag SLURM with auto-chaining (handles 4h time limit)
bash scripts/slurm/chain_jobs.sh \
    --config configs/experiments/phi2_harder_multitask.yaml \
    --seed 42 --partition rtx6000

# Multi-seed launch (3 seeds × auto-chain each)
./scripts/launch_experiment.sh configs/experiments/phi2_harder_multitask.yaml

# Run tests
uv run pytest tests/ -v
```

## Storage / SLURM constraints

- **Ensimag home cap**: 10 GB. Pipeline cleans up checkpoints to HF Hub continuously.
- **SLURM time limit**: 4h max on all partitions → trainer auto-saves + resumes.
- **Reserved nodes**: 6 of 9 turing-* nodes (`scripts/slurm/train.sbatch` uses `--exclude=turing-2,turing-3,turing-9`).

## Reproducibility

Every run is fully specified by a single YAML config + a seed. To reproduce a published result:

```bash
uv run python -m gated_lora.cli --config configs/experiments/<paper_table_X>.yaml --seed <reported>
```

## Memory of the past

Eleven legacy `ensicompute_*` folders (Phi-2 baseline, model variants, ablations, per-layer studies)
are archived in `experiments/legacy/` (READMEs + analysis reports only — checkpoints live on HF Hub).
The actual research history (60+ runs) lives at:
**`huggingface.co/Helain/gated-lora-experiments`** (private dataset).
