# Experiments

This directory tracks experiment **metadata** (configs, results JSON, figures,
analysis reports). It does **not** contain trained checkpoints.

## Where the actual artifacts live

| Artifact | Location | Reason |
|---|---|---|
| Trained checkpoints (`expert_pools.pt`, `optimizer.pt`) | **Hugging Face Hub** — `Helain/gated-lora-experiments` (private dataset) | Too large for git; HF Hub is built for ML weights |
| Wandb runs | `wandb.ai` (offline mode on Ensimag, synced via VPS) | n/a |
| Raw SLURM logs | Ensimag `~/logs/` (transient, not backed up) | Cleaned up by the supervisor pipeline |
| Legacy `ensicompute_*` checkpoints | External SSD (D:) at `D:\ensimag_backup\GatedLoraProject\ensicompute\` | Pre-paper backup, ~106 GB |

## Layout

```
experiments/
├── README.md                       # this file
├── legacy/                         # READMEs and analysis reports from the
│                                   # 11 ensicompute_* folders, kept for
│                                   # context but not re-runnable as-is
└── <experiment_name>_seed<N>/      # Per-run output (created by training)
    ├── experiment_config.json      # Resolved config snapshot
    ├── final_results.json          # Loss + metrics summary
    ├── routing_history.json        # Per-step routing snapshots (gated only)
    ├── visualizations/             # Routing analysis figures
    └── ...
```

## Repro instructions for legacy results

The `legacy/` reports reference experiments run before this refactor. To
reproduce one:

1. Pick the closest matching YAML in `configs/experiments/`
   (e.g. `phi2_harder_multitask.yaml` ≈ legacy `ensicompute_harder_multitask`).
2. Run:
   ```bash
   bash scripts/slurm/chain_jobs.sh \
       --config configs/experiments/phi2_harder_multitask.yaml \
       --seed 42
   ```
3. Compare the resulting `final_results.json` against
   `legacy/<closest match>/analysis_results/`.

Some legacy diffs (notably `Llama3.2_modified` with custom SLURM chaining,
`ensicompute_per_layer_multirun` with routing snapshots) have been **merged
into the unified trainer** — these are no longer separate experiments,
just config flags (`max_runtime_seconds`, `enable_routing_analysis`).

## Pipeline summary

```
Ensimag (10 GB cap)        VPS OVH staging        Hugging Face Hub
─────────────────────      ──────────────         ──────────────────
SLURM trains          ──→  rsync from Ensimag ──→ huggingface_hub.upload
  ↓                        ↓                       (private dataset)
  TRAINING_DONE marker     verify integrity        ↓
  ↓                        delete from Ensimag     synced; published when paper-ready
  4h chain auto-resumes    via cron (5min)
```

See [`scripts/transfer/`](../scripts/transfer/) (Phase K, scheduled next)
for the actual transfer pipeline implementation.
