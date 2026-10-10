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

## v3 protocol (pre-registered 2026-10-10, before any v3 run)

**Why v3.** v2 split the val set row by row from the official train split. For WikiText that
put every val paragraph in an article that also had training paragraphs (630 articles, 100%
shared), so val rewarded memorising training documents: on WikiText val ranked gated best and
r16 worst, test ranked them the other way. WikiText's loss (~2.7) also dominated the plain macro
mean (other tasks 0.03-0.6), so the v2 LR choice and checkpoint selection were biased and the
"val keeps improving with adapter size" reading was a WikiText artefact. Without WikiText, val
and test agreed. v2 results are kept as preliminary and are not pooled with v3.

**Splits (`data_format: v3`).** WikiText-2, CoNLL-2003, XSum: official validation = val,
official test = test, train = the whole official train split. Other tasks unchanged from v2
(SQuAD/MNLI group-level train slice; GSM8K, CommonsenseQA, IMDB row-level: no measurable
overlap). Checked on the real data by `scripts/analysis/check_splits.py`.

**Metric.** Primary: macro test loss = plain mean over the 8 tasks of the answer loss (nats per
answer token, every task weighted equally, as in v2), on the LAST checkpoint; always shown with
the per-task table. LR choice and checkpoint selection use the same mean on val. Secondary:
per-task test losses, generation (GSM8K accuracy, XSum ROUGE-L), test at the best-val
checkpoint. Robustness check only: mean of loss / base-model loss on the same split
(`experiments/v3_base_reference.json`, from `scripts/analysis/base_reference.py`). It is not
primary because the base loss mostly measures answer-format familiarity (MNLI 8.6, SQuAD 0.68),
so its weights (1/base) are arbitrary; on the v2 test data it moves the ranking toward small
ranks (r16 first, vs r32 first on the plain mean). If it disagrees with the primary metric,
the paper says so.

**Phase A' (LR, seed 0, `queue_v3_phase_a.txt`).** LR in {2.5e-5, 5e-5, 1e-4} for gated, r66,
r32, r16. Each arm's LR = argmin of its full-val macro loss (last checkpoint). If the
argmin is at an edge of the grid, one more point beyond it (1.25e-5 or 2e-4) is run before
Phase B'. r56, frozen_gate and equal_rank use gated's LR; frozen_gate_target_top1 = the mean
final top-1 of gated's Phase A' run at that LR.

**Phase B'.** Seeds 1-4 for gated, r66, r32, r16 at their LR (seed 0 = the Phase A' run);
r56 seeds 0-4; frozen_gate and equal_rank seeds 0-2.

**Q1 test.** Per-seed macro test loss, gated vs each plain LoRA (r66, r56, r32, r16): Welch
t-test, two-sided, 5 vs 5, Holm correction over the 4 comparisons; differences reported with
95% CIs. "Gated is better than X" needs Holm-adjusted p < 0.05 AND a lower mean; otherwise it is
reported as no detectable difference (with the CI), never as a win.

**Q2 analyses** (unchanged from v2, fixed in advance): eta^2 of routing by task per depth
block, map correlation across seeds vs frozen_gate/equal_rank, expert knockout on test with
usage-damage Spearman, per-task expert removal chosen on val and scored on test.
