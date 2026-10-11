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

**Phase A' amendment (2026-10-10, after the 3-point results, before any wider run).** Three of the
four argmins were at a grid edge (gated and r66 at 2.5e-5, r16 at 1e-4 by 2e-5), so instead of
one edge point per arm the grid is widened once for every arm to {1e-6, 3e-6, 1e-5, 2.5e-5, 5e-5,
1e-4, 2.5e-4, 5e-4, 1e-3} (`queue_v3_phase_a_wide.txt`). Each arm's LR = argmin of its full-val
macro loss (last checkpoint) over these 9 points. The grid is not extended again: an argmin at
1e-6 or 1e-3 is reported as such.

**Phase B'.** Seeds 1-4 for gated, r66, r32, r16 at their LR (seed 0 = the Phase A' run);
r56 seeds 0-4; frozen_gate and equal_rank seeds 0-2.

**Q1 test.** Per-seed macro test loss, gated vs each plain LoRA (r66, r56, r32, r16): Welch
t-test, two-sided, 5 vs 5, Holm correction over the 4 comparisons; differences reported with
95% CIs. "Gated is better than X" needs Holm-adjusted p < 0.05 AND a lower mean; otherwise it is
reported as no detectable difference (with the CI), never as a win.

**Q2 analyses** (unchanged from v2, fixed in advance): eta^2 of routing by task per depth
block, map correlation across seeds vs frozen_gate/equal_rank, expert knockout on test with
usage-damage Spearman, per-task expert removal chosen on val and scored on test.

**Amendment (2026-10-11, after the Phase A' grid, while Phase B' trains, before any Phase B'
result is looked at).** From a code review of the pipeline:

- *Q1 equivalence.* A non-significant Welch test is not evidence of "as good as". Added, per
  comparison: TOST with margin delta = 0.005 nats (macro test loss; ~3 seed s.d. in v2, 0.25%
  of the fine-tuning gain 2.80 -> 0.78), i.e. two one-sided Welch tests, p_TOST = the larger
  of the two; Holm over the 4 comparisons, separately from the superiority test. Verdicts:
  "gated better" (superiority, as above), "equivalent within 0.005" (Holm-adjusted p_TOST <
  0.05, i.e. the 90% CI of gated - X inside +-0.005), otherwise "inconclusive". The 95% CIs
  reported are unadjusted. Inference is over training seeds; the test subset is fixed.
- *Routing measured at the predicting positions.* `final_examples.npz` averaged gate weights
  over the answer-token positions themselves (incl. the EOS position, which predicts nothing),
  not over the positions that predict answer tokens (`answer_mask[:, 1:]`). Fixed in the
  trainer; every routing-bearing run used by Q2 (gated, equal_rank, frozen_gate, all seeds) is
  re-evaluated on test from its saved final model with the fixed code
  (`knockout_v2.py --conditions none --dump-dir`), so all Q2 numbers use one definition.
- *Map reproducibility.* The correlation of flattened task x layer E[rank] maps is inflated by
  any depth profile shared across seeds, and for equal_rank (19/19/18) E[rank] = 19 - w_3
  tracks one arbitrarily labelled expert. Replaced by: per layer, the task x expert gate-share
  map centred over tasks; for equal_rank the experts of each layer are matched across the two
  seeds by the permutation that maximises that layer's correlation (conservative: it can only
  help the control); identity for mixed ranks, whose experts are identified by their rank;
  then the correlation over the concatenated layers; mean over seed pairs.
- *Frozen gate.* Its target 0.723 was measured on test inputs (no labels) of gated's seed-0
  run; the achieved final top-1 of each frozen_gate run is reported next to it.
- *Checkpoint selection* (secondary metric only) uses the first 250 val examples per task
  every 1000 steps, not the full val set.

**Phase C' (pre-registered 2026-10-11, after the Phase B' results, before any Phase C' run):
small ranks.** Phase B' found test loss flat from r16 to r66 and a router that puts 71% of its
weight on the r32 expert: adapter capacity was not scarce, so gating had nothing to allocate.
Phase C' repeats the comparison where capacity may be scarce. Same model, data, splits, steps
and metric as above; only ranks and learning rates change.

- *Arms.* Plain LoRA r2, r4, r8, r14 (alpha = 2r) and gated 2/4/8 (alphas 4/8/16, total rank
  14, gate network unchanged). r14 is gated 2/4/8 with routing off. The gate network is ~5.5M
  parameters (gated 8/16/32 36.3M vs r56 30.8M), so gated 2/4/8 has ~13.2M trainable parameters
  against ~7.7M for r14; the equal-parameter references are the Phase B' arms r16 (8.8M, fewer)
  and r32 (17.6M, more), no new arm. Exact counts are taken from the training logs.
- *LR grid (seed 0, `queue_v3_phase_c_grid.txt`).* {1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3,
  2.5e-3, 5e-3} for every arm, shifted up from the Phase A' grid because the best LR rose as
  the rank fell (r66 2.5e-5, r32 5e-5, r16 1e-4). Each arm's LR = argmin of its full-val macro
  loss (last checkpoint); a run that diverges (non-finite loss) counts as worse than any finite
  one. Every arm gets its own LR, r14 included. The grid is not extended: an argmin at an edge
  is reported as such.
- *Seeds.* Seeds 1-4 at each arm's LR (seed 0 = the grid run), 5 per arm.
- *Is capacity scarce?* r2, r4, r8, r14 vs the Phase B' r16 runs: Welch, two-sided, Holm over
  the 4 comparisons. "Capacity-bound at rank X" needs Holm-adjusted p < 0.05 AND a mean test
  loss more than 0.005 above r16. If no rank qualifies, the regime was not reached and Phase C'
  is reported as that, without reading the gated comparison as a test of gating under scarcity.
- *Q1 (small).* Gated 2/4/8 vs r14, r8, r4, r2: the Q1 tests above (Welch + TOST with delta =
  0.005, Holm over the 4 comparisons, same verdicts). Secondary, reported with CIs, not tested:
  gated 2/4/8 vs r16 and r32 (equal-parameter references).
- *Q2 (small).* The Q2 analyses above on gated 2/4/8 (gate shares per task and layer block,
  eta^2, map reproducibility across seeds, expert knockout with usage-damage Spearman, removal
  of the two smaller experts from all layers), set next to gated 8/16/32. No frozen-gate or
  equal-rank control is trained for the small mix: its numbers are read against the Phase B'
  controls, which is a limitation.
- *Code.* Phase C' trains with the trainer of commit 00782ed; that fix changed the routing dump
  only, not training, so Phase B' and Phase C' runs are comparable and Phase C' dumps need no
  re-evaluation.
