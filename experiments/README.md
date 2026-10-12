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
- *LR decision (2026-10-11, grid complete, 45 runs).* Full-val macro loss, seed 0:

  | LR | r2 | r4 | r8 | r14 | gated 2/4/8 |
  |---|---|---|---|---|---|
  | 1e-5 | 0.8298 | 0.8187 | 0.8074 | 0.7985 | 0.7909 |
  | 2.5e-5 | 0.8050 | 0.7958 | 0.7871 | 0.7815 | 0.7773 |
  | 5e-5 | 0.7898 | 0.7834 | 0.7775 | 0.7741 | 0.7738 |
  | 1e-4 | 0.7802 | 0.7758 | **0.7728** | **0.7725** | **0.7734** |
  | 2.5e-4 | **0.7756** | **0.7726** | 0.7774 | 0.7816 | 0.7809 |
  | 5e-4 | 0.7771 | 0.7783 | 0.7857 | 0.7922 | 0.7851 |
  | 1e-3 | 0.7840 | 0.7953 | 0.8035 | 0.8283 | 0.7870 |
  | 2.5e-3 | 0.8019 | 0.8257 | 0.9747 | diverged | diverged |
  | 5e-3 | diverged | diverged | diverged | diverged | diverged |

  Chosen: r2 2.5e-4, r4 2.5e-4, r8 1e-4, r14 1e-4, gated 2/4/8 1e-4; no argmin at a grid edge.
  The 7 diverged runs stopped on a non-finite loss (fp16) and are not retried. Seeds 1-4:
  `queue_v3_phase_c_seeds.txt`.
- *Incident.* The final Hub commit of `gated_r2r4r8_lr1em4_seed0` hung on a dead connection after
  training and evaluation had finished (TRAINING_DONE written on the node at 19:55). The same
  31 files were committed by hand from the node's output directory, then the job was cancelled.
  No result was recomputed.

**Phase D (pre-registered 2026-10-12, after the Phase B' and C' results, before any Phase D
run): second base model.**

Phase D asks whether the Qwen2.5-0.5B findings hold on another model. Same data, splits, steps,
batch, precision, gate network, auxiliary losses and metric; only the base model changes.

- *Model.* `HuggingFaceTB/SmolLM-360M` (32 layers, same seven target maps per layer). Chosen
  before any run because it trains with the unchanged code path. A 300-step smoke test (500
  examples per task, never pushed, not a result) confirmed that the gated and the plain model
  train in fp16 without overflow (1.42 and 1.06 s/step).
- *Arms.* Gated 8/16/32, gated 2/4/8, plain r56 and r14 (each gated model with routing off)
  and plain r2. No r66/r32/r16 and no frozen-gate or equal-rank control: a stated limitation.
- *LR grid (seed 0, `queue_v3_phase_d_grid.txt`).* {5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4,
  1e-3, 2.5e-3} for every arm (the union of the Qwen optima, 2.5e-5 to 2.5e-4, with at least two
  points on each side). Each arm's LR = argmin of its full-val macro loss (last checkpoint); a
  diverged run counts as worse than any finite one; the grid is not extended; an argmin at an
  edge is reported as such.
- *Seeds.* Seeds 1-4 at each arm's LR (seed 0 = the grid run), 5 per arm.
- *Q1 tests.* Macro test loss, last checkpoint. Gated 8/16/32 vs r56; gated 2/4/8 vs r14 and vs
  r2: Welch two-sided and TOST with the same margin delta = 0.005, Holm over the 3 comparisons.
  Rank: r2 and r14 vs r56 (Welch, Holm over 2); "capacity-bound" needs adjusted p < 0.05 AND a
  mean more than 0.005 above r56.
- *Q2 analyses* on both gated models, layer blocks = thirds (layers 0-9, 10-20, 21-31): gate
  shares per task and block, eta^2, seed-to-seed correlation of task-centred maps, expert
  knockouts (each expert x each block and all layers), usage-damage Spearman overall and per
  expert, removal of the two smaller experts, Spearman(task loss, expected rank). Also, now
  specified in advance (they were post hoc on Qwen): seed-to-seed correlation of the layer x
  expert depth profile, Spearman between the largest expert's share and the relative loss
  reduction from fine-tuning, and per-task removal chosen on val and scored on test (both gated
  models, r56 and r14).
- *What counts as replicated* (each reported as replicated or not; none redefined afterwards):
  - R1 accuracy: each gated model is equivalent to its routing-off twin (TOST, adjusted
    p < 0.05).
  - R2 rank: r2 is not capacity-bound relative to r56.
  - R3 dominant expert: in both mixes, removing the largest expert from all layers raises the
    macro test loss by at least 10 times the absolute change caused by removing either other
    expert (means over seeds).
  - R4 faithfulness: the usage-damage Spearman of the largest expert is positive on average
    and in at least 12 of the 15 seed x block cases, in both mixes.
  - R5 reproducibility: mean seed-to-seed correlation of the task-centred maps >= 0.5 in both
    mixes (Qwen: 0.71 and 0.61; equal-rank control 0.46; random gates 0.02).
  - R6 task ordering (2/4/8 only): Spearman(task loss, expected rank) negative in all five
    seeds with a mean <= -0.7 (Qwen: -0.90 to -0.95). No prediction for 8/16/32.
  - R7 removable small expert: removing the smallest expert from all layers changes the macro
    test loss by less than 0.005 in absolute value, in both mixes.
- *LR decision (2026-10-12, grid complete, 45 runs, none diverged).* Full-val macro loss, seed 0:

  | LR | r2 | r14 | r56 | gated 2/4/8 | gated 8/16/32 |
  |---|---|---|---|---|---|
  | 5e-6 | 1.2867 | 1.0391 | 0.9804 | 1.0214 | 0.9707 |
  | 1e-5 | 1.0745 | 0.9865 | 0.9365 | 0.9697 | 0.9226 |
  | 2.5e-5 | 0.9945 | 0.9270 | 0.8890 | 0.9085 | 0.8761 |
  | 5e-5 | 0.9476 | 0.8921 | 0.8637 | 0.8782 | 0.8524 |
  | 1e-4 | 0.9076 | 0.8662 | 0.8449 | 0.8559 | 0.8357 |
  | 2.5e-4 | 0.8723 | 0.8426 | 0.8214 | 0.8351 | **0.8143** |
  | 5e-4 | 0.8558 | 0.8261 | **0.8198** | **0.8199** | 0.8180 |
  | 1e-3 | 0.8432 | **0.8237** | 0.8375 | 0.8249 | 0.8241 |
  | 2.5e-3 | **0.8388** | 0.8489 | 0.9672 | 0.8396 | 0.8772 |

  Chosen: r2 2.5e-3, r14 1e-3, r56 5e-4, gated 2/4/8 5e-4, gated 8/16/32 2.5e-4. **The r2
  argmin is at the upper edge of the grid**; as pre-registered the grid is not extended, so r2
  may be under-tuned and any r2 deficit (criterion R2, and the gated 2/4/8 vs r2 comparison) is
  an upper bound on the effect of rank. The other four optima are interior. Seeds 1-4:
  `queue_v3_phase_d_seeds.txt`.
- *Incident.* Four grid runs were first placed on turing-5 and failed at start-up with "CUDA
  device busy or unavailable" before any training step; they were rerun on other nodes, nothing
  else changed. A later probe found the same error on GPU 1 of turing-6 and of turing-7 with
  the nodes otherwise idle, so it is a fault of specific GPUs; turing-5, 6 and 7 are excluded.
- *Status (2026-10-12, 09:15): Phase D interrupted, incomplete.* At 09:05 the cluster
  administrators cancelled every job, and the account's access to the platform was then
  withdrawn. No Phase D test or replication verdict has been computed.
  - Complete on the Hub (15 seed runs + 45 grid runs): gated 8/16/32, gated 2/4/8 and r56 with
    seeds 0-4; r14 with seeds 0-3; r2 with seed 0.
  - Interrupted, resumable from `latest/` on the Hub: `baseline_r14_lr1em3_seed4` (step 1946 of
    5000) and `baseline_r2_lr2p5em3_seed{1,2,3,4}` (steps 1985, 2035, 1980, 1760).
  - Not run: knockouts and per-task removal for seeds 1-4 (seed 0 done; its outputs, the
    routing analysis below and the grid summary are on the cluster, not in this repository;
    all can be regenerated from the Hub).
  - Routing analysis (`analyze_v2.py --version v3smollm --routing-only`, five seeds per gated
    model, final for these quantities): gate shares small/middle/large expert 0.04/0.05/0.91
    (8/16/32) and 0.03/0.07/0.90 (2/4/8); seed-to-seed correlation of task-centred maps 0.80
    (min 0.62) and 0.56 (min 0.44), so R5 holds; Spearman(task loss, expected rank) for 2/4/8
    -0.90, -0.95, -0.90, -0.90, -0.95, so R6 holds (8/16/32: -0.74 to -0.93); eta^2
    early/mid/late 0.50/0.66/0.69 and 0.47/0.64/0.55; depth-profile reproducibility 1.00 and
    0.95; Spearman(largest-expert share, relative loss reduction) +0.67 to +0.88 and +0.86 to
    +0.95. Not replicated, although not a criterion: the more even spread of the Qwen 2/4/8
    router (0.45 on its largest expert).
  - R1-R4 and R7 remain open: they need the five interrupted runs and the knockouts.
