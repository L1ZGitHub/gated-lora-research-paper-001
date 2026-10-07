# v2 refactor — interface contract (2026-10-07)

Branch `perf/fused-experts`. Goal: fastest and *correct* training before any new job.
Old (legacy) runs stay as they are; v2 results are not comparable to them by design.
Measured on the cluster GPU (Quadro RTX 6000, sm_75): matmul fp16 92 TFLOPS vs bf16 7.7 vs
fp32 12; SDPA fp16 has the memory-efficient kernel (0.10 ms), bf16 only the math kernel
(1.86 ms). A40 (sm_86) supports bf16 natively.

## Ownership (one owner per file — do not edit files you do not own)
- MODEL owner: `src/gated_lora/models/{gated_lora_v2,lora_experts,gating_network}.py`
- DATA owner: `src/gated_lora/data/multi_task_dataset.py` (+ new `src/gated_lora/data/*.py` if needed)
- TRAIN owner: `src/gated_lora/training/{gated_trainer,pipeline,config,yaml_loader}.py`,
  `src/gated_lora/cli.py`, `src/gated_lora/analysis/routing_analysis.py`
- SLURM owner: `scripts/slurm/*`, `scripts/transfer/*`, `scripts/maintenance/*`, `experiments/queue.txt`
- Configs (`configs/**`), tests (`tests/**`), docs: integrator (main session), after the owners finish.

## Config fields (TRAIN owner adds them to `config.py`; others just read them)
ModelConfig:
- `precision: str = "auto"`  — "auto" | "fp16" | "bf16" | "fp32". auto = bf16 if
  `torch.cuda.get_device_capability() >= (8, 0)` else fp16. Base model is loaded in that dtype
  (fp32 → fp32). `torch_dtype` is kept only for backward compat and ignored when precision is set.
- `gate_output_scale: str | float = "num_experts"` — multiply the mixed expert output by this
  (num_experts → uniform routing == plain LoRA with sum of ranks, each expert at its own alpha/r).
- `gated_layers: Optional[List[int]]` stays; new `gated_layers_frac: Optional[List[float]]`
  = [start, end) fractions of depth (e.g. [0.75, 1.0] = last quarter). Resolved to indices with
  bounds check; error if both set.
- `use_layer_embedding` only applies when `per_layer_gating=False` (with per-layer gates it is a
  bias shift); MODEL owner logs a warning and does not allocate it otherwise.
- `gate_reg_warmup_ratio: float = 0.1` — entropy ("L1") regulariser ramps linearly 0→1 over this
  fraction of total optimizer steps.
TrainingConfig:
- `max_steps` becomes the primary budget (int > 0 required for v2 configs); `num_epochs` only
  used when max_steps <= 0 (legacy).
- `eval_steps: int = 1000`, `eval_batch_size: int = 32`, `eval_samples_per_task_during_training: int = 250`
  (full validation set at the end).
- `save_steps: int = 500` (local `latest/` only, see checkpoints).
- `collect_routing_stats: bool = False` during training.
- `seed: int` drives model init, sampler generator, data subset selection.
DataConfig:
- `data_format: str = "v2"`; `answer_only_loss: bool = True`; `val_fraction: float = 0.05`
  (seeded slice of TRAIN used for checkpoint selection; official val/test only for final report);
  `max_samples_per_task`, `max_val_samples` keep their meaning but subsets are seeded random,
  not head-of-split; `length_bucketing: bool = False`; `cache_dir: Optional[str] = None`
  (default: `$HF_HOME/glr_data_cache`).

## Batch format (DATA → TRAIN, MODEL)
Collate returns dict with: `input_ids [B,T]`, `attention_mask [B,T]`, `labels [B,T]`
(-100 on padding AND on prompt tokens when answer_only_loss), `task: List[str]`,
`answer_mask [B,T] bool` (positions whose label is an answer token), padding trimmed to the
longest real row (right padding enforced: DATA sets `tokenizer.padding_side = "right"`).
Real EOS token appended after the answer and labelled. Each example is (prompt, answer); the
answer is never truncated — the prompt's context part is truncated instead.
DATA exposes:
- `MultiTaskDatasetLoader.create_weighted_dataloader(split="train", batch_size, seed, epoch=0, skip_samples=0)`
  — seeded generator `seed*1000+epoch`; `skip_samples` skips at the sampler-index level (no
  tokenization of skipped items).
- `create_eval_dataloader(which="val"|"final", batch_size, samples_per_task=None)` — "val" = the
  seeded train slice; "final" = official eval split (test where the old code used it).
  Unshuffled but task-interleaved is NOT required; routing analysis builds its own balanced loader
  via `create_balanced_loader(samples_per_task, batch_size)`.
- Tokenised subsets cached on disk keyed by (tokenizer name, task, split, N, seed, max_length, data_format).

## Model interface (MODEL → TRAIN)
- `forward(input_ids, attention_mask=None, labels=None, return_routing_info=False, **kw)`;
  calls the base model with `use_cache=False`. Stores `attention_mask` for the hooks so that
  load-balancing loss, entropy regulariser and routing stats are computed over real tokens only.
- Regularisers are MEANS over gated layers (not sums). Output dict keys unchanged:
  `loss`, `lm_loss`, `logits`, `load_balancing_loss`, `l1_gate_loss` (entropy), `routing_info`.
- `model.reg_scale: float` (set by TRAIN each step, 0→1 ramp) multiplies the entropy regulariser.
- `model.collect_routing_stats: bool` — routing stats (entropy, usage, top1…) only computed when
  True or when return_routing_info=True.
- Trainable params (experts, gates) are fp32; base model in the precision dtype. MODEL casts
  activations inside the hooks (`x.to(param dtype)` and result back to the module output dtype) so
  that it works both with and without autocast.
- `save_pretrained`/`from_pretrained` stay compatible with legacy checkpoints (param names unchanged;
  fp32 on save, cast on load).
- Top-k uses the tempered gate weights cached per layer (one routing per layer, not per module).

## Training (TRAIN)
- Mixed precision: `torch.autocast("cuda", dtype=fp16|bf16)` around forward; `GradScaler` for fp16;
  trainable params fp32; AdamW on fp32. Same path for baseline (PEFT adapters cast to fp32) and gated.
- Remove trainer-side extra load-balancing addition (the model adds it once).
- Logging syncs: accumulate detached loss on GPU; `.item()` only at logging steps.
- Evaluation: no duplicate eval at the same step; eval every `eval_steps` on
  `eval_samples_per_task_during_training`; full "val" + "final" eval at the end. Reported per task:
  answer loss, answer token accuracy, and teacher-forced exact match (all answer tokens argmax-correct).
- Routing analysis in training: per-task balanced loader, mean over (batch, real tokens).
- Deadline: env `GLR_DEADLINE` (unix seconds, set by sbatch). Stop when
  `now + 2*max_step_time + save_margin > deadline`; save `latest/` + push before exit. Fallback to
  `max_runtime_seconds` if unset.
- Checkpoints (local, atomic: write `name.tmp/` then `os.replace`):
  `latest/` = adapter weights + optimizer + scheduler + scaler + `training_state.json`
  (global_step, epoch, samples_seen, rng states); `best_model/` = adapter weights only;
  `final_model/` = adapter weights only. No `checkpoint-N/`, no `epoch-N/`.
- HF push (when enabled): `latest/` only at deadline/end; `best_model/` at most once per
  `save_steps*4` and at end; `final_model/` + `TRAINING_DONE` at end, with retry + existence check.
  Uploads in a background thread; the final push blocks.
- Grad accumulation: an optimizer step is taken on every full window; leftovers at epoch end are
  carried into the next epoch's window (not an oversized step); resume restores exact sample position.

## SLURM (SLURM owner)
- Export `GLR_DEADLINE=$(( start + walltime - 600 ))` at job start (walltime from `squeue -h -j $SLURM_JOB_ID -o %L` or the sbatch time).
- GPU choice: check the allocated GPU; if memory.used > 500 MB, pick another card with < 500 MB on
  the node and log it; if none, exit 75 (chain resubmits with `--exclude=$(hostname)`).
- `/tmp` hygiene: only delete `glr-<jobid>` dirs whose job id is not in `squeue`. Start-up `df`
  check (< 10 GB free on /tmp → exit 75). Shared HF cache with a size cap.
- Resume: pick the candidate (local or HF `latest/`) with the highest `global_step`.
- Chaining: pre-submit next slice with `--dependency=afterany:$prev`; job name = run name; flock +
  squeue check against double submission; stop after 2 slices without progress.
- Optional packed mode: N small runs as N processes under one `gpu:1` job.
- No apostrophes inside `${VAR:?message}`.

## Integration changes (after the owner pass, 2026-10-07)
- **Selective output head** (`models/selective_head.py`): the trainer wraps the HF output
  embedding and projects only supervised positions (indices computed on the CPU collate tensors,
  no sync); it computes the LM loss itself (`F.cross_entropy`, mean over supervised tokens = the
  HF loss) and adds `model.aux_losses()["total"]` for gated models. `forward(labels=...)` is
  unchanged for analysis scripts. Eval cross-entropy runs in 2048-row fp32 chunks.
- **Run names**: v2 configs live in `configs/experiments/v2/` and are prefixed `v2_` so their HF
  run dirs (`<basename>_seed<N>`) never resume or skip a v1 run. `configs/base/v2.yaml` holds the
  v2 defaults (put it last in `extends`).
- **Exit 78** (EX_CONFIG) from `cli.py` on an invalid config or a numeric overflow
  (`FloatingPointError`: non-finite loss without scaler, or fp16 grad scale < 1);
  `chain_jobs.sh` stops the chain (exit 5) instead of resubmitting.
- **SIGTERM** stops at the next optimizer step through the deadline path (save + push `latest/`);
  an exception pushes the last local `latest/` before re-raising.
- `train.sbatch` passes `--analyze-routing` (gated only, skipped when the deadline is close).
- Param-matched plain LoRA for Qwen2.5-0.5B is r=66 in v2 (per-layer gates have no layer
  embedding): gated 36,323,400 vs r66 36,292,608. Other models: recompute before use.

## Second review pass (2026-10-07)
- **Speed**: `data.sort_within_step` (length sort inside each optimizer step; task mix per step
  unchanged; resume exact); supervised positions computed + pinned in the collator
  (`sup_*`, `ans_*`); fused AdamW (no per-step host sync with GradScaler); PEFT input fp32 cast
  disabled under autocast; `training.skip_base_attention_mask` (exact with right padding; the
  gated forward takes `base_attention_mask=False`, hooks keep the mask); logits freed before
  backward; post-training routing analysis under `trainer.logits_off()`.
- **Design**: per-example dump `final_examples.npz` on the final split (answer NLL / tokens /
  correct, prompt_len, example_idx; gated: per-layer gate means over prompt and over answer
  tokens, [N, L, E] fp16); greedy generation eval (`training.generation_tasks`, GSM8K
  final-number accuracy, XSum ROUGE-L, `generation_results.json`), via `GatedLoRAModelV2.generate`
  (gate cache reset by a pre-hook on every base-model call); frozen-random-gate arm
  (`model.freeze_gating` + `frozen_gate_target_top1`, per-layer calibration at init);
  plain-LoRA rank curve r8/r16/r32; data cap 20k/task; Phase A LR sweep with 2 seeds.
- **Robustness**: SIGTERM handler restored after the loop; precision change on resume warned;
  `GLR_STOP_AT_STEP` test hook; padding efficiency logged.
- Not done: `analysis/cluster/routing_maps_hf.py` still uses the legacy data API and legacy
  model defaults: port it to v2 (data + `gate_output_scale`) before the knockout test on v2 runs.

## Phase B (2026-10-07)
- Phase A (2 seeds x {5e-5, 1e-4, 2e-4, 4e-4}): 5e-5 best for both gated and r66 -> base LR
  5e-5; the 2e-4 configs carry their LR explicitly. Frozen-gate target top-1 0.653 (trained
  gates at 5e-5). Queue `experiments/queue_v2_phase_b.txt` (28 runs, incl. a 2.5e-5 check).
- Best-checkpoint test eval: after the last-step evals, `best_model/` weights are loaded, the
  final split (+ generation) re-evaluated, then the last-step weights restored. Results:
  `eval_results.json["final_best"]` (`same_as_final` when best = last step; null if skipped
  for the deadline), `final_examples_best.npz`, `generation_results_best.json`. Phase A runs
  predate it (their seeds 0-1 at 5e-5 need an eval-only pass on `best_model/`).
