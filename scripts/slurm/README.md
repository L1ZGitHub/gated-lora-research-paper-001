# SLURM scripts (v2)

Cluster: controller `nash`, partition `rtx6000` = turing-2..11 (3 × Quadro
RTX 6000 22 GB per node, GPUs shared via shards: NOT isolated), 4 h MaxTime,
node `/tmp` = root FS (~37 GB, shared by all jobs), tight NFS home quota,
`/data` forbidden. Contract: `docs/v2_contract.md`.

| Script | Purpose |
|---|---|
| `train.sbatch` | One ≤ 4 h slice of a run. Exports `GLR_DEADLINE`, picks an idle GPU, `/tmp` hygiene + `df` check, capped shared HF cache, resume from the highest `global_step` (HF or local `latest/`), runs `python -m gated_lora.cli`. |
| `chain_jobs.sh` | Chains slices of one run until `<run>/TRAINING_DONE` is on HF (pre-submitted `afterany` slice, flock, exclude-on-75, stop on no progress). |
| `launch_queue.sh` | Runs every `config seed` of `experiments/queue.txt` through `chain_jobs.sh` with a concurrency cap. |
| `train_packed.sbatch` | Optional: 2–3 small runs as separate processes under one `gpu:1` job. Measure peak VRAM first. Not driven by `chain_jobs.sh`. |

## Exit codes (train.sbatch)

`0` done or clean deadline stop · `75` retryable elsewhere (GPU busy, `/tmp` < 10 GB,
HF unreachable, home venv missing, < 10 min left) · `2` config error · other = trainer failure.

## Env vars

Job: `GLR_CONFIG`, `GLR_SEED`, `GLR_RUN_NAME` (required); `GLR_CODE_MODE`
(home|clone), `GLR_REPO_DIR`, `GLR_HF_REPO`, `GLR_RESUME` (auto|none|path),
`GLR_HF_TOKEN_FILE` (default `~/.hf_token`), `GLR_MIN_TMP_GB` (10),
`GLR_HF_CACHE_CAP_GB` (25), `GLR_GPU_BUSY_MB` (500), `GLR_DEADLINE_MARGIN` (600),
`GLR_KEEP_LOCAL_HOURS` (24).
Exported to Python: `GLR_DEADLINE`, `GLR_MAX_RUNTIME_SECONDS` (legacy fallback),
`GLR_RUN_NAME`, `GLR_HF_PATH`, `GLR_HF_REPO`, `GLR_OUTPUT_DIR`, `HF_TOKEN`, `HF_HOME`.

Chain: `GLR_EXCLUDE_NODES` / `--exclude` (default none), `GLR_POLL_SECONDS` (120),
`GLR_TOKEN_VIA_ENV=1` (pass the token through the sbatch environment, never argv).

## Usage

```bash
echo "hf_xxx" > ~/.hf_token && chmod 600 ~/.hf_token          # once
bash scripts/slurm/chain_jobs.sh --config configs/experiments/<x>.yaml --seed 42
bash scripts/slurm/launch_queue.sh --dry-run
nohup bash scripts/slurm/launch_queue.sh --max-concurrent 4 >> logs/launch_queue.log 2>&1 &
```
