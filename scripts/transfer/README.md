# Transfer (compute node → Hugging Face Hub)

Since v2 (contract `docs/v2_contract.md`, 2026-10-07) the **trainer pushes by
itself** to the private dataset `Helain/gated-lora-experiments`, under
`<GLR_RUN_NAME>/`:

| Path on HF | When |
|---|---|
| `<run>/latest/` (adapters + optimizer + scheduler + scaler + `training_state.json`) | at deadline / end |
| `<run>/best_model/` | at most every `save_steps*4`, and at end |
| `<run>/final_model/`, root files, then `<run>/TRAINING_DONE` | end (blocking, verified) |
| `<run>/slurm_logs/glr-<jobid>.*` | `train.sbatch` EXIT trap |

No `checkpoint-N/` or `epoch-N/` directories exist or are pushed any more.
Compute nodes reach the Hub directly; the VPS cannot reach the cluster, so
there is no VPS-side or cron-side sync.

## Files

| File | Status | Purpose |
|---|---|---|
| `ensimag_push.py` | kept | `push_single_run(run_dir)` (backward-compat import) + CLI for manual catch-up: pushes `latest/`, `best_model/`, `final_model/`, root files and `TRAINING_DONE` (last). Ignores legacy dirs. Never deletes locally, never calls `login()`. |
| `sync_run_to_hf.py` | deprecated | Pre-pivot VPS→Ensimag rsync path. Unused. |
| `cron_check.sh`, `ensimag_cron.sh` | **removed** | Needed SSH from the VPS to Ensimag (blocked) / pushed from the home `outputs/` that jobs no longer use. |

Manual catch-up (dry run first):

```bash
.venv/bin/python scripts/transfer/ensimag_push.py --outputs-dir /tmp/glr-<jobid>/outputs --dry-run
```

## Token

`HF_TOKEN` env var only (huggingface_hub reads it). The SLURM jobs load it
from `${GLR_HF_TOKEN_FILE:-$HOME/.hf_token}` (chmod 600) inside the job.

## Repo size

Every `latest/` push leaves old blobs in the git history. Squash periodically
with `scripts/maintenance/squash_hf_history.py` (dry run by default).
