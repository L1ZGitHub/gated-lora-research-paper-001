# Transfer pipeline (Ensimag → Hugging Face Hub, direct)

Keeps Ensimag under the 10 GB quota by pushing completed checkpoints to the
HF Hub private dataset `Helain/gated-lora-experiments` and deleting the
local copies as soon as the upload succeeds.

> **Note on architecture**: the OVH VPS **cannot reach Ensimag** through
> the school firewall (port 22 timeout). The push therefore runs **on
> the Ensimag login node**, not on the VPS.
> The `cron_check.sh` / `sync_run_to_hf.py` scripts (VPS-side, kept here
> for completeness) are unused by the production pipeline — they remain
> in case the school VPN is later configured on the VPS.

## Files

| Script | Where it runs | Purpose |
|--------|----|----|
| **`ensimag_push.py`** | **Inside the trainer process** (compute nodes) — called from `GatedLoRATrainer.save_checkpoint()` after every save. Re-authenticates against HF Hub each call (compute nodes don't keep auth state). | Library: `push_single_run(run_dir)`. Pushes non-LATEST checkpoints, deletes locally. On `TRAINING_DONE`, ships root files + cleans the run dir. Idempotent via `.uploaded` state file. |
| `ensimag_push.py` (CLI) | **Ensimag login node**, ad-hoc | Same logic as a standalone script (`--outputs-dir <path>`). Useful for manual catch-up after a bug. |
| `ensimag_cron.sh` | (kept for convenience) | Wraps the CLI for crontab. **No longer the primary path** — trainer-integrated push covers it. Use only if you want an extra safety-net cron. |
| `sync_run_to_hf.py` / `cron_check.sh` (legacy) | VPS, requires VPN | Pre-pivot scripts for the SSH-based path. Unused. |

## How it works

1. Trainer saves a checkpoint (every `save_steps`) → directory
   `outputs/<run>/checkpoint-N/` with `expert_pools.pt`, etc. The trainer
   rotates `optimizer.pt` so only the latest checkpoint keeps it.
2. **Immediately after** the local save, `GatedLoRATrainer.save_checkpoint()`
   imports `ensimag_push.push_single_run(run_dir)`:
   - Re-authenticates against HF Hub from `HF_TOKEN`
     (compute nodes don't persist auth between SLURM jobs)
   - For each `checkpoint-N/` that is NOT the just-saved one and NOT already
     in `.uploaded`: uploads to `<run>/<checkpoint-N>/` (excludes
     `optimizer.pt`), deletes locally, records in `.uploaded`.
   - On exception: logs and swallows — push failure must NEVER crash training.
3. When the trainer writes `TRAINING_DONE`, the next save's push (or
   `final_model` save) ships the root files (`final_results.json`,
   `routing_history.json`, `experiment_config.json`, figures) and reduces the
   run dir to just the `TRAINING_DONE` marker (the marker stays so
   `chain_jobs.sh` knows to stop resubmitting).

## Ensimag setup (one-time, on the login node)

```bash
# 1. Clone repo
cd ~/GatedLoraProject
git clone git@github.com:L1ZGitHub/gated-lora-research-paper-001.git
cd gated-lora-research-paper-001

# 2. Install Python deps via uv (already installed in pyproject.toml — uv sync handles it)
uv sync

# 3. Set HF_TOKEN at the repo root — used by the trainer for both
#    downloading gated models and for the post-save HF Hub push.
echo 'HF_TOKEN=hf_xxx_your_write_token' > .env
chmod 600 .env

# 4. Smoke-test (dry-run, no auth, no writes)
.venv/bin/python scripts/transfer/ensimag_push.py \
    --outputs-dir ./outputs \
    --dry-run --verbose
```

**No cron needed.** The trainer pushes inline after every checkpoint save.
If you want a defensive safety-net cron anyway, it's:

```bash
( crontab -l 2>/dev/null; \
  echo "*/15 * * * * /user/2/zimmermh/GatedLoraProject/gated-lora-research-paper-001/scripts/transfer/ensimag_cron.sh >> ~/glr-push.log 2>&1" ) \
  | crontab -
```

It's idempotent — if everything is already pushed, it's a no-op.

## Tunables (env vars)

- `GLR_OUTPUTS_DIR` — outputs/ root (default: `<repo>/outputs`)
- `GLR_HF_REPO` (default `Helain/gated-lora-experiments`)
- `GLR_ENV_FILE` — path to `.env` (default: repo root)

## Storage math (with this pipeline)

Per active run:
- `checkpoint-LATEST/` (kept locally for resume) = ~74 MB (expert_pools.pt) + 85–240 MB (optimizer.pt) ≈ **160–315 MB**
- Older checkpoints uploaded + deleted within 5 min → **~0 MB extra**

With 6 concurrent runs at peak: **~1–2 GB on Ensimag**. Well below 10 GB cap.
