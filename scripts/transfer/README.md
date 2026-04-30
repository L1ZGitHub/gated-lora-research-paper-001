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
| **`ensimag_push.py`** | **Ensimag login node** | Scan `outputs/*/checkpoint-*/`, push non-LATEST checkpoints to HF, delete locally. On `TRAINING_DONE`, push root files and clean the run dir. Idempotent via `.uploaded` state file. |
| **`ensimag_cron.sh`** | **Ensimag login node**, every 5 min | Bash wrapper for cron registration. Loads `HF_TOKEN` from `.env`. |
| `sync_run_to_hf.py` (legacy) | VPS, requires VPN | rsync from Ensimag + upload (unused without VPN) |
| `cron_check.sh` (legacy) | VPS, requires VPN | Triggers `sync_run_to_hf.py` (unused without VPN) |

## How it works

1. Trainer saves a checkpoint (every `save_steps`) → directory
   `outputs/<run>/checkpoint-N/` with `expert_pools.pt`, etc. The trainer
   rotates `optimizer.pt` so only `checkpoint-LATEST/` keeps it.
2. `ensimag_cron.sh` (Ensimag login node, every 5 min) calls `ensimag_push.py`:
   - For each run dir, finds `checkpoint-N/` directories that are NOT
     `checkpoint-LATEST/` and not already in `.uploaded`.
   - Uploads each to HF Hub under `<run>/<checkpoint-N>/`
     (excludes `optimizer.pt`).
   - Deletes the local checkpoint directory.
   - Records the upload in the run's `.uploaded` file (idempotent).
3. When `TRAINING_DONE` appears, the next cron tick:
   - Uploads root-level files (`final_results.json`, `routing_history.json`,
     `experiment_config.json`, figures).
   - Deletes everything in the run dir except `TRAINING_DONE`
     (the marker stays so `chain_jobs.sh` knows to stop resubmitting).

## Ensimag setup (one-time, on the login node)

```bash
# 1. Clone repo
cd ~/GatedLoraProject
git clone git@github.com:L1ZGitHub/gated-lora-research-paper-001.git
cd gated-lora-research-paper-001

# 2. Install Python deps via uv (already installed in pyproject.toml — uv sync handles it)
uv sync --no-dev

# 3. Set HF_TOKEN (also used by the trainer for downloading gated models like Llama/Gemma)
echo 'HF_TOKEN=hf_xxx_your_write_token' > .env
chmod 600 .env

# 4. Smoke-test (dry-run)
.venv/bin/python scripts/transfer/ensimag_push.py \
    --outputs-dir ./outputs \
    --dry-run --verbose

# 5. Add cron entry on the Ensimag login node
( crontab -l 2>/dev/null; \
  echo "*/5 * * * * /user/2/zimmermh/GatedLoraProject/gated-lora-research-paper-001/scripts/transfer/ensimag_cron.sh >> ~/glr-push.log 2>&1" ) \
  | crontab -
```

## Tunables (env vars)

- `GLR_OUTPUTS_DIR` — outputs/ root (default: `<repo>/outputs`)
- `GLR_HF_REPO` (default `Helain/gated-lora-experiments`)
- `GLR_ENV_FILE` — path to `.env` (default: repo root)

## Storage math (with this pipeline)

Per active run:
- `checkpoint-LATEST/` (kept locally for resume) = ~74 MB (expert_pools.pt) + 85–240 MB (optimizer.pt) ≈ **160–315 MB**
- Older checkpoints uploaded + deleted within 5 min → **~0 MB extra**

With 6 concurrent runs at peak: **~1–2 GB on Ensimag**. Well below 10 GB cap.
