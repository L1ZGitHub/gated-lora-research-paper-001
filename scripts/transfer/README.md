# Transfer pipeline (Ensimag → VPS → Hugging Face Hub)

Keeps Ensimag under the 10 GB quota by streaming completed runs to the HF Hub
private dataset `Helain/gated-lora-experiments`.

## Files

| Script | Where it runs | Purpose |
|--------|----|----|
| `sync_run_to_hf.py` | VPS (or any machine with SSH access to Ensimag) | rsync + HF upload + optional remote cleanup |
| `cron_check.sh` | VPS, every 5 min via cron | Scan for `TRAINING_DONE` markers, trigger `sync_run_to_hf.py` for new ones |

## How it works

1. SLURM job finishes a training run → trainer writes `TRAINING_DONE` marker.
2. `cron_check.sh` runs every 5 min on the VPS, lists Ensimag for new markers
   via a single SSH connection.
3. For each new run found, invokes `sync_run_to_hf.py`:
   - `rsync` from Ensimag to a temp staging dir (excludes `optimizer.pt`,
     `__pycache__`, and `checkpoint-LATEST/` if `--keep-latest`).
   - `huggingface_hub.upload_folder()` to `Helain/gated-lora-experiments`
     under `<run_name>/`.
   - With `--cleanup`: deletes synced files from Ensimag (keeps
     `TRAINING_DONE` so the orchestrator knows the run is finished).
4. State file (`~/.cache/gated-lora-synced.txt`) tracks already-synced runs
   so they're not re-uploaded on every tick.

## VPS setup (one-time)

```bash
# 1. Clone repo
git clone git@github.com:L1ZGitHub/gated-lora-research-paper-001.git
cd gated-lora-research-paper-001

# 2. Install deps (just huggingface_hub, no torch needed)
python3 -m venv .venv
source .venv/bin/activate
pip install huggingface_hub pyyaml

# 3. Set HF_TOKEN
echo 'HF_TOKEN=hf_xxx_your_write_token' > .env
chmod 600 .env

# 4. Set up SSH access to Ensimag (must work non-interactively!)
#    Test: ssh ensimag 'echo OK'

# 5. Smoke-test the sync (dry-run)
python3 scripts/transfer/sync_run_to_hf.py \
    --remote-host ensimag \
    --remote-output-dir /tmp/fake_run_for_test \
    --hf-repo Helain/gated-lora-experiments \
    --hf-path test \
    --dry-run

# 6. Add cron entry
( crontab -l 2>/dev/null; \
  echo "*/5 * * * * /home/debian/gated-lora-research-paper-001/scripts/transfer/cron_check.sh >> /var/log/glr-sync.log 2>&1" ) \
  | crontab -
```

## Tunables (env vars)

- `GLR_REMOTE_HOST` (default `ensimag`) — SSH alias
- `GLR_REMOTE_OUTPUTS` — base path on Ensimag holding `outputs/<run>/` subdirs
- `GLR_HF_REPO` (default `Helain/gated-lora-experiments`) — HF dataset repo
- `GLR_ENV_FILE` — path to .env with HF_TOKEN
- `GLR_STATE_FILE` — path to "already synced" tracker

## Storage math (with this pipeline)

Per active run:
- LATEST checkpoint (kept on Ensimag for resume) = ~74 MB (expert_pools.pt) + 85–240 MB (optimizer.pt, rotated) ≈ **160–315 MB**
- Older checkpoints synced + deleted as soon as TRAINING_DONE → **0 MB after sync**

With 6 concurrent runs: **~1–2 GB on Ensimag** at peak. Well below the 10 GB cap.
