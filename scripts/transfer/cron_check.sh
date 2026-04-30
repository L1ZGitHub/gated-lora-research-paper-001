#!/bin/bash
# Cron job: check Ensimag for completed runs (TRAINING_DONE markers)
# and trigger sync_run_to_hf.py for each.
#
# Designed to run on the VPS (every 5 min via cron).
#
# crontab -e:
#   */5 * * * * /home/debian/gated-lora-research-paper-001/scripts/transfer/cron_check.sh >> /var/log/glr-sync.log 2>&1

set -euo pipefail

REMOTE_HOST="${GLR_REMOTE_HOST:-ensimag}"
REMOTE_OUTPUTS="${GLR_REMOTE_OUTPUTS:-/user/2/zimmermh/GatedLoraProject/gated-lora-research-paper-001/outputs}"
HF_REPO="${GLR_HF_REPO:-Helain/gated-lora-experiments}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${GLR_ENV_FILE:-${PROJECT_ROOT}/.env}"
STATE_FILE="${GLR_STATE_FILE:-${HOME}/.cache/gated-lora-synced.txt}"

mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

# Load HF_TOKEN from .env if present
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[cron] ERROR: HF_TOKEN not set (looked in $ENV_FILE)" >&2
    exit 1
fi

echo "[cron] $(date -Iseconds): scanning $REMOTE_HOST:$REMOTE_OUTPUTS"

# Find directories that have TRAINING_DONE and are not in our state file.
# (single SSH connection — efficient and Ensimag-friendly)
candidates=$(ssh -o ConnectTimeout=10 "$REMOTE_HOST" \
    "find ${REMOTE_OUTPUTS} -maxdepth 2 -name TRAINING_DONE -type f 2>/dev/null | sort" \
    || { echo "[cron] SSH failed — will retry next tick"; exit 0; })

if [[ -z "$candidates" ]]; then
    echo "[cron] no completed runs found"
    exit 0
fi

while IFS= read -r marker; do
    [[ -z "$marker" ]] && continue
    run_dir=$(dirname "$marker")
    run_name=$(basename "$run_dir")

    if grep -qxF "$run_name" "$STATE_FILE"; then
        echo "[cron] already synced: $run_name"
        continue
    fi

    echo "[cron] syncing $run_name → hf://${HF_REPO}/${run_name}"
    python3 "${PROJECT_ROOT}/scripts/transfer/sync_run_to_hf.py" \
        --remote-host "$REMOTE_HOST" \
        --remote-output-dir "$run_dir" \
        --hf-repo "$HF_REPO" \
        --hf-path "$run_name" \
        --keep-latest \
        --cleanup

    # Record success
    echo "$run_name" >> "$STATE_FILE"
    echo "[cron] $run_name synced successfully"
done <<< "$candidates"

echo "[cron] done"
