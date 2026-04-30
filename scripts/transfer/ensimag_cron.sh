#!/bin/bash
# Cron entry on Ensimag login node — pushes new checkpoints to HF Hub
# and cleans up local disk to stay under the 10 GB quota.
#
# Suggested crontab (Ensimag login node):
#   */5 * * * * /user/2/zimmermh/.../scripts/transfer/ensimag_cron.sh >> ~/glr-push.log 2>&1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${GLR_ENV_FILE:-${PROJECT_ROOT}/.env}"
OUTPUTS_DIR="${GLR_OUTPUTS_DIR:-${PROJECT_ROOT}/outputs}"
HF_REPO="${GLR_HF_REPO:-Helain/gated-lora-experiments}"

# Load HF_TOKEN
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[$(date -Iseconds)] ERROR: HF_TOKEN not set (looked in $ENV_FILE)" >&2
    exit 1
fi

# Use uv-managed env if available, fallback to system python3
if [[ -d "${PROJECT_ROOT}/.venv" ]]; then
    PY="${PROJECT_ROOT}/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    cd "$PROJECT_ROOT"
    PY="uv run python"
else
    PY="python3"
fi

echo "[$(date -Iseconds)] starting ensimag_push.py"
$PY "${PROJECT_ROOT}/scripts/transfer/ensimag_push.py" \
    --outputs-dir "$OUTPUTS_DIR" \
    --hf-repo "$HF_REPO"
echo "[$(date -Iseconds)] done"
