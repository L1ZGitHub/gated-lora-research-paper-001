#!/bin/bash
# Invoke Claude Code CLI as the supervisor for the gated-lora project.
#
# Usage:
#   bash scripts/supervisor/supervise.sh                   # routine check
#   bash scripts/supervisor/supervise.sh --on-failure JOB  # post-failure check
#
# Designed to run on the VPS:
#   - Periodic cron entry (every hour)
#   - Triggered immediately by cron_check.sh when a job exits without TRAINING_DONE
#
# Requires:
#   - claude (Claude Code CLI) on PATH, authenticated (Max plan, no API key)
#   - SSH access to ensimag working non-interactively
#   - GLR_NOTIFY_TO env var (and SMTP creds) for email alerts

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${GLR_ENV_FILE:-${PROJECT_ROOT}/.env}"
REASON="${1:-routine}"
JOB_ID="${2:-}"

# Load env (HF_TOKEN, GLR_SMTP_*, GLR_NOTIFY_TO)
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

if ! command -v claude >/dev/null 2>&1; then
    echo "[supervise] ERROR: 'claude' CLI not found on PATH" >&2
    exit 1
fi

# 1. Gather state (single SSH call)
state_file=$(mktemp -t glr-state.XXXXXX)
trap 'rm -f "$state_file"' EXIT

echo "[supervise] Gathering Ensimag state..."
if ! bash "${PROJECT_ROOT}/scripts/supervisor/gather_state.sh" > "$state_file" 2>&1; then
    "${PROJECT_ROOT}/scripts/supervisor/notify.py" \
        --level critical \
        --subject "Supervisor: cannot reach Ensimag" \
        --message "$(cat "$state_file")"
    exit 2
fi

# 2. Build the supervisor input
prompt_file=$(mktemp -t glr-prompt.XXXXXX)
trap 'rm -f "$state_file" "$prompt_file"' EXIT

cat > "$prompt_file" <<EOF
You are the gated-lora supervisor agent. Read your instructions in
\`scripts/supervisor/SUPERVISOR_PROMPT.md\`, analyze the state report below,
and take the appropriate action(s).

Trigger reason: $REASON
$([ -n "$JOB_ID" ] && echo "Failed job: $JOB_ID")

You have access to:
  - Bash (for SSH to ensimag, sbatch, scancel of your own jobs, file reads)
  - Read (for any file in this repo)
  - The notify.py helper at scripts/supervisor/notify.py

When you decide on actions, execute them. When done, emit one notification
summarizing what happened.

==================== ENSIMAG STATE REPORT ====================
$(cat "$state_file")
==================== END STATE REPORT ========================

Begin your analysis.
EOF

# 3. Invoke Claude Code CLI
#    --print: non-interactive (single response)
#    --model: use Opus 4.7 (Max plan, no per-call cost)
#    --append-system-prompt: load supervisor instructions
echo "[supervise] Invoking Claude (model=opus-4.7, reason=$REASON)..."
cd "$PROJECT_ROOT"
claude \
    --model claude-opus-4-7 \
    --print \
    --append-system-prompt "$(cat "${PROJECT_ROOT}/scripts/supervisor/SUPERVISOR_PROMPT.md")" \
    < "$prompt_file" \
    | tee -a "${HOME}/.cache/gated-lora-supervisor.log"

echo "[supervise] Done at $(date -Iseconds)"
