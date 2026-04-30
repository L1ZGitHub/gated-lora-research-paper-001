#!/bin/bash
# Collect current state of experiments from Ensimag.
# Single SSH connection (Ensimag is rate-limit-sensitive).
# Output: JSON-ish report on stdout that the supervisor LLM will read.

set -euo pipefail

REMOTE_HOST="${GLR_REMOTE_HOST:-ensimag}"
REMOTE_OUTPUTS="${GLR_REMOTE_OUTPUTS:-/user/2/zimmermh/GatedLoraProject/gated-lora-research-paper-001/outputs}"
REMOTE_LOGS="${GLR_REMOTE_LOGS:-/user/2/zimmermh/GatedLoraProject/gated-lora-research-paper-001/logs}"

ssh -o ConnectTimeout=10 "$REMOTE_HOST" bash <<EOF
set -euo pipefail

echo "=== TIMESTAMP ==="
date -Iseconds

echo ""
echo "=== DISK USAGE (\$HOME) ==="
df -h "\$HOME" | head -2

echo ""
echo "=== SLURM QUEUE (current user) ==="
squeue -u "\$USER" -o "%.10i %.9P %.20j %.2t %.10M %.10L %R" 2>&1 || echo "squeue unavailable"

echo ""
echo "=== RUN DIRECTORIES ==="
if [ -d "${REMOTE_OUTPUTS}" ]; then
    for d in "${REMOTE_OUTPUTS}"/*/; do
        [ -d "\$d" ] || continue
        run=\$(basename "\$d")
        size=\$(du -sh "\$d" 2>/dev/null | cut -f1)
        done_marker="no"
        [ -f "\$d/TRAINING_DONE" ] && done_marker="yes"
        latest=\$(ls -1d "\$d"/checkpoint-* 2>/dev/null | sort -V | tail -1 | xargs -I {} basename {} 2>/dev/null || echo "none")
        echo "  - \$run | size=\$size | done=\$done_marker | latest_ckpt=\$latest"
    done
else
    echo "  (no outputs/ directory yet)"
fi

echo ""
echo "=== LATEST FINAL_RESULTS (one per run, newest first) ==="
find "${REMOTE_OUTPUTS}" -maxdepth 2 -name "final_results.json" -printf "%T@ %p\n" 2>/dev/null \
  | sort -rn | head -10 | while read ts p; do
    echo "--- \$p ---"
    head -c 4000 "\$p" 2>/dev/null || true
    echo ""
done

echo ""
echo "=== RECENT SLURM LOG TAILS (last 60 lines, errors only) ==="
if [ -d "${REMOTE_LOGS}" ]; then
    for f in \$(ls -1t "${REMOTE_LOGS}"/*.err 2>/dev/null | head -5); do
        echo "--- \$f ---"
        tail -n 60 "\$f" 2>/dev/null | grep -iE "error|fail|nan|traceback|cuda|oom" | head -20 || echo "(no errors)"
    done
fi

echo ""
echo "=== END ==="
EOF
