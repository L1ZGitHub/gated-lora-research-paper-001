#!/bin/bash
# Collect state for the supervisor by reading the HF Hub dataset
# (the OVH VPS cannot reach Ensimag through the school firewall).
# Output is plain text on stdout, designed to be fed to the LLM supervisor.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${GLR_ENV_FILE:-${PROJECT_ROOT}/.env}"
HF_REPO="${GLR_HF_REPO:-Helain/gated-lora-experiments}"

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set"
    exit 1
fi

# Resolve python interpreter — prefer the VPS venv if present
if [[ -d "${PROJECT_ROOT}/.venv" ]]; then
    PY="${PROJECT_ROOT}/.venv/bin/python"
else
    PY="python3"
fi

echo "=== TIMESTAMP ==="
date -Iseconds

echo ""
echo "=== HF DATASET STATE ($HF_REPO) ==="
HF_TOKEN="$HF_TOKEN" HF_REPO="$HF_REPO" $PY - <<'EOF'
import os, json, datetime as dt
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
api = HfApi()
repo = os.environ["HF_REPO"]

# List all files in the dataset
try:
    files = list(api.list_repo_files(repo_id=repo, repo_type="dataset"))
except Exception as exc:
    print(f"  ERROR listing repo: {exc}")
    raise SystemExit(0)

# Group by run (top-level dir)
runs = {}
for f in files:
    parts = f.split("/")
    if len(parts) < 2:
        continue
    runs.setdefault(parts[0], []).append(f)

print(f"  {len(runs)} run(s) on HF Hub")
for run, run_files in sorted(runs.items()):
    checkpoints = sorted({p.split("/")[1] for p in run_files if p.startswith(f"{run}/checkpoint-")})
    has_final = any(p.endswith("/final_results.json") for p in run_files)
    print(f"    {run}: {len(checkpoints)} ckpt | final={has_final}")

# Recent commits — proxy for "is this run still progressing?"
commits = list(api.list_repo_commits(repo_id=repo, repo_type="dataset"))[:20]
now = dt.datetime.now(dt.timezone.utc)
print()
print(f"  Recent commits (last 20):")
for c in commits:
    age = now - c.created_at
    age_h = age.total_seconds() / 3600
    print(f"    [{age_h:5.1f}h ago] {c.commit_id[:8]} — {c.title[:80]}")

if commits:
    last_age_h = (now - commits[0].created_at).total_seconds() / 3600
    print()
    if last_age_h > 6:
        print(f"  WARNING: no upload in {last_age_h:.1f} hours — possible stall")
    else:
        print(f"  Last activity: {last_age_h:.1f}h ago — healthy")
EOF

echo ""
echo "=== LOCAL STATE (VPS) ==="
df -h "$HOME" | head -2
echo ""
echo "Recent supervisor invocations (last 10):"
tail -n 100 "${HOME}/.cache/gated-lora-supervisor.log" 2>/dev/null | grep -E "Invoking|Done at" | tail -10 || echo "  (no log yet)"

echo ""
echo "Recent notifications:"
tail -n 30 "${HOME}/.cache/gated-lora-notify.log" 2>/dev/null | tail -15 || echo "  (none)"

echo ""
echo "=== END ==="
