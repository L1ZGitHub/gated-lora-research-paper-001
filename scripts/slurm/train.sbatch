#!/bin/bash
# SLURM training job — one ≤4h slice of a (config × seed) run.
#
# TWO CODE MODES (GLR_CODE_MODE):
#   home  (default) — run from the NFS home checkout + its .venv.
#       The 2026-04..06 NSS/NFS outage on compute nodes is FIXED (verified
#       2026-07-02: compute resolves the user and mounts the home). This
#       skips the 3-5 min git-clone + uv-sync stage-in of clone mode.
#   clone — legacy self-contained stage-in (git clone + uv into /tmp).
#       Keep as fallback if the home mount regresses.
#
# STORAGE POLICY (Ensimag quota):
#   - outputs/checkpoints  → /tmp (node-local), pushed to HF Hub by the
#     trainer after every save, deleted locally once pushed. Nothing lands
#     on the NFS home.
#   - HF model cache       → /tmp/glr_hf_cache (SHARED per node, survives
#     across jobs → Phi-2 & co are not re-downloaded by every job).
#   - SLURM logs           → /tmp, uploaded to HF by the EXIT trap.
#
# 4h CHAINING:
#   The trainer stops itself at GLR_MAX_RUNTIME_SECONDS (default 3h30),
#   saves + pushes a "latest" checkpoint, and exits 0. chain_jobs.sh sees
#   no TRAINING_DONE on HF and submits the next slice, which resumes from
#   "latest" (or the highest checkpoint-N as fallback).
#
# Required env vars (set by chain_jobs.sh via sbatch --export):
#   GLR_CONFIG    relative path to experiment YAML
#   GLR_SEED      training seed
#   GLR_RUN_NAME  run identifier on HF Hub (e.g. phi2_harder_multitask_seed42)
#   HF_TOKEN      Hugging Face token (gated model downloads + checkpoint push)
#
# Optional:
#   GLR_CODE_MODE            home (default) | clone
#   GLR_REPO_DIR             home-mode checkout (default ~/GatedLoraProject/gated-lora-research-paper-001)
#   GLR_REPO_URL             clone-mode URL (default github.com/L1ZGitHub/gated-lora-research-paper-001)
#   GLR_HF_REPO              default Helain/gated-lora-experiments
#   GLR_RESUME               "auto" (default) | "none" | explicit path
#   GLR_MAX_RUNTIME_SECONDS  trainer time budget (default 12600 = 3h30)
#   GLR_WANDB_MODE           default "disabled"
#   GLR_UV_VERSION           clone mode only (default 0.5.18)

#SBATCH --job-name=glr
#SBATCH --output=/tmp/glr-%j.out
#SBATCH --error=/tmp/glr-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
# NOTE: --partition / --nodelist are set by chain_jobs.sh at submit time.

set -euo pipefail

# ----------------------- sanity ----------------------------
: "${GLR_CONFIG:?must be set}"
: "${GLR_SEED:?must be set}"
: "${GLR_RUN_NAME:?must be set}"
: "${HF_TOKEN:?must be set (load from .env in chain_jobs.sh)}"

CODE_MODE="${GLR_CODE_MODE:-home}"
REPO_DIR="${GLR_REPO_DIR:-$HOME/GatedLoraProject/gated-lora-research-paper-001}"
REPO_URL="${GLR_REPO_URL:-https://github.com/L1ZGitHub/gated-lora-research-paper-001.git}"
HF_REPO="${GLR_HF_REPO:-Helain/gated-lora-experiments}"
UV_VERSION="${GLR_UV_VERSION:-0.5.18}"
RESUME="${GLR_RESUME:-auto}"
export GLR_MAX_RUNTIME_SECONDS="${GLR_MAX_RUNTIME_SECONDS:-12600}"

JOB_ID="${SLURM_JOB_ID:-local-$$}"
WORK="/tmp/glr-${JOB_ID}"

# Opportunistic /tmp hygiene: remove per-JOB dirs/logs of finished jobs.
# NEVER touches /tmp/glr_hf_cache (shared model cache, meant to persist).
find /tmp -maxdepth 1 -name "glr-*" -mmin +30 -type d 2>/dev/null \
    -exec rm -rf {} \; 2>/dev/null || true
find /tmp -maxdepth 1 -name "glr-*.out" -mmin +60 -type f -delete 2>/dev/null || true
find /tmp -maxdepth 1 -name "glr-*.err" -mmin +60 -type f -delete 2>/dev/null || true

mkdir -p "$WORK"
cd "$WORK"

# ----------------------- env -------------------------------
# Shared per-node HF cache: models survive across jobs (huggingface_hub uses
# file locks, safe for concurrent jobs on the same node).
export HF_HOME="/tmp/glr_hf_cache"
export UV_CACHE_DIR="$WORK/uv_cache"
export WANDB_DIR="$WORK/wandb"
export WANDB_MODE="${GLR_WANDB_MODE:-disabled}"
export WANDB_DISABLE_SERVICE=true
mkdir -p "$HF_HOME" "$WANDB_DIR"

echo "==================================================================="
echo "SLURM job $JOB_ID on $(hostname) at $(date -Iseconds)"
echo "  Mode     : $CODE_MODE"
echo "  WORK     : $WORK"
echo "  Config   : $GLR_CONFIG"
echo "  Seed     : $GLR_SEED"
echo "  Run      : $GLR_RUN_NAME"
echo "  Resume   : $RESUME"
echo "  HF repo  : $HF_REPO"
echo "  Budget   : ${GLR_MAX_RUNTIME_SECONDS}s"
echo "==================================================================="

# ----------------------- code + python ---------------------
if [[ "$CODE_MODE" == "home" ]]; then
    if [[ ! -x "$REPO_DIR/.venv/bin/python" ]]; then
        echo "FATAL: $REPO_DIR/.venv/bin/python not found/executable." >&2
        echo "  Home mount broken again? Fall back with GLR_CODE_MODE=clone" >&2
        exit 3
    fi
    CODE_DIR="$REPO_DIR"
    PYBIN="$REPO_DIR/.venv/bin/python"
else
    # clone mode: python3 + pip user-install of huggingface_hub for the trap,
    # then clone + uv sync as before.
    export HOME="$WORK"   # no NFS home assumption in clone mode
    python3 -m pip install --quiet --user --break-system-packages huggingface_hub 2>&1 | tail -3 || true
fi

# Best-effort: upload the SLURM .out/.err to HF when this script exits.
upload_slurm_logs() {
    local rc=$?
    set +e
    local trap_py="${PYBIN:-python3}"
    if ! "$trap_py" -c "import huggingface_hub" 2>/dev/null; then
        echo "[trap] huggingface_hub unavailable — cannot upload logs (rc=$rc)"
        echo "[exit] code=$rc at $(date -Iseconds)"
        exit $rc
    fi
    GLR_JOB_ID="$JOB_ID" GLR_HF_REPO_TRAP="$HF_REPO" GLR_RUN_NAME_TRAP="$GLR_RUN_NAME" \
        HF_TOKEN="$HF_TOKEN" "$trap_py" - <<'PYEOF'
import os
from huggingface_hub import HfApi, login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
api = HfApi()
job_id = os.environ["GLR_JOB_ID"]
repo   = os.environ["GLR_HF_REPO_TRAP"]
run    = os.environ["GLR_RUN_NAME_TRAP"]
for kind in ("out", "err"):
    p = f"/tmp/glr-{job_id}.{kind}"
    if not os.path.isfile(p):
        continue
    try:
        api.upload_file(
            path_or_fileobj=p,
            path_in_repo=f"{run}/slurm_logs/glr-{job_id}.{kind}",
            repo_id=repo, repo_type="dataset",
            commit_message=f"slurm log glr-{job_id}.{kind}",
        )
    except Exception as e:
        print(f"[trap] upload failed for {p}: {e}", flush=True)
PYEOF
    # Reclaim disk so the next job on this node doesn't hit ENOSPC.
    # (The shared HF cache is intentionally NOT removed.)
    rm -rf "$WORK" 2>/dev/null || true
    echo "[exit] code=$rc at $(date -Iseconds)"
    exit $rc
}
trap upload_slurm_logs EXIT

if [[ "$CODE_MODE" != "home" ]]; then
    echo "[stage-in] git clone..."
    git clone --depth=1 --quiet "$REPO_URL" glr
    cd glr
    echo "[stage-in] downloading uv $UV_VERSION..."
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar xz -C "$WORK"
    UV="$WORK/uv-x86_64-unknown-linux-gnu/uv"
    "$UV" --version
    echo "[stage-in] uv sync (~3-5 min)..."
    "$UV" sync --no-dev
    echo "[stage-in] venv: $(du -sh .venv | cut -f1)"
    CODE_DIR="$WORK/glr"
    PYBIN="$WORK/glr/.venv/bin/python"
fi

echo "[env] python: $PYBIN"
"$PYBIN" -c "import torch; print('[env] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ----------------------- stage-in: resume checkpoint -------
# Outputs ALWAYS live on node-local /tmp (never on the NFS home — quota).
OUTPUT_DIR="$WORK/outputs/$GLR_RUN_NAME"
mkdir -p "$OUTPUT_DIR"

if [[ "$RESUME" == "none" ]]; then
    echo "[stage-in] resume=none — skipping checkpoint pull"
else
    echo "[stage-in] checking HF Hub for existing $GLR_RUN_NAME state..."
    GLR_OUTPUT_DIR="$OUTPUT_DIR" GLR_HF_REPO_PULL="$HF_REPO" HF_TOKEN="$HF_TOKEN" \
        GLR_RUN_NAME="$GLR_RUN_NAME" "$PYBIN" - <<'EOF'
import os, re, shutil, sys
from huggingface_hub import HfApi, login, snapshot_download

run_name = os.environ["GLR_RUN_NAME"]
hf_repo  = os.environ["GLR_HF_REPO_PULL"]
out_dir  = os.environ["GLR_OUTPUT_DIR"]
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
api = HfApi()
try:
    files = list(api.list_repo_files(repo_id=hf_repo, repo_type="dataset"))
except Exception as e:
    print(f"  list failed ({e}) — assuming fresh start", flush=True)
    sys.exit(0)

if any(f == f"{run_name}/TRAINING_DONE" for f in files):
    print("  TRAINING_DONE already on HF — nothing to do, exiting cleanly", flush=True)
    open(os.path.join(out_dir, "TRAINING_DONE"), "w").write("(already done)\n")
    sys.exit(0)

def pull(subdir: str) -> bool:
    prefix = f"{run_name}/{subdir}/"
    matching = [f for f in files if f.startswith(prefix)]
    if not matching:
        return False
    print(f"  pulling {len(matching)} files from {prefix} ...", flush=True)
    snapshot_download(
        repo_id=hf_repo, repo_type="dataset",
        allow_patterns=[f"{prefix}*"],
        local_dir=out_dir, local_dir_use_symlinks=False,
    )
    nested = os.path.join(out_dir, run_name, subdir)
    target = os.path.join(out_dir, subdir)
    if os.path.isdir(nested):
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.move(nested, target)
        shutil.rmtree(os.path.join(out_dir, run_name), ignore_errors=True)
    print(f"  resume checkpoint ready at {target}", flush=True)
    return True

# Preferred: "latest" (written on timer exit). Fallback: highest checkpoint-N
# (covers the SIGKILL-before-latest-save case — without this, a killed job
# silently restarted the whole run from scratch).
if not pull("latest"):
    steps = []
    for f in files:
        m = re.match(rf"^{re.escape(run_name)}/checkpoint-(\d+)/", f)
        if m:
            steps.append(int(m.group(1)))
    if steps:
        best = max(steps)
        print(f"  no latest/ on HF — falling back to checkpoint-{best}", flush=True)
        pull(f"checkpoint-{best}")
    else:
        print("  no prior state on HF — fresh start", flush=True)
EOF

    if [[ -f "$OUTPUT_DIR/TRAINING_DONE" ]] && grep -q "(already done)" "$OUTPUT_DIR/TRAINING_DONE"; then
        echo "[done] run already complete on HF — exiting"
        exit 0
    fi
fi

# ----------------------- train -----------------------------
cd "$CODE_DIR"
echo "[train] starting at $(date -Iseconds)"
"$PYBIN" -m gated_lora.cli \
    --config "$GLR_CONFIG" \
    --seed "$GLR_SEED" \
    --output-dir "$OUTPUT_DIR" \
    --resume "$RESUME"

echo "[train] python exited cleanly at $(date -Iseconds)"
