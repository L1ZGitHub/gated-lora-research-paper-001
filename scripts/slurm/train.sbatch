#!/bin/bash
# SLURM training template — stage-in/stage-out on compute /tmp.
#
# WHY: Ensimag compute nodes have NO access to the user's NFS home
# (NSS/LDAP broken on compute → autofs never mounts the home). But /tmp
# is local-writable (≥40 GB), git/curl/python3 work, and HF Hub is
# reachable over HTTPS. So each job is fully self-contained:
#
#   1. Clone the (now public) code repo into /tmp
#   2. Install uv + Python deps into /tmp
#   3. If a `latest` checkpoint exists on HF Hub for this run, pull it
#   4. Run training (post-save push to HF is built into the trainer)
#   5. SLURM wipes /tmp at job end — only HF Hub state matters
#
# Required env vars (set by chain_jobs.sh via sbatch --export):
#   GLR_CONFIG    relative path to experiment YAML
#   GLR_SEED      training seed
#   GLR_RUN_NAME  run identifier on HF Hub (e.g. phi2_harder_multitask_seed42)
#   HF_TOKEN      Hugging Face token (gated model downloads + checkpoint push)
#
# Optional:
#   GLR_REPO_URL  default https://github.com/L1ZGitHub/gated-lora-research-paper-001.git
#   GLR_HF_REPO   default Helain/gated-lora-experiments
#   GLR_RESUME    "auto" (default) | "none" | explicit path
#   GLR_WANDB_MODE  default "offline"
#   GLR_UV_VERSION  default 0.5.18

#SBATCH --job-name=glr
#SBATCH --output=/tmp/glr-%j.out
#SBATCH --error=/tmp/glr-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --nodelist=turing-[4-9]

set -euo pipefail

# ----------------------- sanity ----------------------------
: "${GLR_CONFIG:?must be set}"
: "${GLR_SEED:?must be set}"
: "${GLR_RUN_NAME:?must be set}"
: "${HF_TOKEN:?must be set (load from .env in chain_jobs.sh)}"

REPO_URL="${GLR_REPO_URL:-https://github.com/L1ZGitHub/gated-lora-research-paper-001.git}"
HF_REPO="${GLR_HF_REPO:-Helain/gated-lora-experiments}"
UV_VERSION="${GLR_UV_VERSION:-0.5.18}"
RESUME="${GLR_RESUME:-auto}"

JOB_ID="${SLURM_JOB_ID:-local-$$}"
WORK="/tmp/glr-${JOB_ID}"
mkdir -p "$WORK"
cd "$WORK"

# ----------------------- env redirect ----------------------
# Compute has no NFS home. Reroute every tool that defaults to $HOME so it
# writes to /tmp instead. WANDB_MODE=offline avoids any home-dir access.
export HOME="$WORK"
export HF_HOME="$WORK/hf_cache"
export UV_CACHE_DIR="$WORK/uv_cache"
export WANDB_DIR="$WORK/wandb"
export WANDB_MODE="${GLR_WANDB_MODE:-disabled}"
export WANDB_DISABLE_SERVICE=true
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$WANDB_DIR"

echo "==================================================================="
echo "SLURM job $JOB_ID on $(hostname) at $(date -Iseconds)"
echo "  WORK     : $WORK"
echo "  Config   : $GLR_CONFIG"
echo "  Seed     : $GLR_SEED"
echo "  Run      : $GLR_RUN_NAME"
echo "  Resume   : $RESUME"
echo "  HF repo  : $HF_REPO"
echo "==================================================================="

# Best-effort: upload the SLURM .out/.err to HF when this script exits.
upload_slurm_logs() {
    local rc=$?
    set +e
    local out="/tmp/glr-${JOB_ID}.out" err="/tmp/glr-${JOB_ID}.err"
    if [[ -x "$WORK/glr/.venv/bin/python" ]]; then
        "$WORK/glr/.venv/bin/python" - <<EOF || true
import os
from huggingface_hub import HfApi, login
try:
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    api = HfApi()
    for p in ("$out", "$err"):
        try:
            api.upload_file(
                path_or_fileobj=p,
                path_in_repo=f"{os.environ['GLR_RUN_NAME']}/slurm_logs/{p.split('/')[-1]}",
                repo_id="$HF_REPO", repo_type="dataset",
                commit_message=f"slurm log {p.split('/')[-1]}",
            )
        except Exception as e:
            print(f"  log upload skipped ({p}): {e}", flush=True)
except Exception as e:
    print(f"  HF login failed for log upload: {e}", flush=True)
EOF
    fi
    echo "[exit] code=$rc at $(date -Iseconds)"
    exit $rc
}
trap upload_slurm_logs EXIT

# ----------------------- stage-in: code --------------------
echo "[stage-in] git clone..."
git clone --depth=1 --quiet "$REPO_URL" glr
cd glr

# ----------------------- stage-in: uv ----------------------
echo "[stage-in] downloading uv $UV_VERSION..."
curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar xz -C "$WORK"
UV="$WORK/uv-x86_64-unknown-linux-gnu/uv"
"$UV" --version

# ----------------------- stage-in: python deps -------------
echo "[stage-in] uv sync (this is the long bit, ~3-5 min)..."
"$UV" sync --no-dev
echo "[stage-in] venv: $(du -sh .venv | cut -f1)"

# ----------------------- stage-in: resume checkpoint -------
OUTPUT_DIR="$WORK/glr/outputs/$GLR_RUN_NAME"
mkdir -p "$OUTPUT_DIR"

if [[ "$RESUME" == "none" ]]; then
    echo "[stage-in] resume=none — skipping checkpoint pull"
else
    echo "[stage-in] checking HF Hub for existing $GLR_RUN_NAME state..."
    .venv/bin/python - <<EOF
import os, sys
from huggingface_hub import HfApi, login, snapshot_download
run_name = os.environ["GLR_RUN_NAME"]
hf_repo  = "$HF_REPO"
out_dir  = "$OUTPUT_DIR"
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
api = HfApi()
try:
    files = list(api.list_repo_files(repo_id=hf_repo, repo_type="dataset"))
except Exception as e:
    print(f"  list failed ({e}) — assuming fresh start", flush=True)
    sys.exit(0)

if any(f == f"{run_name}/TRAINING_DONE" for f in files):
    print("  TRAINING_DONE already on HF — nothing to do, exiting cleanly", flush=True)
    # Touch a local marker so the bash script exits cleanly (the chain
    # orchestrator already saw the HF marker and won't resubmit).
    open(os.path.join(out_dir, "TRAINING_DONE"), "w").write("(already done)\n")
    sys.exit(0)

latest_files = [f for f in files if f.startswith(f"{run_name}/latest/")]
if latest_files:
    print(f"  pulling {len(latest_files)} files from {run_name}/latest/ ...", flush=True)
    snapshot_download(
        repo_id=hf_repo, repo_type="dataset",
        allow_patterns=[f"{run_name}/latest/*"],
        local_dir=out_dir, local_dir_use_symlinks=False,
    )
    # snapshot_download lays out as <out_dir>/<run_name>/latest/. Move up.
    import shutil
    nested = os.path.join(out_dir, run_name, "latest")
    target = os.path.join(out_dir, "latest")
    if os.path.isdir(nested):
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.move(nested, target)
        shutil.rmtree(os.path.join(out_dir, run_name), ignore_errors=True)
    print(f"  resume checkpoint ready at {target}", flush=True)
else:
    print("  no prior state on HF — fresh start", flush=True)
EOF

    # Short-circuit if the resume helper dropped a "(already done)" marker
    if [[ -f "$OUTPUT_DIR/TRAINING_DONE" ]] && grep -q "(already done)" "$OUTPUT_DIR/TRAINING_DONE"; then
        echo "[done] run already complete on HF — exiting"
        exit 0
    fi
fi

# ----------------------- train -----------------------------
echo "[train] starting at $(date -Iseconds)"
.venv/bin/python -m gated_lora.cli \
    --config "$GLR_CONFIG" \
    --seed "$GLR_SEED" \
    --output-dir "$OUTPUT_DIR" \
    --resume "$RESUME"

echo "[train] python exited cleanly at $(date -Iseconds)"
