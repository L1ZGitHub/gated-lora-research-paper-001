#!/bin/bash
# SLURM training job: one <= 4h slice of a (config x seed) run  (v2 contract).
#
# CODE MODES (GLR_CODE_MODE):
#   home  (default) run from the NFS home checkout + its .venv (no stage-in).
#   clone           self-contained stage-in (git clone + uv sync into /tmp);
#                   fallback if the home mount regresses. uv cache is shared
#                   per node under /tmp/glr_uv_cache.
#
# NODE-LOCAL LAYOUT (/tmp is the root FS, ~37 GB, shared by every job):
#   /tmp/glr-<jobid>/            per-job WORK dir (outputs, clone, wandb)
#   /tmp/glr-<jobid>.out|.err    SLURM logs (uploaded to HF by the EXIT trap)
#   /tmp/glr_hf_cache/           shared HF cache (models), size-capped
#   /tmp/glr_uv_cache/           shared uv cache (clone mode only)
#   Hygiene only ever deletes /tmp/glr-<jobid>* entries owned by us whose job
#   id is no longer in squeue. A WORK dir holding an unfinished run `latest/`
#   is preserved for GLR_KEEP_LOCAL_HOURS (default 24) so that the next slice
#   on the same node can resume from it if the HF push failed.
#
# DEADLINE: GLR_DEADLINE (unix seconds) = now + time left (squeue %L) - 600.
#   The trainer stops, saves `latest/` and pushes before that time.
#   GLR_MAX_RUNTIME_SECONDS is also exported (= GLR_DEADLINE - launch time)
#   for the legacy trainer code path.
#
# RESUME (GLR_RESUME=auto): candidates = HF `<run>/latest/` and preserved
#   local `/tmp/glr-*/.../outputs/<run>/latest/`; the highest global_step in
#   training_state.json wins. HF unreachable -> exit 75 (never a silent
#   fresh start that would overwrite HF progress).
#
# HF TOKEN: read INSIDE the job from ${GLR_HF_TOKEN_FILE:-$HOME/.hf_token}
#   (raw token or a HF_TOKEN=... line), then $GLR_REPO_DIR/.env, then an
#   inherited HF_TOKEN env var. Never passed on a command line.
#
# EXIT CODES: 0 = done or clean deadline stop (chain resubmits until
#   TRAINING_DONE), 75 = retryable on another node (GPU busy, /tmp full, HF
#   unreachable), 2 = configuration error, other = trainer failure.
#
# Required env: GLR_CONFIG, GLR_SEED, GLR_RUN_NAME
# Optional env:
#   GLR_CODE_MODE home|clone, GLR_REPO_DIR, GLR_REPO_URL, GLR_UV_VERSION
#   GLR_HF_REPO (default Helain/gated-lora-experiments)
#   GLR_RESUME auto (default) | none | <explicit path>
#   GLR_HF_TOKEN_FILE, GLR_WANDB_MODE (default disabled)
#   GLR_MIN_TMP_GB (10), GLR_HF_CACHE_CAP_GB (25), GLR_GPU_BUSY_MB (500)
#   GLR_DEADLINE_MARGIN (600), GLR_KEEP_LOCAL_HOURS (24)
#
# This file doubles as a function library: `GLR_LIB_ONLY=1 source train.sbatch`
# defines the glr_* functions and returns (used by train_packed.sbatch).

# shellcheck disable=SC2317  # trap handlers are invoked indirectly
#SBATCH --job-name=glr
#SBATCH --output=/tmp/glr-%j.out
#SBATCH --error=/tmp/glr-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=42000M  # 3 jobs per rtx6000 node (128520 MB); "42G" = 43008 MB would not fit
# --partition / --exclude / --job-name / --dependency are set by chain_jobs.sh.

# =========================== library ===================================

GLR_TMP="/tmp"
GLR_HF_CACHE="$GLR_TMP/glr_hf_cache"
GLR_UV_CACHE="$GLR_TMP/glr_uv_cache"
GLR_SBATCH_TIME_SECONDS=14400   # keep in sync with #SBATCH --time

glr_log() { echo "[$(date +%H:%M:%S)] $*"; }

# "[D-]HH:MM:SS" | "MM:SS" | "MM" -> seconds on stdout; non-zero if unparsable.
glr_parse_slurm_time() {
    local t="$1" d=0 h=0 m=0 s=0
    local -a p
    [[ "$t" =~ ^([0-9]+-)?[0-9]+(:[0-9]+){0,2}$ ]] || return 1
    if [[ "$t" == *-* ]]; then
        d="${t%%-*}"; t="${t#*-}"
        IFS=: read -r -a p <<<"$t"
        h="${p[0]:-0}"; m="${p[1]:-0}"; s="${p[2]:-0}"
    else
        IFS=: read -r -a p <<<"$t"
        case ${#p[@]} in
            1) m="${p[0]}" ;;
            2) m="${p[0]}"; s="${p[1]}" ;;
            3) h="${p[0]}"; m="${p[1]}"; s="${p[2]}" ;;
        esac
    fi
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

# Exports GLR_DEADLINE (kept if already a valid integer, e.g. packed child).
glr_compute_deadline() {
    local now end="" left limit src
    now=$(date +%s)
    if [[ "${GLR_DEADLINE:-}" =~ ^[0-9]+$ ]]; then
        glr_log "[deadline] inherited GLR_DEADLINE=$GLR_DEADLINE"
        return 0
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        left=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | head -n1 | tr -d " ") || left=""
        if left=$(glr_parse_slurm_time "$left"); then
            end=$(( now + left )); src="squeue %L"
        elif [[ "${SLURM_JOB_END_TIME:-}" =~ ^[0-9]+$ ]]; then
            end="$SLURM_JOB_END_TIME"; src="SLURM_JOB_END_TIME"
        else
            limit=$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | head -n1 | tr -d " ") || limit=""
            if limit=$(glr_parse_slurm_time "$limit"); then
                local start="${SLURM_JOB_START_TIME:-$now}"
                [[ "$start" =~ ^[0-9]+$ ]] || start="$now"
                end=$(( start + limit )); src="squeue %l"
            fi
        fi
    fi
    if [[ -z "$end" ]]; then
        end=$(( now + GLR_SBATCH_TIME_SECONDS )); src="sbatch default"
    fi
    export GLR_DEADLINE=$(( end - ${GLR_DEADLINE_MARGIN:-600} ))
    glr_log "[deadline] GLR_DEADLINE=$GLR_DEADLINE ($(date -d "@$GLR_DEADLINE" -Iseconds), from $src, $(( GLR_DEADLINE - now ))s usable)"
}

# Prints the token found in file $1 (raw token line or HF_TOKEN=... line).
glr_read_token_file() {
    local f="$1" line
    [[ -r "$f" ]] || return 1
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        if [[ "$line" == *HF_TOKEN=* ]]; then
            line="${line#*HF_TOKEN=}"
        elif [[ "$line" == *=* ]]; then
            continue
        fi
        line=$(printf "%s" "$line" | tr -d "\"' \t\r")
        if [[ -n "$line" ]]; then
            printf "%s" "$line"
            return 0
        fi
    done < "$f"
    return 1
}

glr_load_token() {
    local f="${GLR_HF_TOKEN_FILE:-$HOME/.hf_token}" tok="" perms
    if tok=$(glr_read_token_file "$f"); then
        perms=$(stat -c %a "$f" 2>/dev/null || echo "?")
        [[ "$perms" =~ ^[67]00$ ]] || glr_log "[token] WARNING: $f has mode $perms (chmod 600 recommended)"
        glr_log "[token] loaded from $f"
    elif tok=$(glr_read_token_file "${GLR_REPO_DIR:-$HOME/GatedLoraProject/gated-lora-research-paper-001}/.env"); then
        glr_log "[token] loaded from repo .env"
    elif [[ -n "${HF_TOKEN:-}" ]]; then
        tok="$HF_TOKEN"
        glr_log "[token] using inherited HF_TOKEN env var"
    else
        glr_log "[token] FATAL: no HF token (file $f, repo .env, env HF_TOKEN)" >&2
        exit 2
    fi
    export HF_TOKEN="$tok"
    glr_check_token
}

# Fail fast on a token HF rejects (401/403): exit 78 stops the chain instead of training for
# hours and then retrying the final upload until the deadline. Network errors only warn.
# The token goes through curl's stdin config, never argv (visible in ps on shared nodes).
glr_check_token() {
    local code
    command -v curl >/dev/null 2>&1 || { glr_log "[token] curl missing: token not checked"; return 0; }
    code=$(printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" | \
        curl -s -o /dev/null -w "%{http_code}" -m 30 -K - https://huggingface.co/api/whoami-v2) || true
    case "$code" in
        200) glr_log "[token] valid (HF whoami 200)" ;;
        401|403)
            glr_log "[token] FATAL: HF rejects the token (HTTP $code): revoked or expired; put a valid one in ${GLR_HF_TOKEN_FILE:-$HOME/.hf_token}" >&2
            exit 78 ;;
        *) glr_log "[token] WARNING: token check inconclusive (HTTP ${code:-none}), continuing" ;;
    esac
}

# Live job ids (one per line). Non-zero if squeue fails or looks wrong.
glr_live_ids() {
    local ids
    ids=$(squeue -h -o %i 2>/dev/null) || return 1
    [[ -n "$ids" ]] || return 1
    if [[ -n "${SLURM_JOB_ID:-}" ]] && ! grep -qx "$SLURM_JOB_ID" <<<"$ids"; then
        return 1   # our own job must be listed, otherwise do not trust it
    fi
    printf "%s\n" "$ids"
}

# Delete stale /tmp/glr-<jobid>* entries (ours, job no longer in squeue).
glr_hygiene() {
    local ids d base id keep_min
    keep_min=$(( ${GLR_KEEP_LOCAL_HOURS:-24} * 60 ))
    if ! ids=$(glr_live_ids); then
        glr_log "[hygiene] squeue unavailable or inconsistent: skipping cleanup"
        return 0
    fi
    for d in "$GLR_TMP"/glr-[0-9]*; do
        [[ -e "$d" && -O "$d" ]] || continue
        base="${d##*/}"; id="${base#glr-}"; id="${id%%[.-]*}"
        [[ "$id" =~ ^[0-9]+$ ]] || continue
        [[ "$id" == "${SLURM_JOB_ID:-}" ]] && continue
        grep -qx "$id" <<<"$ids" && continue
        if [[ -d "$d" ]]; then
            if find "$d" -maxdepth 5 -type d -path "*/outputs/*/latest" 2>/dev/null | grep -q . \
               && find "$d" -maxdepth 0 -mmin "-$keep_min" 2>/dev/null | grep -q .; then
                glr_log "[hygiene] keeping $d (preserved latest/, < ${GLR_KEEP_LOCAL_HOURS:-24}h old)"
                continue
            fi
            glr_log "[hygiene] removing stale dir $d ($(du -sh "$d" 2>/dev/null | cut -f1))"
            rm -rf -- "$d"
        else
            rm -f -- "$d"
        fi
    done
}

# Take a shared lock on the HF cache for the whole job (fd 7, inherited by
# children). Eviction needs the exclusive lock, i.e. NO other glr job on the
# node holds it: models are never deleted under a running job.
glr_cache_lock_and_evict() {
    local cap_kb total_kb victim vkb
    mkdir -p "$GLR_HF_CACHE"
    chmod 700 "$GLR_HF_CACHE" 2>/dev/null || true
    # Legacy jobs called login(), which wrote the token into the shared cache.
    rm -f "$GLR_HF_CACHE/token" "$GLR_HF_CACHE/stored_tokens" 2>/dev/null || true
    exec 7>"$GLR_HF_CACHE/.glr_cache.lock"
    if flock -x -n 7; then
        cap_kb=$(( ${GLR_HF_CACHE_CAP_GB:-25} * 1024 * 1024 ))
        total_kb=$(du -sk "$GLR_HF_CACHE" 2>/dev/null | cut -f1)
        glr_log "[cache] $GLR_HF_CACHE = $(( total_kb / 1024 )) MB (cap ${GLR_HF_CACHE_CAP_GB:-25} GB)"
        while (( total_kb > cap_kb )); do
            # LRU key = newest access/modify time of any file in the entry.
            victim=$(
                for e in "$GLR_HF_CACHE"/hub/models--* "$GLR_HF_CACHE"/hub/datasets--*; do
                    [[ -d "$e" ]] || continue
                    printf "%s %s\n" "$(find "$e" -type f -printf "%A@\n%T@\n" 2>/dev/null | sort -n | tail -n1)" "$e"
                done | sort -n | head -n1 | cut -d" " -f2-
            )
            [[ -n "$victim" && -d "$victim" ]] || break
            vkb=$(du -sk "$victim" 2>/dev/null | cut -f1)
            glr_log "[cache] evicting LRU $victim ($(( vkb / 1024 )) MB)"
            rm -rf -- "$victim"
            total_kb=$(( total_kb - vkb ))
        done
        flock -s 7
    else
        glr_log "[cache] another glr job holds the cache: no eviction"
        flock -s -w 900 7 || glr_log "[cache] WARNING: shared lock timeout, continuing"
    fi
}

glr_disk_check() {
    local avail_kb min_kb
    min_kb=$(( ${GLR_MIN_TMP_GB:-10} * 1024 * 1024 ))
    avail_kb=$(df -Pk "$GLR_TMP" | awk "NR==2 {print \$4}")
    glr_log "[disk] $GLR_TMP free: $(( avail_kb / 1024 )) MB (min ${GLR_MIN_TMP_GB:-10} GB)"
    if (( avail_kb < min_kb )); then
        glr_log "[disk] not enough space on $GLR_TMP: exit 75" >&2
        df -h "$GLR_TMP" >&2
        exit 75
    fi
}

# Pick a free GPU. GPUs are not isolated (shards), so the allocated card may
# be busy; we may switch to another idle card of the node. Claims are written
# to $WORK/gpu_claim and respected by other glr jobs on the node.
glr_gpu_select() {
    local busy_mb="${GLR_GPU_BUSY_MB:-500}" q alloc ids claimed="" d id
    local a_idx="" a_uuid="" a_used="" best_uuid="" best_idx="" best_used=""
    local idx uuid used attempt
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        glr_log "[gpu] nvidia-smi not found: skipping GPU check"
        return 0
    fi
    exec 8>"$GLR_TMP/glr_gpu.lock"
    flock -w 120 8 || glr_log "[gpu] WARNING: gpu lock timeout, continuing"
    if ids=$(glr_live_ids); then
        for d in "$GLR_TMP"/glr-[0-9]*/gpu_claim; do
            [[ -f "$d" ]] || continue
            id="${d#"$GLR_TMP"/glr-}"; id="${id%%/*}"
            [[ "$id" == "${SLURM_JOB_ID:-}" ]] && continue
            grep -qx "$id" <<<"$ids" && claimed+=" $(cat "$d" 2>/dev/null)"
        done
    fi
    alloc="${CUDA_VISIBLE_DEVICES:-}"; alloc="${alloc%%,*}"
    for attempt in 1 2; do
        q=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null) || {
            glr_log "[gpu] nvidia-smi query failed: skipping GPU check"; exec 8>&-; return 0; }
        a_idx=""; best_uuid=""; best_used=""
        while IFS=, read -r idx uuid used; do
            idx="${idx// /}"; uuid="${uuid// /}"; used="${used// /}"
            [[ "$used" =~ ^[0-9]+$ ]] || continue
            if [[ "$alloc" == "$idx" || "$alloc" == "$uuid" ]]; then
                a_idx="$idx"; a_uuid="$uuid"; a_used="$used"
            fi
            [[ " $claimed " == *" $uuid "* ]] && continue
            if (( used < busy_mb )) && { [[ -z "$best_used" ]] || (( used < best_used )); }; then
                best_uuid="$uuid"; best_idx="$idx"; best_used="$used"
            fi
        done <<<"$q"
        if [[ -n "$a_idx" ]] && (( a_used < busy_mb )) && [[ " $claimed " != *" $a_uuid "* ]]; then
            glr_log "[gpu] allocated GPU $a_idx ($a_uuid) is idle: ${a_used} MB used"
            best_uuid="$a_uuid"; best_idx="$a_idx"; best_used="$a_used"
            break
        fi
        [[ -n "$best_uuid" || $attempt -eq 2 ]] && break
        glr_log "[gpu] no idle GPU yet, re-checking in 15 s"; sleep 15
    done
    if [[ -z "$best_uuid" ]]; then
        glr_log "[gpu] allocated GPU ${a_idx:-?} busy (${a_used:-?} MB) and no other idle GPU: exit 75" >&2
        echo "$q" >&2
        exec 8>&-
        exit 75
    fi
    if [[ "$best_idx" != "$a_idx" ]]; then
        glr_log "[gpu] SWITCH: allocated GPU ${a_idx:-?} busy (${a_used:-?} MB used) -> using GPU $best_idx ($best_uuid, ${best_used} MB used)"
    fi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$best_uuid"
    echo "$best_uuid" > "$WORK/gpu_claim"
    exec 8>&-
}

# Job-level setup shared by train.sbatch and train_packed.sbatch.
# Requires WORK to be set.
glr_job_setup() {
    umask 077
    glr_compute_deadline
    glr_load_token
    mkdir -p "$WORK"
    glr_hygiene
    glr_cache_lock_and_evict
    glr_disk_check
    glr_gpu_select
}

# Upload files to <hf_repo>/<run>/slurm_logs/. Args: python run file...
glr_upload_logs() {
    local py="$1" run="$2"; shift 2
    "$py" -c "import huggingface_hub" 2>/dev/null || { glr_log "[logs] huggingface_hub unavailable: logs not uploaded"; return 0; }
    GLR_UP_RUN="$run" GLR_UP_REPO="${GLR_HF_REPO:-Helain/gated-lora-experiments}" \
        timeout 180 "$py" - "$@" <<'PYEOF' || true
import os, sys
from huggingface_hub import CommitOperationAdd, HfApi
api = HfApi()  # token from HF_TOKEN env; never login() (it writes the token to disk)
run, repo = os.environ["GLR_UP_RUN"], os.environ["GLR_UP_REPO"]
# ONE commit for all log files (HF: 128 commits/hour per repo, shared by every run)
ops = [CommitOperationAdd(path_in_repo=f"{run}/slurm_logs/{os.path.basename(p)}", path_or_fileobj=p)
       for p in sys.argv[1:] if os.path.isfile(p)]
if ops:
    try:
        api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                          commit_message=f"slurm logs {run}")
    except Exception as e:
        print(f"[logs] upload failed: {str(e).splitlines()[0][:200]}", flush=True)
PYEOF
}

# Early, dependency-free TRAINING_DONE probe (curl; token via stdin config).
glr_hf_done_probe() {
    local code url
    command -v curl >/dev/null 2>&1 || return 1
    url="https://huggingface.co/datasets/${GLR_HF_REPO:-Helain/gated-lora-experiments}/resolve/main/$1/TRAINING_DONE"
    code=$(printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" \
        | curl -s -K - -o /dev/null -I -m 30 -w "%{http_code}" "$url" 2>/dev/null) || return 1
    [[ "$code" == 200 || "$code" == 302 || "$code" == 307 ]]
}

if [[ "${GLR_LIB_ONLY:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

# =========================== main ======================================

set -euo pipefail
umask 077

: "${GLR_CONFIG:?GLR_CONFIG must be set}"
: "${GLR_SEED:?GLR_SEED must be set}"
: "${GLR_RUN_NAME:?GLR_RUN_NAME must be set}"

CODE_MODE="${GLR_CODE_MODE:-home}"
export GLR_REPO_DIR="${GLR_REPO_DIR:-$HOME/GatedLoraProject/gated-lora-research-paper-001}"
REPO_URL="${GLR_REPO_URL:-https://github.com/L1ZGitHub/gated-lora-research-paper-001.git}"
export GLR_HF_REPO="${GLR_HF_REPO:-Helain/gated-lora-experiments}"
UV_VERSION="${GLR_UV_VERSION:-0.5.18}"
RESUME="${GLR_RESUME:-auto}"
PACKED_CHILD="${GLR_PACKED_CHILD:-0}"

JOB_ID="${SLURM_JOB_ID:-local-$$}"
WORK="${GLR_WORK:-$GLR_TMP/glr-$JOB_ID}"
OUTPUT_DIR="$WORK/outputs/$GLR_RUN_NAME"
read -r -a LOG_FILES <<<"${GLR_LOG_FILES:-$GLR_TMP/glr-$JOB_ID.out $GLR_TMP/glr-$JOB_ID.err}"
PYBIN=""
[[ "$CODE_MODE" == "home" ]] && PYBIN="$GLR_REPO_DIR/.venv/bin/python"
TRAIN_PID=""

glr_on_exit() {
    local rc=$?
    set +e
    trap - TERM INT
    # Done only when TRAINING_DONE is ON THE HUB: a local TRAINING_DONE whose final commit
    # failed (e.g. HF 429) must keep the outputs on the node for recovery/resume.
    local done_hub=0
    if [[ -f "$OUTPUT_DIR/TRAINING_DONE" ]] && glr_hf_done_probe "$GLR_RUN_NAME"; then
        done_hub=1
    elif [[ -f "$OUTPUT_DIR/TRAINING_DONE" ]]; then
        glr_log "[exit] TRAINING_DONE is local only (push failed): keeping $OUTPUT_DIR" >&2
        rm -f "$OUTPUT_DIR/TRAINING_DONE"  # next slice resumes latest/ and redoes the final push
    fi
    if [[ -d "$OUTPUT_DIR/latest" ]] && (( ! done_hub )); then
        # Keep outputs/ (latest/, best_model/) as a local resume candidate;
        # drop everything else (clone, venv, wandb, partial downloads).
        find "$WORK" -mindepth 1 -maxdepth 1 ! -name outputs ! -name gpu_claim -exec rm -rf {} + 2>/dev/null
        rm -rf "$OUTPUT_DIR"/*.tmp "$OUTPUT_DIR/.hf_pull" 2>/dev/null
        rm -f "$WORK/gpu_claim"
        touch "$WORK"
        glr_log "[exit] preserved $OUTPUT_DIR ($(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)) as local resume candidate"
    fi
    local py="${PYBIN:-python3}"
    [[ -x "$py" ]] || py=python3
    glr_upload_logs "$py" "$GLR_RUN_NAME" "${LOG_FILES[@]}"
    if [[ ! -d "$OUTPUT_DIR/latest" ]] || (( done_hub )); then
        rm -rf "$WORK" 2>/dev/null
    fi
    glr_log "[exit] code=$rc at $(date -Iseconds)"
    exit "$rc"
}
glr_on_term() {
    glr_log "[signal] TERM received, forwarding to trainer"
    [[ -n "$TRAIN_PID" ]] && kill -TERM "$TRAIN_PID" 2>/dev/null
}
trap glr_on_exit EXIT
trap glr_on_term TERM INT

if [[ "$PACKED_CHILD" == "1" ]]; then
    mkdir -p "$WORK"
    [[ -n "${HF_TOKEN:-}" ]] || glr_load_token
    glr_compute_deadline
else
    glr_job_setup
fi

echo "==================================================================="
echo "SLURM job $JOB_ID on $(hostname) at $(date -Iseconds)"
echo "  Mode     : $CODE_MODE$([[ "$PACKED_CHILD" == 1 ]] && echo " (packed child)")"
echo "  WORK     : $WORK"
echo "  Config   : $GLR_CONFIG"
echo "  Seed     : $GLR_SEED"
echo "  Run      : $GLR_RUN_NAME"
echo "  Resume   : $RESUME"
echo "  HF repo  : $GLR_HF_REPO"
echo "  GPU      : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Deadline : $GLR_DEADLINE ($(date -d "@$GLR_DEADLINE" -Iseconds))"
echo "==================================================================="

if glr_hf_done_probe "$GLR_RUN_NAME"; then
    glr_log "[done] $GLR_RUN_NAME/TRAINING_DONE already on HF: nothing to do"
    exit 0
fi

# ----------------------- env -------------------------------
export HF_HOME="$GLR_HF_CACHE"
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DIR="$WORK/wandb"
export WANDB_MODE="${GLR_WANDB_MODE:-disabled}"
export WANDB_DISABLE_SERVICE=true
export PYTHONUNBUFFERED=1
export GLR_RUN_NAME GLR_HF_PATH="$GLR_RUN_NAME" GLR_OUTPUT_DIR="$OUTPUT_DIR"
mkdir -p "$HF_HOME" "$WANDB_DIR" "$OUTPUT_DIR"
cd "$WORK"

# ----------------------- code + python ---------------------
if [[ "$CODE_MODE" == "home" ]]; then
    if [[ ! -x "$PYBIN" ]]; then
        glr_log "FATAL: $PYBIN not found. Home mount broken? Use GLR_CODE_MODE=clone" >&2
        exit 75
    fi
    CODE_DIR="$GLR_REPO_DIR"
else
    export HOME="$WORK"   # no NFS home assumption in clone mode
    export UV_CACHE_DIR="$GLR_UV_CACHE"
    mkdir -p "$UV_CACHE_DIR"
    glr_log "[stage-in] git clone..."
    git clone --depth=1 --quiet "$REPO_URL" "$WORK/glr"
    glr_log "[stage-in] downloading uv $UV_VERSION..."
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar xz -C "$WORK"
    UV="$WORK/uv-x86_64-unknown-linux-gnu/uv"
    glr_log "[stage-in] uv sync (shared cache $UV_CACHE_DIR)..."
    (cd "$WORK/glr" && "$UV" sync --no-dev)
    CODE_DIR="$WORK/glr"
    PYBIN="$WORK/glr/.venv/bin/python"
fi
glr_log "[env] python: $PYBIN"
"$PYBIN" -c "import torch; print('[env] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ----------------------- resume ----------------------------
RESUME_ARGS=()
if [[ "$RESUME" == "none" ]]; then
    glr_log "[resume] GLR_RESUME=none: fresh start (HF latest/ of this run will be overwritten)"
elif [[ "$RESUME" != "auto" ]]; then
    RESUME_ARGS=(--resume "$RESUME")
else
    LIVE_IDS=$(glr_live_ids | tr "\n" " ") || LIVE_IDS="${SLURM_JOB_ID:-}"
    rrc=0
    GLR_WORK_DIR="$WORK" GLR_LIVE_IDS="$LIVE_IDS" "$PYBIN" - <<'PYEOF' || rrc=$?
import glob, json, os, shutil, sys, tempfile, time
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

run, repo = os.environ["GLR_RUN_NAME"], os.environ["GLR_HF_REPO"]
out, work = os.environ["GLR_OUTPUT_DIR"], os.path.realpath(os.environ["GLR_WORK_DIR"])
live = set(os.environ.get("GLR_LIVE_IDS", "").split())
api = HfApi()

def log(msg):
    print(f"[resume] {msg}", flush=True)

def step_of(d):
    try:
        with open(os.path.join(d, "training_state.json")) as f:
            return int(json.load(f)["global_step"])
    except Exception:
        return None

def hf(fn, what):
    last = None
    for i in range(4):
        try:
            return fn()
        except Exception as e:
            last = e
            log(f"HF error on {what} (try {i + 1}/4): {e}")
            time.sleep(min(60, 10 * 2 ** i))
    log(f"HF unreachable for {what}: exit 75 (refusing a silent fresh start)")
    sys.exit(75)

# Local candidates: preserved outputs of earlier slices on this node.
local = []
for pat in (f"/tmp/glr-*/outputs/{run}/latest", f"/tmp/glr-*/slot*/outputs/{run}/latest"):
    for p in glob.glob(pat):
        rp = os.path.realpath(p)
        if rp.startswith(work + os.sep) or os.stat(rp).st_uid != os.getuid():
            continue
        jid = rp[len("/tmp/glr-"):].split("/")[0]
        if jid in live:
            log(f"WARNING: live job {jid} holds {rp}; double submission? ignoring it")
            continue
        s = step_of(rp)
        if s is None:
            log(f"ignoring {rp} (no readable training_state.json)")
            continue
        local.append((s, rp))
        log(f"local candidate step={s}: {rp}")

if hf(lambda: api.file_exists(repo, f"{run}/TRAINING_DONE", repo_type="dataset"), "TRAINING_DONE"):
    log("TRAINING_DONE on HF: nothing to do")
    sys.exit(10)

hf_step = None
if hf(lambda: api.file_exists(repo, f"{run}/latest/training_state.json", repo_type="dataset"), "latest state"):
    tmp = tempfile.mkdtemp(dir=out)
    p = hf(lambda: hf_hub_download(repo, f"{run}/latest/training_state.json",
                                   repo_type="dataset", local_dir=tmp), "latest state download")
    hf_step = step_of(os.path.dirname(p))
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"HF candidate step={hf_step}")

best_local = max(local) if local else None
target = os.path.join(out, "latest")
if hf_step is None and best_local is None:
    log("no prior state: fresh start")
    sys.exit(0)
if best_local is not None and (hf_step is None or best_local[0] > hf_step):
    log(f"using LOCAL candidate step={best_local[0]} ({best_local[1]})")
    shutil.move(best_local[1], target)
    used = best_local[1]
    src_best = os.path.join(os.path.dirname(used), "best_model")
    if os.path.isdir(src_best):  # needed by the best-checkpoint test eval at the end
        shutil.move(src_best, os.path.join(out, "best_model"))
        log(f"carried best_model/ from {src_best}")
else:
    pull = os.path.join(out, ".hf_pull")
    log(f"pulling HF {run}/latest/ (step={hf_step})")
    hf(lambda: snapshot_download(repo, repo_type="dataset",
                                 allow_patterns=[f"{run}/latest/*", f"{run}/best_model/*"],
                                 local_dir=pull), "latest download")
    shutil.move(os.path.join(pull, run, "latest"), target)
    if os.path.isdir(os.path.join(pull, run, "best_model")):
        shutil.move(os.path.join(pull, run, "best_model"), os.path.join(out, "best_model"))
        log("pulled best_model/ from HF")
    shutil.rmtree(pull, ignore_errors=True)
    got = step_of(target)
    if got != hf_step:
        log(f"WARNING: downloaded step {got} != probed step {hf_step} (concurrent push?)")
    used = None
for s, p in local:  # superseded local copies
    if p != used and os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
        log(f"removed superseded local candidate {p}")
log(f"resume from {target} step={step_of(target)}")
PYEOF
    case "$rrc" in
        0)  [[ -d "$OUTPUT_DIR/latest" ]] && RESUME_ARGS=(--resume "$OUTPUT_DIR/latest") ;;
        10) exit 0 ;;
        75) exit 75 ;;
        *)  glr_log "[resume] resume helper failed (rc=$rrc)" >&2; exit "$rrc" ;;
    esac
fi

# ----------------------- train -----------------------------
now=$(date +%s)
budget=$(( GLR_DEADLINE - now ))
if [[ "${GLR_MAX_RUNTIME_SECONDS:-}" =~ ^[0-9]+$ ]] && (( GLR_MAX_RUNTIME_SECONDS < budget )); then
    budget="$GLR_MAX_RUNTIME_SECONDS"
fi
export GLR_MAX_RUNTIME_SECONDS="$budget"
if (( budget < 600 )); then
    glr_log "[train] only ${budget}s before deadline: exit 75" >&2
    exit 75
fi

cd "$CODE_DIR"
glr_log "[train] starting (budget ${budget}s) resume=${RESUME_ARGS[*]:-<none>}"
"$PYBIN" -m gated_lora.cli \
    --config "$GLR_CONFIG" \
    --seed "$GLR_SEED" \
    --output-dir "$OUTPUT_DIR" \
    --analyze-routing \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} &
TRAIN_PID=$!
rc=0
wait "$TRAIN_PID" || rc=$?
while (( rc > 128 )) && kill -0 "$TRAIN_PID" 2>/dev/null; do
    rc=0
    wait "$TRAIN_PID" || rc=$?
done
TRAIN_PID=""

if (( rc != 0 )); then
    glr_log "[train] trainer exited with rc=$rc" >&2
    exit "$rc"
fi
if [[ -f "$OUTPUT_DIR/TRAINING_DONE" ]]; then
    glr_log "[train] TRAINING_DONE written: run complete"
else
    glr_log "[train] clean stop before deadline (chain will resubmit)"
fi
exit 0
