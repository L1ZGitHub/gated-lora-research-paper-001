#!/bin/bash
# Chain <= 4h SLURM slices of one run until TRAINING_DONE appears on HF Hub.
# Run on the SLURM controller / login node (nash). State of truth: HF Hub.
#
# Usage:
#   bash scripts/slurm/chain_jobs.sh --config <yaml> --seed <n> \
#       [--max-jobs 10] [--partition rtx6000] [--exclude turing-2,turing-3] \
#       [--run-name <override>] [--poll-seconds 120]
#
# Behaviour:
#   - Job name = run name. A flock on logs/chains/<run>.lock plus a
#     `squeue -u $USER -n <run>` check refuse double submission.
#   - The next slice is pre-submitted with --dependency=afterany:<current>
#     so it queues while the current one runs. It is scancel-ed when the run
#     is done or the chain stops.
#   - Slice exit 75 (GPU busy, /tmp full, HF unreachable): the host is added
#     to the exclude list; if the pre-submitted slice is still pending it is
#     replaced by one that excludes that host. Exit-75 slices do not count as
#     "no progress".
#   - Stops after 2 consecutive slices without global_step progress on HF
#     (<run>/latest/training_state.json).
#   - Polling is cheap: one squeue call every --poll-seconds; HF is queried
#     only when a slice ends.
#
# HF token: read from ${GLR_HF_TOKEN_FILE:-$HOME/.hf_token} (or HF_TOKEN= in
# <repo>/.env) for the HF polling only. It is NOT passed to sbatch: the job
# reads the same file itself (no token in any argv / ps output). Set
# GLR_TOKEN_VIA_ENV=1 to propagate it through the sbatch environment instead
# (clone mode with a broken home mount); still never on the command line.
#
# Env: GLR_EXCLUDE_NODES (default none), GLR_HF_REPO, GLR_CODE_MODE,
#      GLR_MAX_RUNTIME_SECONDS (optional cap; the job derives GLR_DEADLINE).
#
# Exit codes: 0 done, 1 bad args / submit failure, 2 budget exhausted,
#             3 no progress, 4 another chain already owns this run,
#             5 the trainer rejected the config (cli exit 78).

set -euo pipefail

CONFIG=""
SEED=""
MAX_JOBS=10
PARTITION="rtx6000"
EXCLUDE="${GLR_EXCLUDE_NODES:-}"
RUN_NAME=""
POLL="${GLR_POLL_SECONDS:-120}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_REPO="${GLR_HF_REPO:-Helain/gated-lora-experiments}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --max-jobs)     MAX_JOBS="$2"; shift 2 ;;
        --partition)    PARTITION="$2"; shift 2 ;;
        --exclude)      EXCLUDE="$2"; shift 2 ;;
        --run-name)     RUN_NAME="$2"; shift 2 ;;
        --poll-seconds) POLL="$2"; shift 2 ;;
        --nodelist)     echo "ERROR: --nodelist was removed; use --exclude <nodes>" >&2; exit 1 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$CONFIG" ]] && { echo "ERROR: --config required" >&2; exit 1; }
[[ -z "$SEED"   ]] && { echo "ERROR: --seed required"   >&2; exit 1; }
[[ -f "$PROJECT_ROOT/$CONFIG" || -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -z "$RUN_NAME" ]] && RUN_NAME="$(basename "$CONFIG" .yaml)_seed${SEED}"

log() { echo "[chain $(date -Iseconds)] $*"; }

# ---- token (never exported, never in argv) ---------------------------
read_token_file() {
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
        [[ -n "$line" ]] && { printf "%s" "$line"; return 0; }
    done < "$f"
    return 1
}
TOKEN="$(read_token_file "${GLR_HF_TOKEN_FILE:-$HOME/.hf_token}" \
      || read_token_file "$PROJECT_ROOT/.env" || printf "%s" "${HF_TOKEN:-}")"
if [[ -z "$TOKEN" ]]; then
    echo "ERROR: no HF token (${GLR_HF_TOKEN_FILE:-$HOME/.hf_token}, $PROJECT_ROOT/.env, env HF_TOKEN)" >&2
    exit 1
fi

# ---- single-owner guard ------------------------------------------------
mkdir -p "$PROJECT_ROOT/logs/chains"
exec 9>"$PROJECT_ROOT/logs/chains/${RUN_NAME}.lock"
if ! flock -n 9; then
    echo "ERROR: another chain_jobs.sh holds logs/chains/${RUN_NAME}.lock" >&2
    exit 4
fi
existing=$(squeue -h -u "$USER" -n "$RUN_NAME" -o "%i %T" 2>&1) || {
    echo "ERROR: squeue failed: $existing" >&2; exit 1; }
if [[ -n "$existing" ]]; then
    echo "ERROR: jobs named $RUN_NAME already queued/running:" >&2
    echo "$existing" >&2
    exit 4
fi

# ---- python with huggingface_hub (login node) ---------------------------
PY=""
for candidate in "$PROJECT_ROOT/.venv/bin/python" "$(command -v python3 || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]] && "$candidate" -c "import huggingface_hub" 2>/dev/null; then
        PY="$candidate"; break
    fi
done
[[ -n "$PY" ]] || { echo "ERROR: no Python with huggingface_hub (cd $PROJECT_ROOT && uv sync --no-dev)" >&2; exit 1; }

echo "==================================================================="
echo "Chain orchestrator"
echo "  Config:    $CONFIG"
echo "  Seed:      $SEED"
echo "  Run name:  $RUN_NAME (= SLURM job name)"
echo "  HF repo:   $HF_REPO"
echo "  Partition: $PARTITION"
echo "  Exclude:   ${EXCLUDE:-<none>}"
echo "  Max jobs:  $MAX_JOBS   Poll: ${POLL}s"
echo "  Python:    $PY"
echo "==================================================================="

# Prints "DONE", "STEP <n>" or "NONE"; rc 2 on HF error.
hf_state() {
    HF_TOKEN="$TOKEN" HF_HUB_DISABLE_TELEMETRY=1 GLR_Q_REPO="$HF_REPO" GLR_Q_RUN="$RUN_NAME" \
        timeout 300 "$PY" - <<'PYEOF'
import json, os, sys, tempfile
from huggingface_hub import HfApi, hf_hub_download
repo, run = os.environ["GLR_Q_REPO"], os.environ["GLR_Q_RUN"]
api = HfApi()
try:
    if api.file_exists(repo, f"{run}/TRAINING_DONE", repo_type="dataset"):
        print("DONE"); sys.exit(0)
    if not api.file_exists(repo, f"{run}/latest/training_state.json", repo_type="dataset"):
        print("NONE"); sys.exit(0)
    with tempfile.TemporaryDirectory() as d:
        p = hf_hub_download(repo, f"{run}/latest/training_state.json", repo_type="dataset", local_dir=d)
        with open(p) as f:
            print("STEP", int(json.load(f)["global_step"]))
except Exception as e:
    print(f"HF check failed: {e}", file=sys.stderr); sys.exit(2)
PYEOF
}
hf_state_retry() {
    local out
    for _ in 1 2 3 4 5; do
        if out=$(hf_state); then echo "$out"; return 0; fi
        sleep 60
    done
    return 2
}
step_of() { [[ "$1" == STEP* ]] && echo "${1#STEP }" || echo -1; }

BAD_NODES=()   # hosts that returned 75 recently (last 3 kept)
SUBMITTED=0

exclude_arg() {
    local list="$EXCLUDE" n
    for n in ${BAD_NODES[@]+"${BAD_NODES[@]}"}; do
        [[ ",$list," == *",$n,"* ]] || list="${list:+$list,}$n"
    done
    [[ -n "$list" ]] && echo "--exclude=$list"
    return 0
}

# submit [dependency_job_id] -> sets SUBMIT_ID (call directly, not in $(...))
SUBMIT_ID=""
submit() {
    local dep="${1:-}" out ex
    SUBMIT_ID=""
    local -a args=(--parsable --partition="$PARTITION" --job-name="$RUN_NAME")
    ex=$(exclude_arg); [[ -n "$ex" ]] && args+=("$ex")
    [[ -n "$dep" ]] && args+=(--dependency="afterany:$dep")
    # Job variables go through the submit environment (sbatch exports ALL by default), NOT
    # --export=ALL,VAR=...: on ensicompute (2026-10-07) any --export=ALL,VAR=... job is
    # requeued + held with "user_env_retrieval_failed".
    local -a jobenv=(GLR_CONFIG="$CONFIG" GLR_SEED="$SEED" GLR_RUN_NAME="$RUN_NAME"
                     GLR_HF_REPO="$HF_REPO" GLR_RESUME=auto
                     GLR_CODE_MODE="${GLR_CODE_MODE:-home}" GLR_REPO_DIR="$PROJECT_ROOT")
    if [[ "${GLR_TOKEN_VIA_ENV:-0}" == "1" ]]; then
        out=$(cd "$PROJECT_ROOT" && env "${jobenv[@]}" HF_TOKEN="$TOKEN" sbatch "${args[@]}" scripts/slurm/train.sbatch)
    else
        out=$(cd "$PROJECT_ROOT" && env -u HF_TOKEN "${jobenv[@]}" sbatch "${args[@]}" scripts/slurm/train.sbatch)
    fi
    out="${out%%;*}"
    [[ "$out" =~ ^[0-9]+$ ]] || { echo "ERROR: cannot parse sbatch output: $out" >&2; return 1; }
    SUBMITTED=$(( SUBMITTED + 1 ))
    SUBMIT_ID="$out"
}

# Wait until job $1 leaves squeue; sets LAST_NODE.
LAST_NODE=""
wait_job() {
    local id="$1" line empty=0 state=""
    while (( empty < 3 )); do
        line=$(squeue -h -j "$id" -o "%T %N" 2>/dev/null | head -n1) || line=""
        if [[ -n "$line" ]]; then
            empty=0
            [[ "$state" != "${line%% *}" ]] && { state="${line%% *}"; log "job $id: $line"; }
            [[ -n "${line#* }" && "${line#* }" != "$line" ]] && LAST_NODE="${line#* }"
            sleep "$POLL"
        else
            empty=$(( empty + 1 ))
            (( empty < 3 )) && sleep 20
        fi
    done
    return 0
}

# Prints "STATE RC NODE" for a finished job (sacct, then scontrol fallback).
job_info() {
    local id="$1" line state ec rc sig node
    line=$(sacct -n -X -P -j "$id" -o State,ExitCode,NodeList 2>/dev/null | head -n1) || line=""
    if [[ -n "$line" ]]; then
        IFS="|" read -r state ec node <<<"$line"
    else
        line=$(scontrol show job -o "$id" 2>/dev/null) || line=""
        state=$(grep -o "JobState=[^ ]*" <<<"$line" | cut -d= -f2) || state=""
        ec=$(grep -o "ExitCode=[^ ]*" <<<"$line" | cut -d= -f2) || ec=""
        node=$(grep -o " NodeList=[^ ]*" <<<"$line" | cut -d= -f2) || node=""
    fi
    state="${state%% *}"
    rc="${ec%%:*}"; sig="${ec#*:}"
    [[ "$rc" =~ ^[0-9]+$ ]] || rc=-1
    [[ "$sig" =~ ^[0-9]+$ ]] && (( rc == 0 && sig > 0 )) && rc=$(( 128 + sig ))
    [[ -z "$node" || "$node" == "None" || "$node" == "(null)" ]] && node="$LAST_NODE"
    echo "${state:-UNKNOWN} $rc ${node:-unknown}"
}

cancel_if() {
    local id="$1" st
    [[ -n "$id" ]] || return 0
    st=$(squeue -h -j "$id" -o %T 2>/dev/null) || st=""
    if [[ -n "$st" ]]; then
        log "scancel $id ($st)"
        scancel "$id" 2>/dev/null || true
    fi
}

# ---- main loop -----------------------------------------------------------
init=$(hf_state_retry) || { echo "ERROR: HF unreachable at start" >&2; exit 1; }
if [[ "$init" == DONE ]]; then
    log "$RUN_NAME/TRAINING_DONE already on HF: nothing to do"
    exit 0
fi
last_step=$(step_of "$init")
log "initial HF state: $init"
noprog=0
NEXT=""

submit "" || exit 1
cur="$SUBMIT_ID"
log "submitted slice $cur ($SUBMITTED/$MAX_JOBS)"

while true; do
    if [[ -z "$NEXT" ]] && (( SUBMITTED < MAX_JOBS )); then
        NEXT=""; submit "$cur" && NEXT="$SUBMIT_ID"
        [[ -n "$NEXT" ]] && log "pre-submitted slice $NEXT (afterany:$cur) ($SUBMITTED/$MAX_JOBS)"
    fi

    LAST_NODE=""
    wait_job "$cur"
    read -r state rc node <<<"$(job_info "$cur")"
    log "slice $cur finished: state=$state rc=$rc node=$node"

    if ! hf=$(hf_state_retry); then
        log "WARNING: HF unreachable after slice $cur; continuing with the queued slice"
        hf="UNKNOWN"
    fi
    if [[ "$hf" == DONE ]]; then
        cancel_if "$NEXT"
        log "$RUN_NAME/TRAINING_DONE on HF: done after $SUBMITTED submitted slice(s)"
        exit 0
    fi

    if [[ "$rc" == 78 ]]; then
        cancel_if "$NEXT"
        log "ERROR: slice $cur rejected the config (exit 78): fix it, then relaunch" >&2
        exit 5
    fi

    if [[ "$rc" == 75 ]]; then
        if [[ "$node" != unknown ]]; then
            BAD_NODES+=("$node")
            (( ${#BAD_NODES[@]} > 3 )) && BAD_NODES=("${BAD_NODES[@]:1}")
        fi
        log "slice $cur asked for a retry (75) on $node; excluding: $(exclude_arg)"
        if [[ -n "$NEXT" ]] && [[ "$(squeue -h -j "$NEXT" -o %T 2>/dev/null)" == PENDING ]]; then
            scancel "$NEXT" 2>/dev/null || true
            SUBMITTED=$(( SUBMITTED - 1 ))
            NEXT=""; submit "" && NEXT="$SUBMIT_ID"
            [[ -n "$NEXT" ]] && log "replaced pending slice by $NEXT (with exclude)"
        fi
    elif [[ "$hf" != UNKNOWN ]]; then
        step=$(step_of "$hf")
        if (( step > last_step )); then
            log "progress: global_step $last_step -> $step"
            last_step=$step; noprog=0
        else
            noprog=$(( noprog + 1 ))
            log "WARNING: no global_step progress ($step) after slice $cur [$noprog/2]"
            if (( noprog >= 2 )); then
                cancel_if "$NEXT"
                log "ERROR: 2 consecutive slices without progress: stopping chain" >&2
                exit 3
            fi
        fi
    fi

    if [[ -z "$NEXT" ]]; then
        if (( SUBMITTED < MAX_JOBS )); then
            submit "" || exit 1
            NEXT="$SUBMIT_ID"
            log "submitted slice $NEXT ($SUBMITTED/$MAX_JOBS)"
        else
            break
        fi
    fi
    cur="$NEXT"; NEXT=""
done

if [[ "$(hf_state_retry || true)" == DONE ]]; then
    log "final TRAINING_DONE check OK"
    exit 0
fi
log "ERROR: budget exhausted ($MAX_JOBS slices) without TRAINING_DONE" >&2
exit 2
