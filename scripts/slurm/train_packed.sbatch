#!/bin/bash
# PACKED MODE: N (2-3) small-model runs as N separate processes under ONE
# gpu:1 job. Each run gets its own WORK dir (/tmp/glr-<jobid>/slot<k>), its
# own CPU subset (taskset + OMP_NUM_THREADS) and its own log
# (/tmp/glr-<jobid>-slot<k>.log, uploaded to <run>/slurm_logs/ on HF).
# Every slot resumes independently from its own latest/ (same logic as
# train.sbatch), so resubmitting the same packed job continues all runs.
#
# !! MEASURE PEAK VRAM FIRST !!
#   Run each config alone for a few hundred steps (train.sbatch) and record
#   torch.cuda.max_memory_reserved() + ~0.5 GB CUDA context per process.
#   Pack only if the SUM stays below ~20 GB (Quadro RTX 6000 = 22 GB, keep
#   headroom for eval batches and fragmentation). An OOM in one slot fails
#   that slot only; the job exit code reports it.
#
# Requirements: home code mode (each slot runs scripts/slurm/train.sbatch
# from the NFS checkout as a child process); chain_jobs.sh does NOT drive
# packed jobs: submit and resubmit them by hand until every run has
# TRAINING_DONE on HF.
#
# Usage (from the repo root on nash):
#   GLR_PACK="configs/a.yaml:42 configs/b.yaml:42 configs/c.yaml:42" \
#       sbatch --partition=rtx6000 --job-name=glr-pack scripts/slurm/train_packed.sbatch
#   (pass variables through the environment: --export=ALL,VAR=... jobs are held on ensicompute)
#   GLR_PACK entries: <config>:<seed>[:<run_name>]; run_name defaults to
#   <config basename>_seed<seed> (same as chain_jobs.sh).
#
# Exit code: 0 if every slot exited 0; else the first slot code that is
# neither 0 nor 75; else 75 (all failures retryable).

# shellcheck disable=SC2317,SC2094  # trap handlers; slot log is only written by the child
#SBATCH --job-name=glr-pack
#SBATCH --output=/tmp/glr-%j.out
#SBATCH --error=/tmp/glr-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB

set -euo pipefail
umask 077

: "${GLR_PACK:?GLR_PACK must be set, e.g. cfgA.yaml:42 cfgB.yaml:42}"
if [[ "${GLR_CODE_MODE:-home}" != "home" ]]; then
    echo "FATAL: packed mode requires GLR_CODE_MODE=home" >&2
    exit 2
fi
export GLR_CODE_MODE=home
export GLR_REPO_DIR="${GLR_REPO_DIR:-${SLURM_SUBMIT_DIR:-$HOME/GatedLoraProject/gated-lora-research-paper-001}}"
export GLR_HF_REPO="${GLR_HF_REPO:-Helain/gated-lora-experiments}"
TRAIN_SBATCH="$GLR_REPO_DIR/scripts/slurm/train.sbatch"
if [[ ! -r "$TRAIN_SBATCH" || ! -x "$GLR_REPO_DIR/.venv/bin/python" ]]; then
    echo "FATAL: $TRAIN_SBATCH or the repo .venv is missing (home mount?)" >&2
    exit 75
fi

# shellcheck source=train.sbatch
GLR_LIB_ONLY=1 source "$TRAIN_SBATCH"

JOB_ID="${SLURM_JOB_ID:-local-$$}"
WORK="$GLR_TMP/glr-$JOB_ID"

read -r -a SPECS <<<"$GLR_PACK"
N=${#SPECS[@]}
if (( N < 1 || N > 4 )); then
    echo "FATAL: GLR_PACK must list 1..4 runs (got $N)" >&2
    exit 2
fi
(( N > 3 )) && echo "WARNING: $N runs packed on one 22 GB GPU; did you measure peak VRAM?"

RUNS=(); CONFIGS=(); SEEDS=()
for spec in "${SPECS[@]}"; do
    IFS=: read -r cfg seed run <<<"$spec"
    if [[ -z "$cfg" || ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "FATAL: bad GLR_PACK entry: $spec (want config:seed[:run])" >&2
        exit 2
    fi
    [[ -z "${run:-}" ]] && run="$(basename "$cfg" .yaml)_seed${seed}"
    CONFIGS+=("$cfg"); SEEDS+=("$seed"); RUNS+=("$run")
done

PIDS=()
glr_pack_on_term() {
    glr_log "[pack] TERM received, forwarding to slots"
    local p
    for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -TERM "$p" 2>/dev/null || true; done
}
glr_pack_on_exit() {
    local rc=$? r
    set +e
    trap - TERM INT
    for r in "${RUNS[@]}"; do
        glr_upload_logs "$GLR_REPO_DIR/.venv/bin/python" "$r" "$GLR_TMP/glr-$JOB_ID.out" "$GLR_TMP/glr-$JOB_ID.err"
    done
    rm -f "$WORK/gpu_claim"
    rmdir "$WORK" 2>/dev/null   # slots that preserved latest/ keep it alive
    glr_log "[pack] exit code=$rc at $(date -Iseconds)"
    exit "$rc"
}
trap glr_pack_on_exit EXIT
trap glr_pack_on_term TERM INT

# Job-level checks once: deadline, token, /tmp hygiene, cache lock, df, GPU.
glr_job_setup

# ---- CPU split: allowed CPU list -> N contiguous chunks -------------------
expand_cpus() {
    local list="$1" part a b
    local -a out=()
    IFS=, read -r -a parts <<<"$list"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            a="${part%-*}"; b="${part#*-}"
            for (( c = a; c <= b; c++ )); do out+=("$c"); done
        else
            out+=("$part")
        fi
    done
    echo "${out[@]}"
}
ALLOWED=$(awk "/^Cpus_allowed_list/ {print \$2}" /proc/self/status)
read -r -a CPUS <<<"$(expand_cpus "$ALLOWED")"
NC=${#CPUS[@]}
PER=$(( NC / N ))
(( PER >= 1 )) || { echo "FATAL: $NC CPUs for $N runs" >&2; exit 2; }

echo "==================================================================="
echo "PACKED job $JOB_ID on $(hostname): $N runs, $NC CPUs ($PER per run), GPU ${CUDA_VISIBLE_DEVICES:-?}"
echo "  Deadline: $GLR_DEADLINE ($(date -d "@$GLR_DEADLINE" -Iseconds))"
for (( k = 0; k < N; k++ )); do
    echo "  slot$k: ${RUNS[$k]}  (${CONFIGS[$k]} seed ${SEEDS[$k]})"
done
echo "==================================================================="

for (( k = 0; k < N; k++ )); do
    chunk=$(echo "${CPUS[@]:$(( k * PER )):$PER}" | tr " " ",")
    slot_log="$GLR_TMP/glr-$JOB_ID-slot$k.log"
    GLR_PACKED_CHILD=1 \
    GLR_WORK="$WORK/slot$k" \
    GLR_CONFIG="${CONFIGS[$k]}" GLR_SEED="${SEEDS[$k]}" GLR_RUN_NAME="${RUNS[$k]}" \
    GLR_LOG_FILES="$slot_log" \
    OMP_NUM_THREADS="$PER" MKL_NUM_THREADS="$PER" \
        taskset -c "$chunk" bash "$TRAIN_SBATCH" > "$slot_log" 2>&1 &
    PIDS+=($!)
    glr_log "[pack] slot$k pid=$! cpus=$chunk log=$slot_log"
    sleep 20   # stagger start-up (model load / tokenisation peaks)
done

# ---- wait per PID, propagate failures --------------------------------------
final=0; any75=0
for (( k = 0; k < N; k++ )); do
    rc=0
    wait "${PIDS[$k]}" || rc=$?
    while (( rc > 128 )) && kill -0 "${PIDS[$k]}" 2>/dev/null; do
        rc=0; wait "${PIDS[$k]}" || rc=$?
    done
    glr_log "[pack] slot$k (${RUNS[$k]}) exited rc=$rc"
    if (( rc == 75 )); then
        any75=1
    elif (( rc != 0 && final == 0 )); then
        final=$rc
    fi
done
(( final == 0 && any75 == 1 )) && final=75
exit "$final"
