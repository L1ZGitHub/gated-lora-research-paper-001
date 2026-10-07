#!/bin/bash
#SBATCH --job-name=glr-validate-v2
#SBATCH --output=/user/2/zimmermh/GatedLoraProject/v2_validate/logs/%x_%j.out
#SBATCH --error=/user/2/zimmermh/GatedLoraProject/v2_validate/logs/%x_%j.out
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
# One-shot validation of the v2 code before any real run. Steps (each logged, none fatal to the
# others; results in $V/results/<jobid>/):
#   1. pytest tests/ (tiny CPU models, configs, sampler, collator, scoring) -> pytest.txt
#   2. bench (bench_step.py): old code (installed repo) vs v2 equivalence in fp32 + mask check;
#      v2 step time / VRAM / tokens/s / padding on real v2 batches: bs 4, 8, 16; bs 8 with
#      dropout 0; bs 8 with the base attention mask kept
#   3. smoke gated: stop at step 150 (GLR_STOP_AT_STEP) + --resume auto to 400, and the same
#      run uninterrupted (resume exactness: compare final val loss)
#   4. smoke baseline r66 (PEFT path, generation), 5. smoke frozen gate (calibration)
# Code: $V/code (rsynced v2 tree, first on PYTHONPATH). Never pushes to HF (no HF_TOKEN).
# No apostrophes in any VAR:? message (node bash bug).
set -u
echo "=== NODE $(hostname) $(date -Iseconds) ==="
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 500 {print $1; exit}')
if [ -z "$FREE_GPU" ]; then echo "!!! no idle GPU on $(hostname), giving up"; exit 75; fi
export CUDA_VISIBLE_DEVICES="$FREE_GPU"
nvidia-smi -i "$FREE_GPU" --query-gpu=name,memory.used --format=csv,noheader
FREE_TMP=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
if (( FREE_TMP < 15 )); then echo "!!! only ${FREE_TMP}G free on /tmp"; exit 75; fi

V=/user/2/zimmermh/GatedLoraProject/v2_validate
OLD=/user/2/zimmermh/GatedLoraProject/gated-lora-research-paper-001
PY=$OLD/.venv/bin/python
W=/tmp/glr_validate_$SLURM_JOB_ID
OUT=$V/results/$SLURM_JOB_ID
mkdir -p "$W" "$OUT"
trap 'rm -rf "$W"' EXIT
export HF_HOME="$W/hf" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
unset HF_TOKEN
NEW_PP="$V/code/src"
step() { echo; echo "=== [$1] $(date -Iseconds)"; }
# GLR_VALIDATE_ONLY="1 2 3 4 5" (default all): run only these steps
ONLY=" ${GLR_VALIDATE_ONLY:-1 2 3 4 5} "
want() { [[ "$ONLY" == *" $1 "* ]]; }
cd "$W"  # the CLI creates ./outputs/<name> before --output-dir applies

want 1 && step "1 pytest" && (cd "$V/code" && PYTHONPATH="$NEW_PP" $PY -m pytest -p no:cacheprovider -rfE tests \
    > "$OUT/pytest.txt" 2>&1; echo "pytest rc=$?")
want 1 && tail -n 60 "$OUT/pytest.txt"

B="$V/code/scripts/bench_step.py"
J='^\{|Error|Traceback|error:'
if want 2; then
step "2 bench"
$PY "$B" --data-only --out "$OUT" 2>&1 | grep -E "$J" || true
$PY "$B" --tag old --out "$OUT" 2>&1 | grep -E "$J" || true
v2() { PYTHONPATH="$NEW_PP" $PY "$B" --out "$OUT" --v2-path "$@" 2>&1 | grep -E "$J" || true; }
v2 --tag v2_bs4 --batch-size 4 --check-mask
$PY "$B" --compare "$OUT/old.npz" "$OUT/v2_bs4.npz" || true
v2 --tag v2_bs8 --batch-size 8 --skip-equivalence
v2 --tag v2_bs16 --batch-size 16 --skip-equivalence
v2 --tag v2_bs8_dropout0 --batch-size 8 --lora-dropout 0.0 --skip-equivalence
v2 --tag v2_bs8_basemask --batch-size 8 --keep-base-mask --skip-equivalence
fi

smoke() {  # $1 kind, $2 output dir name, rest: extra CLI args
    local kind="$1" name="$2"; shift 2
    PYTHONPATH="$NEW_PP" $PY -m gated_lora.cli \
        --config "$V/code/configs/experiments/smoke_v2_qwen25_05b_$kind.yaml" --seed 0 \
        --output-dir "$W/$name" "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    grep -E "Peak VRAM|Step [0-9]+/|DEADLINE|STOP_AT_STEP|Resuming|Generation eval|Frozen gates|final_test|ERROR|Error|Traceback" \
        "$OUT/$name.log" | grep -v "HTTP Request" | tail -n 40
    echo "rc=$rc"
    mkdir -p "$OUT/$name"
    cp "$W/$name"/*.json "$W/$name"/*.npz "$OUT/$name/" 2>/dev/null
    ls -R "$W/$name" | head -40 > "$OUT/$name/tree.txt"
}

if want 3; then
step "3a smoke gated: stop at step 150"
GLR_STOP_AT_STEP=150 smoke gated gated_resume --analyze-routing
step "3b smoke gated: resume to 400"
smoke gated gated_resume --analyze-routing --resume auto
mv "$OUT/gated_resume.log" "$OUT/gated_resume_part2.log"
step "3c smoke gated: uninterrupted"
smoke gated gated_straight --analyze-routing
step "3d resume exactness"
$PY - "$W" <<'PYEOF'
import json, sys
w = sys.argv[1]
r = {k: json.load(open(f"{w}/{k}/eval_results.json")) for k in ("gated_resume", "gated_straight")}
for k, v in r.items():
    print(k, "final val loss", v["val_full"]["eval_loss"], "final loss", v["final"]["eval_loss"])
d = abs(r["gated_resume"]["val_full"]["eval_loss"] - r["gated_straight"]["val_full"]["eval_loss"])
print(f"|delta val loss| = {d:.2e} ({'OK' if d < 1e-2 else 'MISMATCH'}; nondeterministic kernels allow ~1e-3)")
PYEOF
fi

if want 4; then
step "4 smoke baseline r66"
smoke baseline_r66 baseline
fi

if want 5; then
step "5 smoke frozen gate"
smoke frozen_gate frozen
cat "$W/frozen/frozen_gate_calibration.json" 2>/dev/null | head -c 600; echo
fi

echo "=== EXIT=0 $(date -Iseconds) ==="
