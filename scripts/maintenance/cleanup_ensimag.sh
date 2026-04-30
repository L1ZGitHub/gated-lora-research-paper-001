#!/bin/bash
# One-time Ensimag disk cleanup script.
#
# To be run ON ENSIMAG. Default mode = dry-run (lists what *would* be deleted).
# Pass explicit --confirm-* flags to actually delete.
#
# What's safe to delete (already backed up to D:\ensimag_backup\):
#   - ensicompute_*/outputs/                    (LoRA weights live on D:)
#
# What's NOT backed up anywhere — DELETING THESE LOSES THEM:
#   - old/                  (51 GB, contents unknown)
#   - experiment2_*/        (LDA-vs-embeddings project, ~2.4 GB)
#   - capacity_collapse*/   (~200 MB)
#
# Run pattern (recommended, in order):
#   1. ./cleanup_ensimag.sh                          # dry-run (default)
#   2. ./cleanup_ensimag.sh --confirm-legacy-outputs # frees ~290 GB safely
#   3. ./cleanup_ensimag.sh --inspect old            # decide what's in old/
#   4. (case by case for the rest)
#
# After running, verify:
#   df -h ~

set -euo pipefail

ROOT="${GLR_ENSICOMPUTE_ROOT:-$HOME/GatedLoraProject/ensicompute}"

CONFIRM_LEGACY_OUTPUTS=0
CONFIRM_OLD=0
CONFIRM_EXPERIMENT2=0
CONFIRM_CAPACITY=0
INSPECT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --confirm-legacy-outputs)  CONFIRM_LEGACY_OUTPUTS=1; shift ;;
        --confirm-old)             CONFIRM_OLD=1; shift ;;
        --confirm-experiment2)     CONFIRM_EXPERIMENT2=1; shift ;;
        --confirm-capacity)        CONFIRM_CAPACITY=1; shift ;;
        --inspect)                 INSPECT="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d "$ROOT" ]]; then
    echo "ERROR: ensicompute root not found at $ROOT"
    echo "Set GLR_ENSICOMPUTE_ROOT or fix the default in this script."
    exit 1
fi

echo "============================================================"
echo "Ensimag cleanup — root=$ROOT"
date -Iseconds
echo "============================================================"
df -h "$HOME" | head -2
echo ""

if [[ -n "$INSPECT" ]]; then
    target="$ROOT/$INSPECT"
    [[ -d "$target" ]] || { echo "Not found: $target" >&2; exit 1; }
    echo "=== Inspecting $target ==="
    ls -la "$target" | head -30
    echo "---"
    if [[ -f "$target/README.md" ]]; then
        echo "README.md:"
        head -30 "$target/README.md"
    fi
    echo "---"
    echo "Total size:"
    du -sh "$target"
    echo "---"
    echo "Subdirectory sizes (top 20):"
    du -sh "$target"/*/ 2>/dev/null | sort -h | tail -20
    exit 0
fi

# --- Section 1: legacy ensicompute_*/outputs/ (safe — already on D:) ---
echo "=== Section 1: ensicompute_*/outputs/ (BACKED UP on external D:) ==="
total_safe=0
for d in "$ROOT"/ensicompute_*/; do
    [[ -d "$d/outputs" ]] || continue
    size_kb=$(du -sk "$d/outputs" 2>/dev/null | cut -f1)
    size_h=$(du -sh "$d/outputs" 2>/dev/null | cut -f1)
    echo "  $size_h  $d/outputs"
    total_safe=$((total_safe + size_kb))
done
total_safe_h=$(echo "$total_safe / 1024 / 1024" | bc -l 2>/dev/null | head -c 6 || echo "?")
echo "  --- total: ~${total_safe_h} GB ---"
if [[ $CONFIRM_LEGACY_OUTPUTS -eq 1 ]]; then
    echo "  >> DELETING (--confirm-legacy-outputs)"
    for d in "$ROOT"/ensicompute_*/; do
        [[ -d "$d/outputs" ]] || continue
        rm -rf "$d/outputs"
        echo "    removed: $d/outputs"
    done
else
    echo "  (dry-run; pass --confirm-legacy-outputs to delete)"
fi
echo ""

# --- Section 2: old/ (NOT backed up, contents unknown) ---
echo "=== Section 2: old/ (NOT backed up — investigate first) ==="
if [[ -d "$ROOT/old" ]]; then
    du -sh "$ROOT/old" 2>/dev/null
    echo "  Run with: --inspect old   to see contents before deciding"
    if [[ $CONFIRM_OLD -eq 1 ]]; then
        echo "  >> DELETING $ROOT/old (--confirm-old)"
        rm -rf "$ROOT/old"
    else
        echo "  (dry-run; --inspect old, then --confirm-old if OK to lose it)"
    fi
else
    echo "  (already absent)"
fi
echo ""

# --- Section 3: experiment2*/ (LDA project) ---
echo "=== Section 3: experiment2*/ (LDA-vs-embeddings, NOT backed up) ==="
exp2_count=0
for d in "$ROOT"/experiment2*/; do
    [[ -d "$d" ]] || continue
    exp2_count=$((exp2_count+1))
    echo "  $(du -sh "$d" 2>/dev/null | cut -f1)  $d"
done
if [[ $exp2_count -eq 0 ]]; then
    echo "  (already absent)"
elif [[ $CONFIRM_EXPERIMENT2 -eq 1 ]]; then
    echo "  >> DELETING (--confirm-experiment2)"
    for d in "$ROOT"/experiment2*/; do
        [[ -d "$d" ]] || continue
        rm -rf "$d"
        echo "    removed: $d"
    done
else
    echo "  (dry-run; pass --confirm-experiment2 to delete)"
fi
echo ""

# --- Section 4: capacity_collapse*/ ---
echo "=== Section 4: capacity_collapse*/ (NOT backed up, ~200 MB) ==="
cap_count=0
for d in "$ROOT"/capacity_collapse*/; do
    [[ -d "$d" ]] || continue
    cap_count=$((cap_count+1))
    echo "  $(du -sh "$d" 2>/dev/null | cut -f1)  $d"
done
if [[ $cap_count -eq 0 ]]; then
    echo "  (already absent)"
elif [[ $CONFIRM_CAPACITY -eq 1 ]]; then
    echo "  >> DELETING (--confirm-capacity)"
    for d in "$ROOT"/capacity_collapse*/; do
        [[ -d "$d" ]] || continue
        rm -rf "$d"
        echo "    removed: $d"
    done
else
    echo "  (dry-run; pass --confirm-capacity to delete)"
fi
echo ""

echo "============================================================"
echo "Done. Disk after cleanup:"
df -h "$HOME" | head -2
echo "============================================================"
