#!/bin/bash
# Portable (non-HPCC) BPB/PPL rerun for the four models shipped on Hugging Face:
#
#   LCLM (multithread_lm)  r20  d24_r20_climbmix_blockmt_v4   step 14800  15.5B tokens
#   Vanilla AR (simple_gpt) r20 d24_r20_climbmix_simple        step 14800  15.5B tokens
#   LCLM (multithread_lm)  r8   d24_r8_climbmix_blockmt_v4    step  5952   6.2B tokens
#   Vanilla AR (simple_gpt) r8  d24_r8_climbmix_simple         step  5952   6.2B tokens
#
# Reruns the exact eval used to produce paper/eval_bundle_d24/paper_tables/main_results.csv
# (val_bpb column). No SBATCH/cluster assumptions here. Works on any box with the
# repo's .venv, a CUDA GPU (or several, via NPROC), and NANOCHAT_BASE_DIR pointing
# at a base_checkpoints/ directory holding these four checkpoints.
#
# Usage:
#   NPROC=1 bash experiments/repro/run_bpb_eval.sh
#   NPROC=4 SELECT_TAGS=d24_r20_climbmix_blockmt_v4 bash experiments/repro/run_bpb_eval.sh
#   DRY_RUN=1 bash experiments/repro/run_bpb_eval.sh   # print commands, run nothing
#
# Env vars:
#   NPROC          torchrun workers / GPUs (default 1; original paper run used 4)
#   RUN_ID         output subfolder name (default d24_release_repro_bpb)
#   EVAL_TOKENS    eval budget in tokens (default 20971520 = 20M, matches paper)
#   SELECT_TAGS    space-separated subset of the four model tags (default: all four)
#   DRY_RUN        1 = print the commands instead of running them

set -uo pipefail

resolve_repo_root() {
    # Repo-root marker is nanochat/ itself, not .env -- .env is lab-only and
    # gitignored, so an external checkout of this branch will never have one.
    local candidate
    candidate="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)" || candidate=""
    if [[ -n "$candidate" && -d "${candidate}/nanochat" ]]; then
        echo "$candidate"; return 0
    fi
    echo "ERROR: cannot resolve repo root; run from a checkout containing nanochat/" >&2
    return 1
}

REPO_ROOT="$(resolve_repo_root)" || exit 1
cd "$REPO_ROOT"
[ -f .venv/bin/activate ] && source .venv/bin/activate
[ -f .env ] && { set -a; source .env; set +a; }

if [[ -z "${NANOCHAT_BASE_DIR:-}" ]]; then
    echo "ERROR: NANOCHAT_BASE_DIR is unset; run bootstrap.sh first (or check .env on the lab cluster)" >&2
    exit 1
fi

NPROC="${NPROC:-1}"
RUN_ID="${RUN_ID:-d24_release_repro_bpb}"
EVAL_RUN_DIR="${NANOCHAT_BASE_DIR}/eval_runs/${RUN_ID}"
RAW_DIR="${EVAL_RUN_DIR}/raw"
EVAL_TOKENS="${EVAL_TOKENS:-20971520}"
SELECT_TAGS="${SELECT_TAGS:-}"
DRY_RUN="${DRY_RUN:-0}"
TR=(torchrun --standalone --nproc_per_node="$NPROC" -m)

# Bug fixed 2026-09-02: this was missing, so every prior run of this script
# silently read the raw (unprocessed) climbmix val shard instead of the
# block-split climbmix_split val shard the paper's numbers were computed
# from -- val() picks the alphabetically-last shard_*.parquet in whichever
# dataset dir is active, and shard_06542.parquet exists (not bit-identical)
# in both, so it failed silently rather than erroring.
export NANOCHAT_DATASET="${DATASET:-climbmix_split}"

mkdir -p "$RAW_DIR"

want_tag() {
    local tag="$1"
    [[ -z "$SELECT_TAGS" || " $SELECT_TAGS " == *" $tag "* ]]
}

step_dir() {
    local tag="$1" step="$2"
    printf '%s/%s/step%06d' "$RAW_DIR" "$tag" "$step"
}

maybe() {
    local out="$1" desc="$2"; shift 2
    if [[ "${FORCE:-0}" != "1" && -s "$out" ]]; then
        echo "skip (exists): $desc -> $out"; return 0
    fi
    mkdir -p "$(dirname "$out")"
    echo; echo ">>> $desc"
    echo "out=$out"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY_RUN command:'; printf ' %q' "$@"; echo
        return 0
    fi
    "$@" || echo "!!! FAILED: $desc"
}

echo "=== d24 release-repro BPB eval (NPROC=$NPROC, RUN_ID=$RUN_ID, EVAL_TOKENS=$EVAL_TOKENS) ==="

# ---------- LCLM / multithread_lm (tokenizer nanochat_64k_mt_v4) ----------
export NANOCHAT_TOKENIZER="nanochat_64k_mt_v4"
for ts in d24_r20_climbmix_blockmt_v4:14800 d24_r8_climbmix_blockmt_v4:5952; do
    tag="${ts%%:*}"; st="${ts##*:}"
    want_tag "$tag" || continue
    out="$(step_dir "$tag" "$st")/val_loss_eval.json"
    maybe "$out" "val-bpb $tag" "${TR[@]}" scripts.eval.blockmt_val_loss \
        --model-tag "$tag" --step "$st" --eval-tokens "$EVAL_TOKENS" --out "$out"
done

# ---------- Vanilla AR / simple_gpt (tokenizer nanochat_64k_mt2) ----------
export NANOCHAT_TOKENIZER="nanochat_64k_mt2"
for ts in d24_r20_climbmix_simple:14800 d24_r8_climbmix_simple:5952; do
    tag="${ts%%:*}"; st="${ts##*:}"
    want_tag "$tag" || continue
    out="$(step_dir "$tag" "$st")/val_loss_eval.json"
    maybe "$out" "val-bpb $tag" "${TR[@]}" scripts.eval.simple_val_loss \
        --model-tag "$tag" --step "$st" --eval-tokens "$EVAL_TOKENS" --out "$out"
done

echo
echo "=== done. Raw outputs under: $RAW_DIR ==="
echo "Compare against the paper with:"
echo "  python experiments/repro/compare_to_paper.py --bpb-raw-dir $RAW_DIR"
