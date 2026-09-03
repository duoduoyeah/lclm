#!/bin/bash
# Portable (non-HPCC) CORE rerun for the four models shipped on Hugging Face.
# Reruns the eval behind paper/eval_bundle_d24/paper_tables/main_results.csv
# (core_two_line_10shot column) plus the 0-shot appendix variant.
#
# CAUTION (verified against scripts/base_eval.py): the CORE CSV writer names
# its output `base_model_<step>.csv` -- it does NOT include the model tag.
# d24_r20_climbmix_blockmt_v4 and d24_r20_climbmix_simple share step 14800,
# so back-to-back runs at the same step silently clobber each other's CSV
# unless moved out of the way first. (The existing paper bundle hit exactly
# this: see the "stale/wrong source" note in
# paper/eval_bundle_d24/phase2_core/provenance/source_paths.csv.) This script
# renames each output to `<model_tag>_<step>_<shots>shot.csv` immediately
# after it's written, so run each model one at a time -- don't parallelize
# across tags into the same NANOCHAT_BASE_DIR.
#
# Usage:
#   NPROC=1 bash experiments/repro/run_core_eval.sh
#   SHOTS=0 NPROC=1 bash experiments/repro/run_core_eval.sh   # 0-shot appendix row
#   DRY_RUN=1 bash experiments/repro/run_core_eval.sh
#
# Env vars:
#   NPROC          torchrun workers / GPUs (default 1; original paper run used 4)
#   RUN_ID         output subfolder name (default d24_release_repro_core)
#   SHOTS          "" (task default, i.e. the 10-shot main-table row) or "0"
#                  (forces every task to 0-shot, the appendix row)
#   CORE_SHOT_LAYOUT  two_line (default, matches main table) or one_line
#                  (same-line ablation row). These are our/the paper's labels;
#                  mapped internally to base_eval.py's real --core-shot-layout
#                  values (normal, one_line) -- base_eval.py has no "two_line"
#                  choice, so passing the label straight through fails argparse.
#   SELECT_TAGS    space-separated subset of the four model tags (default: all four)
#   MAX_PER_TASK   debugging subsample; unset = full 22-task CORE
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
RUN_ID="${RUN_ID:-d24_release_repro_core}"
OUT_DIR="${NANOCHAT_BASE_DIR}/eval_runs/${RUN_ID}/core"
SHOTS="${SHOTS:-}"
CORE_SHOT_LAYOUT="${CORE_SHOT_LAYOUT:-two_line}"
MAX_PER_TASK="${MAX_PER_TASK:-}"
SELECT_TAGS="${SELECT_TAGS:-}"
DRY_RUN="${DRY_RUN:-0}"
SHOT_LABEL="${SHOTS:-default}"

mkdir -p "$OUT_DIR"

want_tag() {
    local tag="$1"
    [[ -z "$SELECT_TAGS" || " $SELECT_TAGS " == *" $tag "* ]]
}


# base_eval.py's actual --core-shot-layout choices are {normal, one_line} --
# it has no "two_line" value. "two_line" is our/the paper's label for what
# base_eval.py calls its "normal" (default) layout, kept here because it's
# what paper_tables/main_results.csv and the eval bundle call it. Map the
# public label to the real CLI value; never pass CORE_SHOT_LAYOUT straight
# through to base_eval.py.
case "$CORE_SHOT_LAYOUT" in
    two_line) CLI_SHOT_LAYOUT=normal ;;
    one_line) CLI_SHOT_LAYOUT=one_line ;;
    *) echo "ERROR: CORE_SHOT_LAYOUT must be two_line or one_line, got '$CORE_SHOT_LAYOUT'" >&2; exit 1 ;;
esac

FAILED_TAGS=()

run_one() {
    local tag="$1" step="$2" kind="$3"  # kind = mt | simple
    want_tag "$tag" || return 0

    local dest="${OUT_DIR}/${tag}_${step}_${SHOT_LABEL}shot_${CORE_SHOT_LAYOUT}.csv"
    if [[ "${FORCE:-0}" != "1" && -s "$dest" ]]; then
        echo "skip (exists): $tag -> $dest"; return 0
    fi

    local extra=(--model-tag "$tag" --step "$step" --eval core --core-shot-layout "$CLI_SHOT_LAYOUT")
    [[ -n "$SHOTS" ]] && extra+=(--core-num-fewshot "$SHOTS")
    [[ -n "$MAX_PER_TASK" ]] && extra+=(--max-per-task "$MAX_PER_TASK")
    [[ "$kind" == "mt" ]] && extra+=(--mt-encoding v4_tail)

    echo; echo ">>> CORE $tag (step $step, shots=${SHOT_LABEL}, layout=$CORE_SHOT_LAYOUT -> --core-shot-layout $CLI_SHOT_LAYOUT)"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY_RUN command: torchrun --standalone --nproc_per_node=%s -m scripts.base_eval' "$NPROC"
        printf ' %q' "${extra[@]}"; echo
        echo "DRY_RUN would then move base_eval/base_model_$(printf '%06d' "$step")*.csv -> $dest"
        return 0
    fi

    torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_eval "${extra[@]}" \
        || { echo "!!! FAILED: $tag"; return 1; }

    # base_eval.py names its CSV base_model_<step>[_suffix].csv with no model-tag
    # in the name -- find the newest matching file and claim it before the next
    # model (same step) can overwrite it.
    local produced
    produced="$(ls -t "${NANOCHAT_BASE_DIR}/base_eval/base_model_$(printf '%06d' "$step")"*.csv 2>/dev/null | head -1)"
    if [[ -z "$produced" ]]; then
        echo "!!! FAILED: no base_eval/base_model_$(printf '%06d' "$step")*.csv produced for $tag"
        return 1
    fi
    mv "$produced" "$dest"
    echo "-> $dest"
}

echo "=== d24 release-repro CORE eval (NPROC=$NPROC, RUN_ID=$RUN_ID, shots=$SHOT_LABEL, layout=$CORE_SHOT_LAYOUT) ==="

export NANOCHAT_TOKENIZER="nanochat_64k_mt_v4"
run_one d24_r20_climbmix_blockmt_v4 14800 mt || FAILED_TAGS+=(d24_r20_climbmix_blockmt_v4)
run_one d24_r8_climbmix_blockmt_v4  5952  mt || FAILED_TAGS+=(d24_r8_climbmix_blockmt_v4)

export NANOCHAT_TOKENIZER="nanochat_64k_mt2"
run_one d24_r20_climbmix_simple 14800 simple || FAILED_TAGS+=(d24_r20_climbmix_simple)
run_one d24_r8_climbmix_simple  5952  simple || FAILED_TAGS+=(d24_r8_climbmix_simple)

echo
if [[ ${#FAILED_TAGS[@]} -gt 0 ]]; then
    echo "=== FAILED (${#FAILED_TAGS[@]}/4): ${FAILED_TAGS[*]} ==="
    echo "    (nonzero exit -- don't read this as a clean run; see the FAILED lines above)"
    exit 1
fi
echo "=== done. CSVs under: $OUT_DIR ==="
echo "Compare against the paper with:"
echo "  python experiments/repro/compare_to_paper.py --core-dir $OUT_DIR"
