#!/bin/bash
# Get the climbmix_split validation shard used by run_bpb_eval.sh / run_core_eval.sh,
# starting from nothing but internet access.
#
# climbmix_split is NOT hosted anywhere (nanochat/data/dataset.py: base_url="").
# It's a local derivative of the public `karpathy/climbmix-400b-shuffle` HF
# dataset, produced by scripts/split_long_blocks.py. So this is two steps:
#   1. download the public climbmix val shard (HF dataset, no auth needed)
#   2. run split_long_blocks.py locally to turn it into climbmix_split's val shard
#
# Needs a tokenizer on disk first (run download_checkpoints.py, or bootstrap.sh,
# before this) -- split_long_blocks.py tokenizes candidate blocks to enforce the
# length cap. The split is computed once against nanochat_64k_mt2 and reused by
# every model regardless of that model's own tokenizer (it's a text-level split;
# each eval script retokenizes with its own tokenizer at read time).
#
# Usage:
#   NANOCHAT_BASE_DIR=/path/to/cache bash experiments/repro/download_dataset.sh

set -euo pipefail

resolve_repo_root() {
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

if [[ -z "${NANOCHAT_BASE_DIR:-}" ]]; then
    echo "ERROR: NANOCHAT_BASE_DIR is unset; run bootstrap.sh first (or export it yourself)" >&2
    exit 1
fi

SPLIT_TOKENIZER="${SPLIT_TOKENIZER:-nanochat_64k_mt2}"
TOK_DIR="${NANOCHAT_BASE_DIR}/tokenizers/${SPLIT_TOKENIZER}"
if [[ ! -f "${TOK_DIR}/tokenizer.pkl" ]]; then
    echo "ERROR: ${TOK_DIR}/tokenizer.pkl not found." >&2
    echo "       Run download_checkpoints.py (or bootstrap.sh) first -- it bundles this tokenizer." >&2
    exit 1
fi

echo "=== 1/2: downloading the climbmix val shard (public, no auth) ==="
NANOCHAT_DATASET=climbmix python -m nanochat.data.dataset --split val -n 1

echo
echo "=== 2/2: deriving climbmix_split's val shard (local preprocessing, tokenizer=$SPLIT_TOKENIZER) ==="
NANOCHAT_DATASET=climbmix python -m scripts.split_long_blocks \
    --tokenizer "$SPLIT_TOKENIZER" --split val

echo
echo "=== done. climbmix_split val data under: ${NANOCHAT_BASE_DIR}/base_data_climbmix_split ==="
