#!/bin/bash
# One-shot setup for reproducing the d24 LCLM-vs-Vanilla-AR results from
# nothing but internet access + this checkout -- no lab cluster, no .env,
# no pre-populated NANOCHAT_BASE_DIR required.
#
# Everything downloaded/derived lands under ./.cache/nanochat inside this
# checkout (mirrors the normal nanochat default of ~/.cache/nanochat, just
# rooted in the repo instead of the home dir, so a clone is self-contained
# and easy to throw away). Override with NANOCHAT_BASE_DIR if you want it
# elsewhere.
#
# What this does:
#   1. downloads the 4 released checkpoints + their tokenizers from HF
#   2. downloads the public climbmix val shard and derives climbmix_split
#      from it locally (climbmix_split itself isn't hosted anywhere)
#
# Needs: this repo's Python env (torch, huggingface_hub, pyarrow, etc. --
# see the main README for environment setup) on PATH or in ./.venv.
#
# Auth note: the 4 checkpoints are private as of 2026-09-01. Set HF_TOKEN
# (an account with read access) until they're made public; unset works
# once they are.
#
# Usage:
#   bash experiments/repro/bootstrap.sh
#   HF_TOKEN=hf_... bash experiments/repro/bootstrap.sh
#   NANOCHAT_BASE_DIR=/somewhere/else bash experiments/repro/bootstrap.sh

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

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${REPO_ROOT}/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
echo "=== NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR ==="

echo
echo "=== step 1/2: checkpoints + tokenizers ==="
python experiments/repro/download_checkpoints.py

echo
echo "=== step 2/2: climbmix_split val data ==="
bash experiments/repro/download_dataset.sh

echo
echo "=== bootstrap done. Now run the evals, e.g.: ==="
echo "  NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR NPROC=1 bash experiments/repro/run_bpb_eval.sh"
echo "  NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR NPROC=1 bash experiments/repro/run_core_eval.sh"
