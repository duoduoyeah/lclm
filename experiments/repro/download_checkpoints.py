#!/usr/bin/env python3
"""Download the four released d24 checkpoints + their tokenizers from Hugging Face
into the nanochat on-disk layout (base_checkpoints/<tag>/, tokenizers/<name>/),
so run_bpb_eval.sh / run_core_eval.sh can load them with plain --model-tag.

This is the internet-download counterpart to the lab's pre-populated
base_checkpoints/ -- for anyone outside the lab, this script is what gets you
from "nothing" to "the four models are on disk."

Requires: huggingface_hub (already a nanochat dependency).

Auth: the four repos are private as of 2026-09-01. Set HF_TOKEN (same name
the rest of this repo uses, e.g. in .env) to an account with read access.
Once the repos are made public this script needs no token at all -- leave
HF_TOKEN unset and it still works.

Usage:
    NANOCHAT_BASE_DIR=/path/to/cache python experiments/repro/download_checkpoints.py
    NANOCHAT_BASE_DIR=/path/to/cache python experiments/repro/download_checkpoints.py --only d24_r20_climbmix_blockmt_v4
"""
import argparse
import os
import shutil
import sys

# (hf_repo, local_model_tag, step, tokenizer_name)
CHECKPOINTS = [
    ("duoduoyeah/nanochat-d24-blockmt-v4-r20", "d24_r20_climbmix_blockmt_v4", 14800, "nanochat_64k_mt_v4"),
    ("duoduoyeah/nanochat-d24-simple-r20", "d24_r20_climbmix_simple", 14800, "nanochat_64k_mt2"),
    ("duoduoyeah/nanochat-d24-blockmt-v4-r8", "d24_r8_climbmix_blockmt_v4", 5952, "nanochat_64k_mt_v4"),
    ("duoduoyeah/nanochat-d24-simple-r8", "d24_r8_climbmix_simple", 5952, "nanochat_64k_mt2"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=None,
                     help="local model tag to restrict to (repeatable); default: all four")
    ap.add_argument("--base-dir", default=None,
                     help="override NANOCHAT_BASE_DIR (default: read from env, error if unset)")
    args = ap.parse_args()

    base_dir = args.base_dir or os.environ.get("NANOCHAT_BASE_DIR")
    if not base_dir:
        raise SystemExit("ERROR: NANOCHAT_BASE_DIR is unset and --base-dir not given")

    # Keep HF's own blob cache inside NANOCHAT_BASE_DIR instead of the default
    # ~/.cache/huggingface. Hit this for real on this cluster: $HOME has a
    # small personal quota separate from /bigdata's much larger group quota,
    # and NANOCHAT_BASE_DIR normally lives under /bigdata -- a plain 3.3GB
    # checkpoint download failed with "Disk quota exceeded" against the
    # default ~/.cache/huggingface location. Must be set before importing
    # huggingface_hub (it reads this at import time). setdefault so a caller
    # who already set HF_HOME/HUGGINGFACE_HUB_CACHE keeps their own choice.
    os.environ.setdefault("HF_HOME", os.path.join(base_dir, "hf_home"))

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed in this environment")

    token = os.environ.get("HF_TOKEN")  # optional once repos are public
    ckpt_root = os.path.join(base_dir, "base_checkpoints")
    tok_root = os.path.join(base_dir, "tokenizers")
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(tok_root, exist_ok=True)

    wanted = set(args.only) if args.only else None
    done_tokenizers = set()

    for repo, tag, step, tok_name in CHECKPOINTS:
        if wanted is not None and tag not in wanted:
            continue

        tag_dir = os.path.join(ckpt_root, tag)
        os.makedirs(tag_dir, exist_ok=True)
        meta_name = f"meta_{step:06d}.json"
        model_name = f"model_{step:06d}.pt"

        for fname in (meta_name, model_name):
            dest = os.path.join(tag_dir, fname)
            if os.path.exists(dest):
                print(f"skip (exists): {dest}")
                continue
            print(f"downloading {repo}/{fname} -> {dest}")
            local = hf_hub_download(repo, fname, token=token)
            shutil.copy2(local, dest)

        if tok_name not in done_tokenizers:
            tok_dir = os.path.join(tok_root, tok_name)
            os.makedirs(tok_dir, exist_ok=True)
            for fname in ("tokenizer.pkl", "token_bytes.pt"):
                dest = os.path.join(tok_dir, fname)
                if os.path.exists(dest):
                    print(f"skip (exists): {dest}")
                    continue
                print(f"downloading {repo}/{fname} -> {dest}")
                local = hf_hub_download(repo, fname, token=token)
                shutil.copy2(local, dest)
            done_tokenizers.add(tok_name)

    print()
    print(f"Done. Checkpoints under: {ckpt_root}")
    print(f"Tokenizers under:        {tok_root}")


if __name__ == "__main__":
    main()
