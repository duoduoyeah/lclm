"""
Training-free parameter-count tool.

Builds the model on the meta device (no GPU allocation, no data), reads
`num_scaling_params()`, prints the breakdown. Use to size pilot runs
without paying for any actual compute.

Example:
    python -m scripts.param_count --model multithread_lm --depth 20
    python -m scripts.param_count --model gpt --depth 24 --aspect-ratio 64
    python -m scripts.param_count --model multithread_lm --depth 16 --json

Block-MT design (per design/blockmt/block-mt-open-questions.md E2) wants the
first pilot at >=200M params, so common depth probes are d=16 (~270M),
d=20 (~520M), d=24 (~850M) with the default aspect=64 head_dim=128.
"""

import argparse
import json

import torch

from nanochat.models.gpt import GPT, GPTConfig
from nanochat.models.simple_gpt import SimpleGPT, SimpleGPTConfig
from nanochat.models.multithread_lm import MultithreadLM, MultithreadLMConfig


MODEL_CLASSES = {
    "gpt": (GPT, GPTConfig),
    "simple_gpt": (SimpleGPT, SimpleGPTConfig),
    "multithread_lm": (MultithreadLM, MultithreadLMConfig),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="multithread_lm",
                        choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--aspect-ratio", type=int, default=64,
                        help="model_dim = depth * aspect_ratio, rounded up to head_dim")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=65553,
                        help="Vocab size (default: nanochat_64k_mt = 65553)")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--json", action="store_true", dest="json_out",
                        help="JSON output (machine-readable)")
    args = parser.parse_args()

    ModelCls, ConfigCls = MODEL_CLASSES[args.model]
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    n_head = model_dim // args.head_dim

    # All configs share these field names; per-model extras
    # (window_pattern, rope_*) take their defaults.
    config = ConfigCls(
        sequence_len=args.seq_len, vocab_size=args.vocab_size,
        n_layer=args.depth, n_head=n_head, n_kv_head=n_head,
        n_embd=model_dim,
    )

    with torch.device("meta"):
        model = ModelCls(config)
    counts = model.num_scaling_params()
    # Augment with shape info for the report
    counts["depth"] = args.depth
    counts["n_embd"] = model_dim
    counts["n_head"] = n_head
    counts["head_dim"] = args.head_dim
    counts["seq_len"] = args.seq_len
    counts["vocab_size"] = args.vocab_size

    if args.json_out:
        print(json.dumps(counts, indent=2))
        return

    print(f"Model: {args.model}")
    print(f"  depth        = {args.depth}")
    print(f"  aspect_ratio = {args.aspect_ratio}")
    print(f"  head_dim     = {args.head_dim}")
    print(f"  n_head       = {n_head}")
    print(f"  n_embd       = {model_dim}")
    print(f"  seq_len      = {args.seq_len}")
    print(f"  vocab_size   = {args.vocab_size}")
    print()
    print("Param breakdown:")
    shape_keys = {"depth", "n_embd", "n_head", "head_dim", "seq_len", "vocab_size"}
    for k, v in counts.items():
        if k in shape_keys:
            continue
        if k == "total":
            print(f"  {k:>26}: {v:>14,}  ({v/1e6:.1f}M)")
        else:
            print(f"  {k:>26}: {v:>14,}")


if __name__ == "__main__":
    main()
