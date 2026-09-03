"""
Extend a base BPE tokenizer with multithread (block-MT) special tokens.

Loads a base tokenizer pickle (a `tiktoken.Encoding`), reconstructs a new
Encoding with the same `pat_str` and `mergeable_ranks` but with the
specials dict extended to include the MT specials, then saves the new
encoding to a new directory and recomputes `token_bytes.pt`.

MT specials added (per design/blockmt/multithread-block-design.md §D4 + §C1
and design/blockmt/blockmt-newline-preserve-runs.md):
  - <|sot|>: start of thread.
  - <|bot_K|> for K = 1..K_max: wave-size markers (default K_max=16).
  - <|eot|>: end of thread. Target-only override for non-decision
    blocks' c_L logit (predicted, never fed back as input). Required
    by the run-preservation design so non-decision threads can retire
    explicitly after a newline run, without speculative discard.
  - <|sot_tail|>: v4 tail-line anchor. Included only with
    --layout-version=v4.

Example:
    python -m scripts.tok_extend_mt \
        --base-name=nanochat_64k \
        --out-name=nanochat_64k_mt \
        --k-max=16

    python -m scripts.tok_extend_mt \
        --base-name=nanochat_64k_mt2 \
        --out-name=nanochat_64k_mt_v4 \
        --layout-version=v4
"""

import argparse
import os
import pickle
import tiktoken
import torch

from nanochat.common import get_base_dir


def mt_special_names(k_max, layout_version="v2"):
    """Names of MT specials to append.

    Order: <|sot|>, <|bot_1|>..<|bot_K_max|>, <|eot|>. <|eot|> is at the
    end so existing <|bot_K|> ids (65537..65552 for K_max=16) stay stable
    across the v1->v2 mt-tokenizer transition; <|eot|> takes the next
    free id (65553). v4 appends <|sot_tail|> after <|eot|>.
    """
    names = (
        ["<|sot|>"]
        + [f"<|bot_{k}|>" for k in range(1, k_max + 1)]
        + ["<|eot|>"]
    )
    if layout_version == "v4":
        names.append("<|sot_tail|>")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-name", required=True,
                        help="Base tokenizer dir name under $NANOCHAT_BASE_DIR/tokenizers/")
    parser.add_argument("--out-name", required=True,
                        help="Output tokenizer dir name under $NANOCHAT_BASE_DIR/tokenizers/")
    parser.add_argument("--k-max", type=int, default=16,
                        help="Maximum K for <|bot_K|> specials (default 16; parametric per C1)")
    parser.add_argument("--layout-version", choices=("v2", "v3", "v4"), default="v2",
                        help="MT tokenizer layout version. v2/v3 add SOT/BOT/EOT; v4 also adds <|sot_tail|>.")
    args = parser.parse_args()

    base_dir = get_base_dir()
    base_path = os.path.join(base_dir, "tokenizers", args.base_name)
    out_path = os.path.join(base_dir, "tokenizers", args.out_name)

    base_pkl = os.path.join(base_path, "tokenizer.pkl")
    assert os.path.exists(base_pkl), f"Base tokenizer pickle not found: {base_pkl}"
    with open(base_pkl, "rb") as f:
        base_enc = pickle.load(f)

    print(f"Base : {base_path}")
    print(f"  vocab_size = {base_enc.n_vocab}")
    print(f"  specials = {sorted(base_enc.special_tokens_set)}")

    existing_specials = dict(base_enc._special_tokens)
    new_specials = mt_special_names(
        args.k_max,
        layout_version="v4" if args.layout_version == "v4" else "v2",
    )
    next_id = base_enc.n_vocab
    extended_specials = dict(existing_specials)
    added_specials = []
    for name in new_specials:
        if name in extended_specials:
            continue
        extended_specials[name] = next_id
        next_id += 1
        added_specials.append(name)

    new_enc = tiktoken.Encoding(
        name=base_enc.name + "_mt_" + args.layout_version,
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens=extended_specials,
    )

    print(f"Out  : {out_path}")
    print(f"  vocab_size = {new_enc.n_vocab}")
    print(f"  requested {len(new_specials)} MT specials: {new_specials}")
    print(f"  added {len(added_specials)} new specials: {added_specials}")

    os.makedirs(out_path, exist_ok=True)
    out_pkl = os.path.join(out_path, "tokenizer.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(new_enc, f)
    print(f"Saved {out_pkl}")

    vocab_size = new_enc.n_vocab
    special_set = set(new_enc.special_tokens_set)
    token_bytes = []
    for tid in range(vocab_size):
        try:
            token_str = new_enc.decode([tid])
        except Exception:
            token_str = ""
        if token_str in special_set:
            token_bytes.append(0)
        else:
            token_bytes.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    out_pt = os.path.join(out_path, "token_bytes.pt")
    torch.save(token_bytes_tensor, out_pt)
    print(f"Saved {out_pt}")


if __name__ == "__main__":
    main()
