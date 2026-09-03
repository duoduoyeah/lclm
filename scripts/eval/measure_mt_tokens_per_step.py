"""Measure per-step token yield for MT row layouts.

The metric is structural: generated content tokens per logical decoding step
for one sequence under the row-layout attention relation. The script is
CPU-only and never loads model weights. It mirrors the tokenizer/block splitter
and document topology used by the training/eval dataloader, then counts row
emission groups.

Supported layout: v4 only — ``build_doc_v4`` plus the ``compile_doc_v4``
SOT-ramp schedule, including the checkpoint's saved ``mt_v4_parallel_cap``
when present. (The legacy v2/v3 ``build_doc`` accounting was removed in the
v4-only cleanup.)

Outputs keep the historical ``waterfall`` field names for compatibility. For v4,
``waterfall`` means the real one-step-offset v4 row schedule; ``flat`` is an
appendix/audit accounting that removes the inter-line one-step offset while
respecting the same cap in batches.

Usage:
    NANOCHAT_DATASET=climbmix_split \
    NANOCHAT_TOKENIZER=nanochat_64k_mt_v4 \
        python -m scripts.eval.measure_mt_tokens_per_step \
            --model-tag d24_r8_climbmix_blockmt_v4_cap4 \
            --step 5952 \
            --target-tokens 5242880 \
            --out /tmp/tokens_per_step.json
"""
import argparse
import json
import os
from collections import Counter

import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.data.dataset import list_parquet_files
from nanochat.data.multithread_dataloader import (
    NL_CLASS_PURE, NL_CLASS_TRAIL, assert_walker_invariants,
    build_doc_v4, split_tokens_into_blocks,
)
from nanochat.tokenizer import RustBPETokenizer


def _group(groups, size=1):
    if size > 0:
        groups.append(size)


def decode_step_groups_v4(doc, v4_parallel_cap=None, include_bos=True):
    """Mirror ``compile_doc_v4`` emission groups at decode-step granularity."""
    groups = []

    if include_bos:
        _group(groups, 1)
    for _tok in doc.warmup[0].content:
        _group(groups, 1)

    for para in doc.warmup[1:]:
        _group(groups, 1)  # forced/input SOT
        for _tok in para.content:
            _group(groups, 1)

    K = len(doc.middle)
    if K:
        lengths = [p.length for p in doc.middle]
        if v4_parallel_cap is None:
            max_w = max(t + lengths[t] for t in range(K))
            for w in range(max_w + 1):
                n = 0
                for t in range(K):
                    if w < t:
                        continue
                    ls = w - t
                    if ls <= lengths[t]:
                        n += 1
                if doc.tail is not None and w == K:
                    n += 1
                _group(groups, n)
        else:
            next_middle = 0
            active = []
            next_ls = {}
            tail_candidate_emitted = False
            while (next_middle < K or active
                   or (doc.tail is not None and not tail_candidate_emitted)):
                active_at_step_start = list(active)
                finished = set()
                n = 0
                for t in active_at_step_start:
                    n += 1
                    ls = next_ls[t]
                    if ls == lengths[t]:
                        finished.add(t)
                    else:
                        next_ls[t] = ls + 1

                if len(active_at_step_start) < v4_parallel_cap:
                    if next_middle < K:
                        t = next_middle
                        n += 1
                        active.append(t)
                        next_ls[t] = 1
                        next_middle += 1
                    elif doc.tail is not None and not tail_candidate_emitted:
                        n += 1
                        tail_candidate_emitted = True

                _group(groups, n)
                if finished:
                    active = [t for t in active if t not in finished]
    elif doc.tail is not None:
        _group(groups, 1)  # tail-candidate SOT

    if doc.tail is not None:
        _group(groups, 1)  # <|sot_tail|>
        for _tok in doc.tail.content:
            _group(groups, 1)

    return groups


def flat_step_groups_v4(doc, v4_parallel_cap=None, include_bos=True):
    """Flat active-line audit accounting for v4.

    Warmup and tail remain sequential. Ordinary middle lines are grouped into
    batches of size ``cap`` (or all lines when uncapped); within each batch all
    SOT anchors are emitted together, then all active content depths advance in
    lockstep. This is not the actual causal row order; it is an audit value that
    removes the one-step line offset while respecting the cap.
    """
    groups = []

    if include_bos:
        _group(groups, 1)
    for _tok in doc.warmup[0].content:
        _group(groups, 1)

    for para in doc.warmup[1:]:
        _group(groups, 1)
        for _tok in para.content:
            _group(groups, 1)

    K = len(doc.middle)
    if K:
        lengths = [p.length for p in doc.middle]
        batch_size = K if v4_parallel_cap is None else v4_parallel_cap
        for start in range(0, K, batch_size):
            batch = lengths[start:start + batch_size]
            _group(groups, len(batch))  # SOT anchors
            max_len = max(batch)
            for depth in range(1, max_len + 1):
                _group(groups, sum(1 for L in batch if depth <= L))
        if doc.tail is not None:
            _group(groups, 1)  # tail-candidate SOT
    elif doc.tail is not None:
        _group(groups, 1)

    if doc.tail is not None:
        _group(groups, 1)  # <|sot_tail|>
        for _tok in doc.tail.content:
            _group(groups, 1)

    return groups


def v4_content_tokens(doc):
    total = sum(p.length for p in doc.warmup)
    total += sum(p.length for p in doc.middle)
    if doc.tail is not None:
        total += doc.tail.length
    return total


def v4_doc_summary(doc, v4_parallel_cap=None):
    one_step = decode_step_groups_v4(doc, v4_parallel_cap=v4_parallel_cap)
    flat = flat_step_groups_v4(doc, v4_parallel_cap=v4_parallel_cap)
    tokens_incl = sum(one_step)
    assert tokens_incl == sum(flat), (tokens_incl, sum(flat))
    tokens_excl = v4_content_tokens(doc)
    return {
        "tokens_incl": tokens_incl,
        "tokens_excl": tokens_excl,
        "steps_waterfall": len(one_step),
        "steps_flat": len(flat),
        "warmup_lines": len(doc.warmup),
        "middle_lines": len(doc.middle),
        "tail_lines": 1 if doc.tail is not None else 0,
        "warmup_tokens": sum(p.length for p in doc.warmup),
        "middle_tokens": sum(p.length for p in doc.middle),
        "tail_tokens": doc.tail.length if doc.tail is not None else 0,
    }


def latest_meta_path(ckpt_dir, step=None):
    if step is None:
        metas = sorted(
            f for f in os.listdir(ckpt_dir)
            if f.startswith("meta_") and f.endswith(".json")
        )
        assert metas, f"no meta_*.json in {ckpt_dir}"
        return os.path.join(ckpt_dir, metas[-1])
    return os.path.join(ckpt_dir, f"meta_{step:06d}.json")


def resolve_cap(value):
    if value is None or int(value) == 0:
        return None
    value = int(value)
    if value < 0:
        raise ValueError("parallel cap must be >= 0")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-tag", default=None,
                   help="optional: read layout knobs from checkpoint meta")
    p.add_argument("--step", type=int, default=None,
                   help="optional: meta step (default: latest)")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--target-tokens", type=int, default=5 * 1024 * 1024,
                   help="stop once incl-special row tokens reaches this")
    p.add_argument("--kmax", type=int, default=None,
                   help="v2/v3 override for meta's mt_kmax")
    p.add_argument("--warmup-threshold", type=int, default=None,
                   help="override meta's mt_warmup_threshold")
    p.add_argument("--layout-version", default=None,
                   choices=["v4"],
                   help="override meta's mt_layout_version (v4 only)")
    p.add_argument("--mt-v4-parallel-cap", type=int, default=None,
                   help="v4 cap override. Default uses checkpoint meta; 0 forces uncapped.")
    p.add_argument("--tokenizer", default=None,
                   help="override tokenizer. Default: checkpoint meta, then env.")
    p.add_argument("--out", default=None,
                   help="JSON output path (default: stdout only)")
    args = p.parse_args()

    base = get_base_dir()
    meta = None
    meta_path = None
    uc = {}
    if args.model_tag is not None:
        ckpt_dir = os.path.join(base, "base_checkpoints", args.model_tag)
        meta_path = latest_meta_path(ckpt_dir, args.step)
        with open(meta_path) as f:
            meta = json.load(f)
        uc = meta.get("user_config", {})
        print(f"Read meta: {meta_path}")

    layout_version = (args.layout_version
                      or uc.get("mt_layout_version", "v4"))
    if layout_version != "v4":
        raise SystemExit(
            f"layout_version={layout_version!r} not supported; this tool is "
            f"v4-only (the v2/v3 build_doc accounting was removed)."
        )
    kmax = args.kmax if args.kmax is not None else int(uc.get("mt_kmax", 16))
    warmup_threshold = (args.warmup_threshold
                        if args.warmup_threshold is not None
                        else int(uc.get("mt_warmup_threshold", 0)))
    cap_train = resolve_cap(
        uc.get("mt_v4_parallel_cap",
               meta.get("mt_v4_parallel_cap", 0) if meta else 0)
    )
    v4_parallel_cap = (cap_train if args.mt_v4_parallel_cap is None
                       else resolve_cap(args.mt_v4_parallel_cap))

    tokenizer_name = (args.tokenizer
                      or (meta or {}).get("tokenizer_name")
                      or os.environ.get("NANOCHAT_TOKENIZER", "nanochat_64k_mt2"))
    tokenizer_dir = os.path.join(base, "tokenizers", tokenizer_name)
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    print(f"Tokenizer: {tokenizer_name} (vocab {tokenizer.get_vocab_size()})")
    print(f"Layout: {layout_version} kmax={kmax} warmup_threshold={warmup_threshold} "
          f"v4_parallel_cap={v4_parallel_cap if v4_parallel_cap is not None else 'uncapped'}")

    nl_class = assert_walker_invariants(tokenizer)
    nl_token_ids = {tid for tid, c in nl_class.items()
                    if c in (NL_CLASS_PURE, NL_CLASS_TRAIL)}
    assert nl_token_ids

    n_tokens_incl = 0
    n_tokens_excl = 0
    n_steps_waterfall = 0
    n_steps_flat = 0
    n_stages_by_K = Counter()
    n_tokens_excl_by_K = Counter()
    n_steps_waterfall_by_K = Counter()
    n_docs_processed = 0
    n_stages_total = 0
    n_opener_tokens = 0
    n_opener_steps = 0
    v4_phase_counts = Counter()

    shards = list_parquet_files(split=args.split)
    print(f"Streaming {len(shards)} {args.split} shards "
          f"until tokens >= {args.target_tokens:,}")

    done = False
    for shard_path in shards:
        if done:
            break
        pf = pq.ParquetFile(shard_path)
        for rg_idx in range(pf.num_row_groups):
            if done:
                break
            rg = pf.read_row_group(rg_idx)
            doc_texts = rg.column("text").to_pylist()
            cleaned = []
            for t in doc_texts:
                if not t or not t.strip():
                    continue
                t = t.lstrip("\n\r")
                if t:
                    cleaned.append(t)
            if not cleaned:
                continue
            token_lists = tokenizer.encode(cleaned)
            for doc_tokens in token_lists:
                blocks = split_tokens_into_blocks(doc_tokens, nl_class)
                if not blocks:
                    continue
                if layout_version == "v4":
                    doc = build_doc_v4(blocks, warmup_threshold=warmup_threshold)
                    if doc is None:
                        continue
                    ss = v4_doc_summary(doc, v4_parallel_cap=v4_parallel_cap)
                    n_docs_processed += 1
                    n_tokens_incl += ss["tokens_incl"]
                    n_tokens_excl += ss["tokens_excl"]
                    n_steps_waterfall += ss["steps_waterfall"]
                    n_steps_flat += ss["steps_flat"]
                    n_stages_total += (ss["warmup_lines"]
                                       + (1 if ss["middle_lines"] else 0)
                                       + ss["tail_lines"])
                    v4_phase_counts.update(ss)
                if n_tokens_incl >= args.target_tokens:
                    done = True
                    break

    def div(a, b):
        return float(a) / b if b else 0.0

    K_hist = {str(k): n_stages_by_K[k] for k in sorted(n_stages_by_K)}
    K_tokens = {str(k): n_tokens_excl_by_K[k] for k in sorted(n_tokens_excl_by_K)}
    K_steps_waterfall = {str(k): n_steps_waterfall_by_K[k]
                         for k in sorted(n_steps_waterfall_by_K)}
    avg_K_uniform = (sum(k * n for k, n in n_stages_by_K.items())
                     / max(sum(n_stages_by_K.values()), 1))
    avg_K_tok_weighted = (sum(k * n_tokens_excl_by_K[k] for k in n_tokens_excl_by_K)
                          / max(n_tokens_excl, 1))

    out = {
        "model_tag": args.model_tag,
        "step": (meta or {}).get("step"),
        "meta_path": meta_path,
        "layout_version": layout_version,
        "kmax": kmax,
        "warmup_threshold": warmup_threshold,
        "mt_v4_parallel_cap": v4_parallel_cap,
        "mt_v4_parallel_cap_train": cap_train,
        "tokenizer": tokenizer_name,
        "split": args.split,
        "target_tokens": args.target_tokens,
        "n_docs_processed": n_docs_processed,
        "n_stages_total": n_stages_total,
        "n_tokens_incl_specials": n_tokens_incl,
        "n_tokens_excl_specials": n_tokens_excl,
        "n_opener_tokens": n_opener_tokens,
        "n_opener_steps": n_opener_steps,
        "n_engine_steps_waterfall": n_steps_waterfall,
        "n_engine_steps_one_step_offset": n_steps_waterfall,
        "n_engine_steps_flat": n_steps_flat,
        "avg_tokens_per_step_incl_waterfall": div(n_tokens_incl, n_steps_waterfall),
        "avg_tokens_per_step_excl_waterfall": div(n_tokens_excl, n_steps_waterfall),
        "avg_tokens_per_step_incl_one_step_offset": div(n_tokens_incl, n_steps_waterfall),
        "avg_tokens_per_step_excl_one_step_offset": div(n_tokens_excl, n_steps_waterfall),
        "avg_tokens_per_step_incl_flat": div(n_tokens_incl, n_steps_flat),
        "avg_tokens_per_step_excl_flat": div(n_tokens_excl, n_steps_flat),
        "avg_K_uniform": avg_K_uniform,
        "avg_K_token_weighted": avg_K_tok_weighted,
        "n_stages_by_K": K_hist,
        "n_tokens_excl_by_K": K_tokens,
        "n_steps_waterfall_by_K": K_steps_waterfall,
    }
    if layout_version == "v4":
        out["v4_phase_counts"] = dict(v4_phase_counts)
        out["accounting_notes"] = {
            "waterfall": "actual v4 one-step-offset row schedule",
            "flat": "audit schedule: no inter-line one-step offset, same cap in batches",
        }

    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
