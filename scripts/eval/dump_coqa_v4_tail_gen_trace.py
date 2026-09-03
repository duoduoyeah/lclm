"""Synthesize MT-v4 v4_tail generation traces for selected CoQA items.

CPU-only. NO model is loaded — the trace is reconstructed from
`compile_doc_v4`'s RowMeta. Optionally applies the Mod 1 CoQA-text
rewrite ("joined") before tokenization so we can visualize exactly what
the v4_tail eval saw under each preprocessing mode.

For each requested item we emit one trace conforming to
`llm-visual/SCHEMA-generation.md`. Step convention (per design):

  - Row positions 0 and 1 → gen_step = 0 (the "prompt context").
  - Later row positions are grouped by decode step: adjacent slots with
    increasing thread ids in the same stage share a gen_step. A new step
    begins on thread wrap, same-thread continuation, or stage change.

So cap > 1 traces show multiple active threads decoding in the same visual
step, while cap = 1 remains effectively serial.

Usage:
    python -m scripts.eval.dump_coqa_v4_tail_gen_trace \\
        --items 0,2,11 \\
        --coqa-rewrite joined \\
        --tokenizer nanochat_64k_mt_v4 \\
        --mt-v4-parallel-cap 2

Output: llm-visual/examples/blockmt_v4_cap/generation/coqa_v4_tail_<rewrite>.json
"""
import argparse
import json
import os
import re

import yaml

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.eval.core import render_prompts_lm
from nanochat.eval.core_mt import (
    _get_nl_class, _get_specials, TRAIN_DIST_WARMUP_THRESHOLD,
)
from nanochat.data.multithread_dataloader import (
    build_doc_v4, split_tokens_into_blocks,
)
from nanochat.data.multithread_layout import compile_doc_v4


DEFAULT_ITEMS = (0, 2, 11)
COQA_DATASET_URI = "reading_comprehension/coqa.jsonl"


# Mirror of scripts.base_eval._rewrite_coqa_joined. Inlined to keep this
# CPU-only dumper free of torch (importing scripts.base_eval transitively
# pulls in nanochat.engine + torch and OOMs on small nodes).
_COQA_PRECEDING_BLOCK_RE = re.compile(
    r'(Preceding questions:\n)(.*?\n)(\nFinal question:\n)',
    re.DOTALL,
)
_COQA_QA_PAIR_RE = re.compile(r'(Question: [^\n]*)\n(Answer: [^\n]*)\n')
_COQA_FINAL_SUFFIX_RE = re.compile(
    r'\nFinal question:\nQuestion: ([^\n]*)\nAnswer: $'
)


def _rewrite_coqa_joined(ctx: str) -> str:
    def _collapse(m):
        head, body, tail = m.group(1), m.group(2), m.group(3)
        return head + _COQA_QA_PAIR_RE.sub(r'\1 \2\n', body) + tail
    out = _COQA_PRECEDING_BLOCK_RE.sub(_collapse, ctx)
    out = _COQA_FINAL_SUFFIX_RE.sub(
        r'\nFinal question: Question: \1 Answer: ', out)
    return out


def render_and_tokenize(item, delim, tokenizer):
    """Mirror evaluate_example_mt's language_modeling text -> tokens path.

    Returns: (full_tokens, candidate_token_count, prompt_text).
    """
    prompts = render_prompts_lm(item, delim, fewshot_examples=None)
    full_per_cand = tokenizer.encode(prompts)
    without_cont, with_cont = full_per_cand
    cand_n = len(with_cont) - len(without_cont)
    return with_cont, cand_n, prompts[1]


def encode_v4_tail(tokenizer, full_tokens, candidate_token_count,
                   rope_stride=256,
                   warmup_threshold=TRAIN_DIST_WARMUP_THRESHOLD,
                   v4_parallel_cap=None):
    """Run the v4_tail layout pipeline and return (RowMeta, DocV4).

    Returns None when the candidate spans more than the final block
    (matches the eval encoder's clean-skip rule).
    """
    nl_class = _get_nl_class(tokenizer)
    sp = _get_specials(tokenizer)
    if sp["sot_tail_id"] is None:
        raise SystemExit("tokenizer has no <|sot_tail|> — wrong tokenizer "
                         "for v4_tail layout (expected nanochat_64k_mt_v4).")
    blocks = split_tokens_into_blocks(full_tokens, nl_class)
    assert blocks, "splitter returned no blocks"
    N = len(blocks)
    if N >= 2 and candidate_token_count > len(blocks[-1]):
        return None
    doc = build_doc_v4(blocks, warmup_threshold=warmup_threshold)
    assert doc is not None
    meta = compile_doc_v4(
        doc, rope_stride=rope_stride,
        sot_id=sp["sot_id"], sot_tail_id=sp["sot_tail_id"],
        bos_id=sp["bos_id"],
        v4_parallel_cap=v4_parallel_cap,
    )
    return meta, doc


def flatten_step_groups(groups, prompt_positions=2):
    """Convert decode-step group sizes into per-token gen_step ids."""
    steps = []
    next_step = 1
    for size in groups:
        remaining = size
        if len(steps) < prompt_positions:
            prompt_n = min(prompt_positions - len(steps), remaining)
            steps.extend([0] * prompt_n)
            remaining -= prompt_n
        if remaining:
            steps.extend([next_step] * remaining)
            next_step += 1
    return steps


def decode_step_groups_v4(doc, v4_parallel_cap=None, include_bos=True):
    """Mirror compile_doc_v4 emission groups at decode-step granularity."""
    groups = []

    def group(size=1):
        if size > 0:
            groups.append(size)

    if include_bos:
        group(1)
    for _tok in doc.warmup[0].content:
        group(1)

    for para in doc.warmup[1:]:
        group(1)  # forced/input SOT
        for _tok in para.content:
            group(1)

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
                group(n)
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

                group(n)
                if finished:
                    active = [t for t in active if t not in finished]
    elif doc.tail is not None:
        group(1)

    if doc.tail is not None:
        group(1)  # <|sot_tail|>
        for _tok in doc.tail.content:
            group(1)

    return groups


def assign_decode_steps_v4(doc, meta, v4_parallel_cap=None):
    groups = decode_step_groups_v4(doc, v4_parallel_cap=v4_parallel_cap)
    steps = flatten_step_groups(groups)
    assert len(steps) == meta.N, (len(steps), meta.N)
    return steps


def gen_trace_from_meta(meta, tokenizer, name, prompt_text, gen_steps):
    """Convert a v4 RowMeta to a SCHEMA-generation trace.

    Tagging:
      - gen_step: inferred decode step; cap > 1 can place multiple tokens
        in the same step when parallel threads advance together.
      - thread_id: meta.thread_idx[i] (parent = 0, others 1+).
      - wave_id:   meta.stage_idx[i].
      - block_id:  meta.thread_idx[i] (v4: 1 source block per thread).
      - role: 'stream_start' for <bos>, 'thread_start' for <sot> or
              <sot_tail>, 'block_open' for the first content token of
              each block (local_step == 1).
    """
    sp = _get_specials(tokenizer)
    bos_id = sp["bos_id"]
    sot_id = sp["sot_id"]
    sot_tail_id = sp["sot_tail_id"]
    tokens_list = []
    for i, (tok, t, s, ls, _rope, gen_step) in enumerate(zip(
        meta.tokens, meta.thread_idx, meta.stage_idx,
        meta.local_step, meta.rope_idx, gen_steps,
    )):
        thread_id = max(int(t), 0)

        role = None
        if tok == bos_id:
            role = "stream_start"
        elif tok == sot_id or tok == sot_tail_id:
            role = "thread_start"
        elif ls == 1:
            role = "block_open"

        entry = {
            "id": f"t{i}",
            "vocab_id": int(tok),
            "gen_step": int(gen_step),
            "position": {"abs": int(i)},
            "forced": False,
            "thread_id": thread_id,
            "wave_id": int(s),
            "block_id": thread_id,
        }
        if role is not None:
            entry["role"] = role
        tokens_list.append(entry)

    return {
        "name": name,
        "prompt_text": prompt_text,
        "tokens": tokens_list,
    }


def build_vocab(tokenizer, all_ids):
    sp = _get_specials(tokenizer)
    special_names = {sp["bos_id"]: "<|bos|>", sp["sot_id"]: "<|sot|>"}
    if sp["sot_tail_id"] is not None:
        special_names[sp["sot_tail_id"]] = "<|sot_tail|>"
    for k in range(1, 17):
        try:
            special_names[sp["bot_fn"](k)] = f"<|bot_{k}|>"
        except Exception:
            pass
    try:
        special_names[tokenizer.encode_special("<|eot|>")] = "<|eot|>"
    except Exception:
        pass

    vocab, vocab_meta = {}, {}
    for tid in sorted(set(int(x) for x in all_ids)):
        if tid in special_names:
            vocab[str(tid)] = special_names[tid]
            vocab_meta[str(tid)] = {"is_special": True}
        else:
            try:
                text = tokenizer.decode([tid])
            except Exception:
                text = f"id={tid}"
            vocab[str(tid)] = text
    return vocab, vocab_meta


def count_turns(item):
    """Return the # of preceding Q/A pairs embedded in `item['context']`."""
    pre = item["context"].split("Final question:")[0]
    return pre.count("\nAnswer: ")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--items", type=str,
                   default=",".join(str(i) for i in DEFAULT_ITEMS),
                   help="comma-separated CoQA jsonl row indices (default 0,2,11)")
    p.add_argument("--coqa-rewrite", type=str, default="none",
                   choices=["none", "joined"],
                   help="apply Mod 1 join-Q/A-pairs rewrite before tokenization")
    p.add_argument("--tokenizer", type=str, default="nanochat_64k_mt_v4",
                   help="tokenizer name (must have <|sot_tail|>)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="default: llm-visual/examples/blockmt_v4_cap/generation/")
    p.add_argument("--rope-stride", type=int, default=256)
    p.add_argument("--mt-warmup-threshold", type=int,
                   default=TRAIN_DIST_WARMUP_THRESHOLD,
                   help="v4 warmup threshold W used for layout reconstruction")
    p.add_argument("--mt-v4-parallel-cap", type=int, default=0,
                   help="v4-only active ordinary middle-thread cap; 0 = uncapped")
    args = p.parse_args()

    assert args.mt_warmup_threshold >= 0, args.mt_warmup_threshold
    assert args.mt_v4_parallel_cap >= 0, args.mt_v4_parallel_cap
    v4_parallel_cap = (args.mt_v4_parallel_cap
                       if args.mt_v4_parallel_cap > 0 else None)

    base = get_base_dir()
    bundle = os.path.join(base, "eval_bundle")
    cfg = yaml.safe_load(open(os.path.join(bundle, "core.yaml")))
    coqa_meta = next(t for t in cfg["icl_tasks"] if t["label"] == "coqa")
    delim = coqa_meta.get("continuation_delimiter", " ")
    data_path = os.path.join(bundle, "eval_data", COQA_DATASET_URI)
    data = [json.loads(line) for line in open(data_path)]

    tokenizer = RustBPETokenizer.from_directory(
        os.path.join(base, "tokenizers", args.tokenizer))
    print(f"Tokenizer: {args.tokenizer}")
    print(f"CoQA items loaded: {len(data)}")
    print(f"Rewrite mode: {args.coqa_rewrite}")
    print(f"Warmup threshold: {args.mt_warmup_threshold}")
    print(f"v4 parallel cap: {v4_parallel_cap if v4_parallel_cap is not None else 'uncapped'}")

    item_indices = [int(s.strip()) for s in args.items.split(",") if s.strip()]
    if args.out_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        args.out_dir = os.path.join(repo_root, "llm-visual", "examples",
                                    "blockmt_v4_cap", "generation")
    os.makedirs(args.out_dir, exist_ok=True)

    traces = []
    all_ids = []
    for idx in item_indices:
        if idx >= len(data):
            print(f"  skip item {idx}: out of range")
            continue
        # Apply Mod 1 to a copy of the item so we don't mutate `data`.
        item = dict(data[idx])
        n_turns = count_turns(item)
        if args.coqa_rewrite == "joined":
            item["context"] = _rewrite_coqa_joined(item["context"])

        full, cand_n, prompt_text_full = render_and_tokenize(
            item, delim, tokenizer)
        encoded = encode_v4_tail(
            tokenizer, full, cand_n, rope_stride=args.rope_stride,
            warmup_threshold=args.mt_warmup_threshold,
            v4_parallel_cap=v4_parallel_cap)
        if encoded is None:
            print(f"  item {idx}: skipped (candidate spans blocks)")
            continue
        meta, doc = encoded
        gen_steps = assign_decode_steps_v4(
            doc, meta, v4_parallel_cap=v4_parallel_cap)

        # First 2 row tokens are the "prompt" we surface above the figure.
        prompt_preview = tokenizer.decode(list(meta.tokens[:2]))
        trace = gen_trace_from_meta(
            meta, tokenizer,
            name=(f"coqa item{idx} ({n_turns} turns, {args.coqa_rewrite}, "
                  f"W={args.mt_warmup_threshold}, "
                  f"cap={v4_parallel_cap if v4_parallel_cap is not None else 'uncapped'})"),
            prompt_text=prompt_preview,
            gen_steps=gen_steps,
        )
        traces.append(trace)
        all_ids.extend(meta.tokens)
        # Sanity summary
        n_pos = len(meta.tokens)
        n_threads = max(meta.thread_idx) + 1
        n_stages = max(meta.stage_idx) + 1
        sp = _get_specials(tokenizer)
        n_sot = sum(1 for t in meta.tokens if t == sp["sot_id"])
        n_sot_tail = sum(1 for t in meta.tokens if t == sp["sot_tail_id"])
        n_bos = sum(1 for t in meta.tokens if t == sp["bos_id"])
        sot_tail_pos = (meta.tokens.index(sp["sot_tail_id"])
                        if sp["sot_tail_id"] in meta.tokens else None)
        print(f"  item {idx:>3} ({n_turns:>2}t): row={n_pos:>4} "
              f"threads={n_threads:>2} stages={n_stages:>2} "
              f"bos={n_bos} sot={n_sot} sot_tail={n_sot_tail} "
              f"sot_tail_at={sot_tail_pos}")

    if not traces:
        print("No traces produced.")
        return

    vocab, vocab_meta = build_vocab(tokenizer, all_ids)
    payload = {
        "schema_version": "3",
        "vocab": vocab,
        "vocab_meta": vocab_meta,
        "traces": traces,
    }
    cap_suffix = (f"_cap{v4_parallel_cap}" if v4_parallel_cap is not None else "")
    warmup_suffix = (f"_W{args.mt_warmup_threshold}"
                     if args.mt_warmup_threshold != TRAIN_DIST_WARMUP_THRESHOLD else "")
    out_path = os.path.join(
        args.out_dir,
        f"coqa_v4_tail_{args.coqa_rewrite}{warmup_suffix}{cap_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    llm_root = os.path.join(repo_root, "llm-visual")
    print(f"\nWrote {len(traces)} trace(s) -> {out_path}")
    if os.path.commonpath([os.path.abspath(out_path), llm_root]) == llm_root:
        rel = os.path.relpath(out_path, llm_root)
        print(f"Open via llm-visual/index.html or viewer.html?trace=./{rel}")
    print(f"View via: cd llm-visual && python serve.py")


if __name__ == "__main__":
    main()
