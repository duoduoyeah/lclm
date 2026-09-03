"""
MT-LM-faithful CORE encoder.

When `evaluate_core` runs on a `MultithreadLM`, the vanilla
`forward_model(model, input_ids)` path feeds the model `arange(T)`
RoPE positions (multithread_lm.py:305-309 fallback). Past row
position 256 this is out of distribution — the model was trained on
paragraph-strided RoPE where every paragraph anchor sits at a
multiple of `rope_stride`.

This module re-encodes CORE prompts through the same v4 row layout the
dataloader uses (`compile_doc_v4` / `build_doc_v4` from
`multithread_layout.py` / `multithread_dataloader.py`): the final source
block is scored as the row-last tail-SOT line. For each rendered prompt
the encoder returns:

  - `tokens`        — list[int], the 1-D row the model will see
  - `rope_idx`      — list[int], paragraph-strided RoPE indices
  - `start_idx`     — int, first candidate row position
  - `end_idx`       — int, one past the last candidate row position

CORE's `forward_model` already produces losses at every row position;
the only changes downstream are (a) `forward_model` gains a `rope_idx`
kwarg, (b) `losses[si-1:ei-1].mean()` indexes into the MT-row
positions returned here.

See `design/eval/core-mt-eval.md` for the full design rationale and
decisions (locked 2026-05-23). In short:

  - v4 tail-SOT layout (the final source block scored as the row-last tail)
  - Option (A) candidate encoding: concatenate `prompt + candidate`
    and let `split_tokens_into_blocks` decide line boundaries
  - Multi-line candidates (newline inside `choices` / `continuation`)
    are dropped — empirical scan found 7/89500 such items, all LAMBADA
  - Dispatch via `isinstance(model, MultithreadLM)` in `core.py`
"""

import torch

from nanochat.data.multithread_dataloader import (
    assert_walker_invariants,
    build_doc_v4,
    split_tokens_into_blocks,
)
from nanochat.data.multithread_layout import (
    compile_doc_v4,
)
from nanochat.eval.core import (
    render_prompts_mc,
    render_prompts_schema,
    render_prompts_lm,
    stack_sequences,
    find_common_length,
)


# ---------- cached per-tokenizer resources -----------------------------------

# A single tokenizer is reused across the whole `evaluate_core` run (22 tasks ×
# thousands of items). `assert_walker_invariants` walks the entire vocab;
# resolving specials hits BPE lookups. Cache by tokenizer identity to avoid re-doing this
# work on every item. We key on `id(tokenizer)` because the tokenizer object
# itself isn't hashable in general; the lifetime of the eval matches the
# lifetime of the tokenizer so the id is stable enough.

_NL_CLASS_CACHE = {}
_SPECIALS_CACHE = {}


def _get_nl_class(tokenizer):
    key = id(tokenizer)
    cached = _NL_CLASS_CACHE.get(key)
    if cached is None:
        cached = assert_walker_invariants(tokenizer)
        _NL_CLASS_CACHE[key] = cached
    return cached


def _get_specials(tokenizer):
    """Return dict with structural special ids for this tokenizer."""
    key = id(tokenizer)
    cached = _SPECIALS_CACHE.get(key)
    if cached is None:
        sot_id = tokenizer.encode_special("<|sot|>")
        bos_id = tokenizer.get_bos_token_id()
        try:
            sot_tail_id = tokenizer.encode_special("<|sot_tail|>")
        except Exception:
            sot_tail_id = None
        cached = {
            "sot_id": sot_id,
            "bos_id": bos_id,
            "sot_tail_id": sot_tail_id,
        }
        _SPECIALS_CACHE[key] = cached
    return cached


# ---------- core MT-faithful encoder -----------------------------------------

# After the v3 retirement the CORE-MT scorer is v4-only. `arange` is a
# separate diagnostic handled in `nanochat.eval.core` (it bypasses the
# MT-faithful encoder), so it is not one of these encoder modes.
VALID_ENCODINGS = ("v4_tail",)

# Warmup budget for the v4 prefix (mirrors the base_train v4 default; see
# `multithread_dataloader.build_doc_v4`).
TRAIN_DIST_WARMUP_THRESHOLD = 16


def mt_encode_for_scoring(tokenizer, full_tokens, candidate_token_count,
                          rope_stride=256, encoding="v4_tail",
                          v4_parallel_cap=None):
    """Compile a single rendered-prompt token list into an MT-LM v4 row.

    Args:
        tokenizer: the tokenizer (used only for cached nl-class + specials).
        full_tokens: list[int] — tokens of the FULL rendered text
            (prompt + delimiter + candidate concatenated and tokenized).
        candidate_token_count: int — how many of the trailing tokens belong
            to the candidate slice (rest is prompt context). The caller
            determines this via common-prefix detection across choices.
        rope_stride: int — should match the model's training stride (256).
        encoding: "v4_tail" — the only supported encoder mode after the v3
            retirement. The final source block is scored as the row-last
            tail-SOT line. (The legacy v3 "chain"/"bot_k"/"train_dist"
            encodings were removed; "arange" is a separate diagnostic
            handled in `nanochat.eval.core`.)
        v4_parallel_cap: int | None — forwarded into `compile_doc_v4` so the
            eval row matches the training cap.

    Returns:
        (tokens, rope_idx, start_idx, end_idx) — or None if the candidate
        spans multiple blocks (which would put the candidate across the
        tail boundary and break the loss slice).

            tokens: list[int], row tokens with bos/sot/bot-K interleaved
            rope_idx: list[int], paragraph-strided RoPE indices
            start_idx: int, first candidate row position
            end_idx: int, one past the last candidate row position

    The candidate slice in the row is `tokens[start_idx:end_idx]`;
    loss is computed at `losses[start_idx-1:end_idx-1]`, same convention
    as the vanilla CORE path.
    """
    assert encoding in VALID_ENCODINGS, \
        f"encoding must be one of {VALID_ENCODINGS}, got {encoding!r}"
    assert v4_parallel_cap is None or encoding == "v4_tail", (
        f"v4_parallel_cap is only meaningful for encoding='v4_tail'; "
        f"got encoding={encoding!r} with v4_parallel_cap={v4_parallel_cap}"
    )
    assert candidate_token_count >= 1, \
        f"candidate must have >=1 token, got {candidate_token_count}"
    assert candidate_token_count <= len(full_tokens), \
        f"candidate ({candidate_token_count}) > full ({len(full_tokens)})"

    nl_class = _get_nl_class(tokenizer)
    sp = _get_specials(tokenizer)

    # Walker → blocks. Same as multithread_dataloader.py:365.
    blocks = split_tokens_into_blocks(full_tokens, nl_class)
    assert blocks, "splitter returned no blocks (empty input?)"

    N = len(blocks)
    sot_tail_id = sp.get("sot_tail_id")

    # v4 scorer path: reserve the final source block as the row-last
    # <|sot_tail|> line and score only the candidate suffix inside it.
    if sot_tail_id is None:
        raise ValueError("encoding=v4_tail requires tokenizer with <|sot_tail|>")
    if N >= 2 and candidate_token_count > len(blocks[-1]):
        return None  # candidate spans blocks; skip cleanly
    doc = build_doc_v4(blocks, warmup_threshold=TRAIN_DIST_WARMUP_THRESHOLD)
    assert doc is not None, "build_doc_v4 returned None for non-empty blocks"
    meta = compile_doc_v4(
        doc, rope_stride=rope_stride,
        sot_id=sp["sot_id"], sot_tail_id=sot_tail_id, bos_id=sp["bos_id"],
        v4_parallel_cap=v4_parallel_cap,
    )

    # Walk the row and collect content positions PER BLOCK, then concatenate
    # in block order. The v4 middle lines are waterfall-interleaved in the
    # row, so we sort each thread's content by local_step to recover document
    # order. A "content" position has thread_idx >= 0 (real thread) AND
    # local_step >= 1 (not an anchor slot like <bos>/<sot>); the row-last
    # <sot_tail> marker is skipped explicitly.
    content_by_thread = {b: [] for b in range(N)}
    for i in range(meta.N):
        if meta.thread_idx[i] >= 0 and meta.local_step[i] >= 1:
            if sot_tail_id is not None and meta.tokens[i] == sot_tail_id:
                continue
            t = meta.thread_idx[i]
            # Map global paragraph_idx (= thread_idx in compile_doc_v4) back
            # to a block index: paragraph_idx is the block's position in our
            # blocks list (compile_doc_v4 numbers threads warmup→middle→tail).
            content_by_thread[t].append((meta.local_step[i], i))
    content_positions = []
    for b in range(N):
        for _, row_idx in sorted(content_by_thread[b]):
            content_positions.append(row_idx)

    # Sanity: content_positions must be 1-1 with full_tokens, in block order.
    assert len(content_positions) == len(full_tokens), (
        f"content_positions ({len(content_positions)}) != "
        f"full_tokens ({len(full_tokens)}); compile_doc structural layout drift"
    )
    for ci, row_idx in enumerate(content_positions):
        assert meta.tokens[row_idx] == full_tokens[ci], (
            f"content[{ci}] token mismatch: row[{row_idx}]={meta.tokens[row_idx]}, "
            f"full_tokens[{ci}]={full_tokens[ci]}"
        )

    # Candidate slice = last `candidate_token_count` content positions in
    # block order. Under v4_tail this is row-contiguous: the last block is
    # the row-last tail line, decoded sequentially after the <sot_tail>
    # anchor with no interleaving with other threads.
    first_cand_content = len(full_tokens) - candidate_token_count
    start_idx = content_positions[first_cand_content]
    end_idx = content_positions[-1] + 1

    return list(meta.tokens), list(meta.rope_idx), start_idx, end_idx


# ---------- batch wrappers for the three task types --------------------------

def _candidate_token_count(full_tokens_i, prompt_token_len):
    """Length of the candidate slice in token space.

    `prompt_token_len` is the common-prefix length across all candidates
    (in token space, found via find_common_length). For each candidate's
    full tokenization, the trailing portion past prompt_token_len is the
    candidate. Falls back to len-1 if the BPE-boundary merge ate part of
    the candidate (which would make prompt_token_len > len(full_tokens) -
    1; never observed in practice but cheap to guard)."""
    cand = len(full_tokens_i) - prompt_token_len
    return max(cand, 1)


def batch_sequences_mt_mc(tokenizer, prompts, rope_stride=256, encoding="v4_tail",
                          v4_parallel_cap=None):
    """MT-faithful encoding for multiple-choice prompts.

    `prompts` is a list of N rendered prompts (each = prompt + delimiter
    + choice_i), one per candidate. Returns parallel lists of (tokens,
    rope_idxs, start_idxs, end_idxs). No BOS prepended here — bos lives
    inside the compiled row at position 0 (compile_doc_v4 emits it as the
    parent's anchor). Rows where the candidate spans multiple blocks are
    skipped (the per-row tuple is None and the caller is expected to handle
    that by skipping the item)."""
    full_tokens_per_cand = tokenizer.encode(prompts)
    prompt_token_len = find_common_length(full_tokens_per_cand, direction='left')
    rows_tokens, rows_rope, start_idxs, end_idxs = [], [], [], []
    for ft in full_tokens_per_cand:
        cand_n = _candidate_token_count(ft, prompt_token_len)
        out = mt_encode_for_scoring(
            tokenizer, ft, cand_n, rope_stride=rope_stride, encoding=encoding,
            v4_parallel_cap=v4_parallel_cap)
        if out is None:
            return None  # signal "skip this item" to caller
        toks, rope, si, ei = out
        rows_tokens.append(toks)
        rows_rope.append(rope)
        start_idxs.append(si)
        end_idxs.append(ei)
    return rows_tokens, rows_rope, start_idxs, end_idxs


def batch_sequences_mt_schema(tokenizer, prompts, rope_stride=256, encoding="v4_tail",
                              v4_parallel_cap=None):
    """MT-faithful encoding for schema (Winograd-style) prompts.

    Schema = common-suffix semantics: contexts vary, continuation is the
    same. Decision §10.8: no structural change — `mt_encode_for_scoring`
    returns explicit `(start, end)` per row, so the common-suffix length
    in token space simply becomes the candidate slice length on each
    row. Like MC, no BOS prepended here."""
    full_tokens_per_cand = tokenizer.encode(prompts)
    suffix_len = find_common_length(full_tokens_per_cand, direction='right')
    rows_tokens, rows_rope, start_idxs, end_idxs = [], [], [], []
    for ft in full_tokens_per_cand:
        cand_n = max(suffix_len, 1)
        out = mt_encode_for_scoring(
            tokenizer, ft, cand_n, rope_stride=rope_stride, encoding=encoding,
            v4_parallel_cap=v4_parallel_cap)
        if out is None:
            return None
        toks, rope, si, ei = out
        rows_tokens.append(toks)
        rows_rope.append(rope)
        start_idxs.append(si)
        end_idxs.append(ei)
    return rows_tokens, rows_rope, start_idxs, end_idxs


def batch_sequences_mt_lm(tokenizer, prompts, rope_stride=256, encoding="v4_tail",
                          v4_parallel_cap=None):
    """MT-faithful encoding for language-modeling (LAMBADA) prompts.

    `prompts` is a 2-element list [without_continuation, with_continuation];
    the candidate is the portion in `with` past the end of `without`.
    Mirrors batch_sequences_lm — only the with-continuation row is
    encoded and returned (batch size 1)."""
    full_tokens = tokenizer.encode(prompts)
    tokens_without, tokens_with = full_tokens
    assert len(tokens_without) < len(tokens_with), \
        "lm prompts: 'without' must be a proper prefix of 'with'"
    assert tokens_without == tokens_with[:len(tokens_without)], \
        "lm prompts: 'without' must be a token-level prefix of 'with'"
    cand_n = len(tokens_with) - len(tokens_without)
    out = mt_encode_for_scoring(
        tokenizer, tokens_with, cand_n, rope_stride=rope_stride, encoding=encoding,
        v4_parallel_cap=v4_parallel_cap)
    if out is None:
        return None
    toks, rope, si, ei = out
    return [toks], [rope], [si], [ei]


# ---------- multi-line candidate filter --------------------------------------

def _item_has_multiline_candidate(item, task_type):
    """True if any candidate text in this item contains '\n'.

    Empirical scan (2026-05-23, 22 CORE tasks, ~89,500 items): exactly
    7 LAMBADA items hit this, e.g. `'of\\n\\nPower'`. All formatting
    noise from the source corpus."""
    if task_type == 'multiple_choice':
        for c in item.get('choices', []):
            if isinstance(c, str) and '\n' in c:
                return True
    elif task_type == 'schema':
        cont = item.get('continuation')
        if isinstance(cont, str) and '\n' in cont:
            return True
    elif task_type == 'language_modeling':
        cont = item.get('continuation')
        if isinstance(cont, str) and '\n' in cont:
            return True
    return False


def filter_multiline_items(data, task_type):
    """Drop items where any candidate text contains '\n'. Returns the
    filtered list, plus the count dropped (for logging)."""
    kept = [it for it in data if not _item_has_multiline_candidate(it, task_type)]
    dropped = len(data) - len(kept)
    return kept, dropped


# ---------- evaluate_example_mt ----------------------------------------------

# Late import to avoid circular references — core.py also imports from this
# module for the dispatch. We rebind these inside the function instead.

@torch.no_grad()
def evaluate_example_mt(idx, model, tokenizer, data, device, task_meta,
                        rope_stride=256, encoding="v4_tail",
                        v4_parallel_cap=None, shot_layout="normal"):
    """MT-LM version of evaluate_example.

    Mirrors the structure of `core.evaluate_example` but routes through
    the MT-faithful encoder and passes `rope_idx` into `forward_model`.

    Args:
        encoding: "v4_tail" — the only supported encoder mode (the final
            source block is scored as the row-last tail-SOT line).
        v4_parallel_cap: int | None — forwarded to compile_doc_v4 so the
            eval row layout matches the cap the checkpoint was trained with.

    Returns:
        True/False — whether the model's prediction is correct, OR
        None if the item was skipped (e.g. multi-line candidate, or a
        candidate spanning multiple blocks).
    """
    from nanochat.eval.core import forward_model  # delayed import to avoid cycle
    import random

    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Safety guard: even if the caller did pre-filter, items in `data`
    # *also* serve as the few-shot pool — and a few-shot example could
    # itself carry a multi-line continuation. We don't drop the eval
    # item, but we do skip multi-line ones from being chosen as shots
    # so the rendered prompt stays consistent with the K=1-chain layout.

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available = [
            i for i in range(len(data))
            if i != idx and not _item_has_multiline_candidate(data[i], task_type)
        ]
        if len(available) >= num_fewshot:
            fewshot_indices = rng.sample(available, num_fewshot)
            fewshot_examples = [data[i] for i in fewshot_indices]
        else:
            # Degenerate fallback — almost never triggered (LAMBADA only
            # has 7 such items out of 5153, and LAMBADA is 0-shot anyway).
            return None

    # Render + batch sequences via the MT encoder.
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(
            item, continuation_delimiter, fewshot_examples,
            shot_layout=shot_layout)
        out = batch_sequences_mt_mc(
            tokenizer, prompts, rope_stride=rope_stride, encoding=encoding,
            v4_parallel_cap=v4_parallel_cap)
    elif task_type == 'schema':
        prompts = render_prompts_schema(
            item, continuation_delimiter, fewshot_examples,
            shot_layout=shot_layout)
        out = batch_sequences_mt_schema(
            tokenizer, prompts, rope_stride=rope_stride, encoding=encoding,
            v4_parallel_cap=v4_parallel_cap)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(
            item, continuation_delimiter, fewshot_examples,
            shot_layout=shot_layout)
        out = batch_sequences_mt_lm(
            tokenizer, prompts, rope_stride=rope_stride, encoding=encoding,
            v4_parallel_cap=v4_parallel_cap)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    if out is None:
        return None  # encoder rejected this item (candidate spans blocks)
    tokens, ropes, start_idxs, end_idxs = out

    # Truncation: MultithreadLM doesn't set `max_seq_len`, but defensive
    # for future variants. Skip if any row exceeds; LM rope cache may
    # also assert if rope_idx overflows rotary_seq_len (defensive guard
    # mirrors multithread_dataloader.py:450-458).
    cos_cache_len = model.cos.size(1) if hasattr(model, 'cos') else None
    if cos_cache_len is not None:
        for r in ropes:
            if max(r) >= cos_cache_len:
                return None  # would assert in RoPE gather; skip cleanly

    # Stack into a batch. Use BOS as the token pad. Rope pads must be a
    # real in-cache position because the model gathers RoPE for every slot
    # before loss slicing, even for right-padding that is never scored.
    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    rope_idx = stack_sequences(ropes, 0).to(device)
    losses, predictions = forward_model(model, input_ids, rope_idx=rope_idx)

    # Score selection — same logic as vanilla evaluate_example.
    if task_type == 'language_modeling':
        si, ei = start_idxs[0], end_idxs[0]
        predicted_tokens = predictions[0, si - 1:ei - 1]
        actual_tokens = input_ids[0, si:ei]
        return bool(torch.all(predicted_tokens == actual_tokens).item())
    elif task_type in ('multiple_choice', 'schema'):
        mean_losses = [
            losses[i, si - 1:ei - 1].mean().item()
            for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))
        ]
        pred_idx = mean_losses.index(min(mean_losses))
        return pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
