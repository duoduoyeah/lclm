"""
SFT-MT dataloader — promoted from `scripts/dump/_sft_mt_layout.py` and
extended with the production training loader used by
`scripts/chat_sft_mt.py`.

The helpers (`compile_conv_turns`, `derive_forced_per_position`, etc.)
build per-turn MultithreadLM v4 `DocV4`s with these SFT-specific rules
(see `design/posttrain/block-mt-sft-design.md` and
`design/blockmt/blockmt-v4-tail-sot.md`):

  1. **Per-turn parent**: each turn-pair (`<|user_start|>` …
     `<|assistant_end|>`) compiles to its own DocV4. The parent
     (`warmup[0]`) is the user prompt only (up to `<|user_end|>`); the
     assistant reply lives in the v4 SOT ramp + row-last tail that
     follow. Multi-turn convs produce multiple DocV4s, each with its
     own `<bos>` anchor.

  2. **Row-last answer (v4 tail)**: the last assistant block (after the
     Q1 newline-walk split) becomes the row-last `tail` line, so it
     decodes with full causal visibility over the preceding content.
     This subsumes the v3 `####`-isolated-answer rule — the
     GSM8K/SpellingBee answer is the final block and naturally lands in
     the tail.

  3. **Multi-turn = multiple DocV4s**: each turn-pair is compiled
     independently and packed as its own piece.

  4. **Warmup (optional)**: when `warmup_threshold > 0`, leading
     assistant blocks join `warmup` as sequential K=1 lines until the
     cumulative (parent prompt + asst) token count exceeds W; the rest
     become `middle` SOT-ramp lines. Mirrors pretrain v4
     (`multithread_dataloader.build_doc_v4`); typical SFT user prompts
     already exceed W so warmup rarely fires for chat data.

The production loader
(`tokenizing_distributed_data_loader_with_state_sft_mt`) mirrors
`nanochat.data.multithread_dataloader
.tokenizing_distributed_data_loader_with_state_multithread`:
  - Iterates a `TaskMixture` (list-indexable) instead of streaming parquet.
  - Compiles each conversation to per-turn pieces via `compile_conv_turns`.
  - Applies the SFT loss mask: at content positions whose
    `render_conversation` mask=0, override target to -1.
  - Best-fit packs `(inputs, targets, rope_idx)` with DDP rank sharding.
  - Bakes a per-piece rope offset at pack time so cross-piece rope
    values don't collide.
  - Yields `(inputs, targets, rope_idx, state_dict)` per batch.

`state_dict` carries `{conv_idx, epoch, consumed}` for resume + outer
progress tracking (analogous to the pretrain MT loader's
`{pq_idx, rg_idx, epoch}`). `consumed` is the per-rank view of "total
conversations consumed across all ranks" — incremented by world_size
per pop, matching the existing vanilla SFT loader's convention so
`chat_sft_mt.py` can reuse the `consumed >= dataset_size` end-of-epoch
trigger.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import torch

from nanochat.common import get_dist_info
from nanochat.data.multithread_dataloader import compute_nl_token_ids
from nanochat.data.multithread_layout import (
    Paragraph, DocV4, RowMeta, SOT_TAIL,
    compile_doc_v4, compute_targets_v4,
)


SpecialIds = dict   # {'user_start','user_end','assistant_start','assistant_end','bos'}


# ===================================================================== helpers
# Verbatim from the original `scripts/dump/_sft_mt_layout.py`. The dump
# scripts re-import these by their public names from here.

def split_into_blocks_with_mask(ids, mask, nl_ids):
    """Walk (ids, mask) together; close a block at every NL-bearing token.
    Mirrors the Q1 walk used by the production MT dataloader but carries
    the SFT loss mask in lockstep. Appends a defensive bare `\\n`
    (mask=0) when the trailing block has no NL terminator."""
    blocks_t, blocks_m = [], []
    cur_t, cur_m = [], []
    for tid, mv in zip(ids, mask):
        cur_t.append(tid); cur_m.append(mv)
        if tid in nl_ids:
            blocks_t.append(cur_t); blocks_m.append(cur_m)
            cur_t, cur_m = [], []
    if cur_t:
        cur_t.append(10); cur_m.append(0)
        blocks_t.append(cur_t); blocks_m.append(cur_m)
    return blocks_t, blocks_m


def find_turn_boundaries(ids, special_ids) -> List[Tuple[int, int, int, int]]:
    """Walk `ids` and return [(user_start_idx, user_end_idx,
    asst_start_idx, asst_end_idx), …]  for every turn-pair. Asserts
    well-formed alternation. Indices are inclusive on both ends."""
    US = special_ids["user_start"]
    UE = special_ids["user_end"]
    AS_ = special_ids["assistant_start"]
    AE = special_ids["assistant_end"]

    out = []
    i, n = 0, len(ids)
    while i < n:
        if ids[i] != US:
            i += 1
            continue
        us = i
        ue = None
        for j in range(i + 1, n):
            if ids[j] == UE:
                ue = j
                break
        assert ue is not None, f"user_start at {us} without user_end"
        as_ = None
        for j in range(ue + 1, n):
            if ids[j] == AS_:
                as_ = j
                break
        assert as_ is not None, f"user_end at {ue} without assistant_start"
        ae = None
        for j in range(as_ + 1, n):
            if ids[j] == AE:
                ae = j
                break
        assert ae is not None, f"assistant_start at {as_} without assistant_end"
        out.append((us, ue, as_, ae))
        i = ae + 1
    return out


def is_answer_block(block_tokens, tokenizer) -> bool:
    """Last-block answer detector. Returns True iff the decoded text of
    `block_tokens` (stripped of leading whitespace) starts with `####`.
    Validated against `tasks/gsm8k.py:22` (GSM8K answer regex) — same
    marker is reused by SpellingBee (`tasks/spellingbee.py:195`)."""
    try:
        text = tokenizer.decode(block_tokens)
    except Exception:
        return False
    return text.lstrip().startswith("####")


def build_turn_doc_v4(user_ids, user_mask, asst_ids, asst_mask,
                      tokenizer, nl_ids, kmax=None,
                      warmup_threshold=0) -> Tuple[DocV4, List[List[int]]]:
    """Build a single-turn SFT-MT v4 `DocV4` and its per-block mask list.

    v4 tail-SOT topology (replaces the v3 K-wave layout):
      - warmup[0] = parent = the user prompt as ONE block (no internal
        newline split — `<|user_start|>` … `<|user_end|>` stays whole).
      - assistant blocks (Q1-split on NL-bearing tokens): the LAST block
        becomes the row-last `tail` line so it decodes with full causal
        visibility — this subsumes the v3 `####`-isolated-answer rule,
        since the GSM8K/SpellingBee answer is the final block. Leading
        assistant blocks per `warmup_threshold` join `warmup` as
        sequential K=1 lines; the remainder become `middle` SOT-ramp lines.

    `kmax` is accepted for call-site compatibility but unused — v4 has no
    K-waves or `<bot_K>` openers.

    `warmup_threshold`: leading assistant blocks form a sequential warmup
    region until the cumulative (parent + asst) content first exceeds W,
    mirroring `multithread_dataloader.build_doc_v4`. The parent prompt
    counts toward the accumulator. For typical SFT data the prompt alone
    already exceeds W, so no assistant block enters warmup.

    Returns (DocV4, blocks_m_list). `blocks_m_list` is ordered to match
    `compile_doc_v4` thread numbering — [parent, warmup_asst…, middle…,
    tail] — so the generic `blocks_m_list[t][ls-1]` SFT-mask indexing
    aligns. The tail entry carries a leading dummy mask slot for the
    row-last `<sot_tail>` position (tail content local_step starts at 2:
    0=`<sot>`, 1=`<sot_tail>`, 2..=content)."""
    _ = kmax  # unused in v4
    parent_para = Paragraph(content=list(user_ids))
    parent_mask = list(user_mask)

    a_ids = list(asst_ids)
    a_mask = list(asst_mask)
    if a_ids and a_ids[-1] not in nl_ids:
        a_ids.append(10); a_mask.append(0)
    asst_blocks_t, asst_blocks_m = split_into_blocks_with_mask(a_ids, a_mask, nl_ids)

    # No assistant content: parent is the whole (and final) line.
    if not asst_blocks_t:
        return DocV4(warmup=[parent_para], middle=[], tail=None), [parent_mask]

    # Reserve the last assistant block as the row-last tail.
    prefix_t = asst_blocks_t[:-1]
    prefix_m = asst_blocks_m[:-1]
    tail_t = asst_blocks_t[-1]
    tail_m = asst_blocks_m[-1]

    # v3-style warmup: leading asst blocks become sequential warmup K=1
    # lines until cumulative (parent + asst) content first exceeds W.
    if warmup_threshold > 0:
        acc = len(parent_para.content)
        n_warmup_asst = 0
        if acc <= warmup_threshold:
            for i in range(len(prefix_t)):
                acc += len(prefix_t[i])
                n_warmup_asst = i + 1  # include the boundary block once acc > W
                if acc > warmup_threshold:
                    break
    else:
        n_warmup_asst = 0

    warmup_paras = [parent_para] + [
        Paragraph(content=b) for b in prefix_t[:n_warmup_asst]
    ]
    middle_paras = [Paragraph(content=b) for b in prefix_t[n_warmup_asst:]]
    tail_para = Paragraph(content=tail_t)

    blocks_m_list = [parent_mask]
    blocks_m_list += prefix_m[:n_warmup_asst]
    blocks_m_list += prefix_m[n_warmup_asst:]
    # Dummy slot for the row-last <sot_tail> position (ls=1, always masked).
    blocks_m_list.append([0] + tail_m)

    doc = DocV4(warmup=warmup_paras, middle=middle_paras, tail=tail_para)
    return doc, blocks_m_list


def compile_conv_turns(tokenizer, conv, nl_ids, special_ids,
                        sot_id, bos_id, rope_stride=256,
                        sot_tail_id=SOT_TAIL,
                        warmup_threshold=0,
                        kmax=None, bot_fn=None,
                        ) -> Optional[List[Tuple[RowMeta, DocV4, List[List[int]]]]]:
    """Render a conversation, split into turn-pairs, and compile each
    turn to its own RowMeta via `compile_doc_v4`. Returns a list of
    `(meta, doc, blocks_m_list)` per turn, in source order — or None
    if the conv has no usable content.

    `sot_tail_id`: the row-last `<sot_tail>` token id. Default = the synth
    `SOT_TAIL` constant for viz-only dumps; the production training loader
    passes the real `tokenizer.encode_special("<|sot_tail|>")`.

    `kmax` / `bot_fn` are accepted for v3 call-site compatibility but
    ignored — v4 has no K-waves or `<bot_K>` openers.

    `warmup_threshold`: forwarded to `build_turn_doc_v4`. See its docstring.
    When SFT-ing on top of a base checkpoint, pass the same threshold the
    base was trained with.
    """
    _ = (kmax, bot_fn)  # unused in v4
    ids, mask = tokenizer.render_conversation(conv)
    ids = list(ids); mask = list(mask)
    if not ids:
        return None
    if ids[0] == bos_id:
        ids = ids[1:]; mask = mask[1:]

    turns = find_turn_boundaries(ids, special_ids)
    if not turns:
        return None

    out = []
    for (us, ue, as_, ae) in turns:
        user_ids = ids[us:ue + 1]
        user_mask = mask[us:ue + 1]
        asst_ids = ids[as_:ae + 1]
        asst_mask = mask[as_:ae + 1]

        doc, blocks_m_list = build_turn_doc_v4(
            user_ids, user_mask, asst_ids, asst_mask,
            tokenizer, nl_ids,
            warmup_threshold=warmup_threshold,
        )
        if not doc.warmup:
            continue
        meta = compile_doc_v4(doc, rope_stride=rope_stride, sot_id=sot_id,
                              sot_tail_id=sot_tail_id, bos_id=bos_id)
        out.append((meta, doc, blocks_m_list))
    return out or None


def derive_forced_per_position(meta: RowMeta, doc: DocV4,
                                blocks_m_list: List[List[int]]) -> List[bool]:
    """Per-row-position `forced` flag carrying the SFT loss mask.
    `forced=True` ↔ this position's target is masked out in SFT.

    Structural openers (`<sot>`, `<sot_tail>`, `<bos>`) are always forced
    (no logit supervision). Content positions inherit their original
    mask: mask=1 → forced=False; mask=0 → forced=True.

    `blocks_m_list[gt]` is the per-block mask for the gt-th thread in
    `doc` (parallel to compile_doc_v4 thread numbering — warmup, middle,
    tail). The tail entry carries a leading dummy slot so `ls-1` indexing
    aligns despite the row-last `<sot_tail>` at local_step 1."""
    forced = [True] * meta.N
    for i in range(meta.N):
        t = meta.thread_idx[i]
        ls = meta.local_step[i]
        if t < 0:
            forced[i] = True
            continue
        if ls == 0:
            forced[i] = True
            continue
        forced[i] = (blocks_m_list[t][ls - 1] == 0)
    return forced


# ============================================================ production loader

def _compile_conv_to_pieces(tokenizer, conv, *, T, kmax, rope_stride,
                            nl_ids, special_ids, sot_id, bos_id,
                            sot_tail_id, eot_id, warmup_threshold):
    """Compile one conversation into a list of training pieces.

    Each piece is a dict `{tokens, targets, rope_idx, rope_max, N}` ready
    for best-fit packing. The SFT loss mask is applied to `targets`:
    at content positions (ls >= 1) whose `render_conversation` mask was
    0, the natural-shift target is overridden to -1. Opener positions
    (ls == 0: `<bos>`/`<sot>`) keep their natural target. The tail's
    `<sot_tail>` slot is masked via the dummy slot in `blocks_m_list`.

    `sot_tail_id` and `eot_id` MUST be real tokenizer ids — caller resolves
    them via `tokenizer.encode_special(...)` and passes through so the row
    layout and the c_L target overrides land on the same ids the model will
    see at inference. Synth defaults from `multithread_layout` are NOT used
    here (they would silently corrupt SFT supervision).

    Turn-pairs with `meta.N > T` are silently skipped — they can't fit
    in a single row no matter how the row is packed, and an MT row's
    interleaved layout doesn't survive truncation.
    """
    try:
        turns = compile_conv_turns(
            tokenizer, conv, nl_ids, special_ids,
            sot_id=sot_id, bos_id=bos_id,
            rope_stride=rope_stride, sot_tail_id=sot_tail_id,
            warmup_threshold=warmup_threshold,
        )
    except Exception:
        return []
    if not turns:
        return []

    pieces = []
    for (meta, doc, blocks_m_list) in turns:
        if meta.N > T:
            continue
        nat_targets = compute_targets_v4(
            meta, doc, bos_id=bos_id, eot_id=eot_id,
        )
        sft_targets = list(nat_targets)
        for i in range(meta.N):
            t = meta.thread_idx[i]
            ls = meta.local_step[i]
            if t < 0 or ls == 0:
                continue
            if blocks_m_list[t][ls - 1] == 0:
                sft_targets[i] = -1
        pieces.append({
            "tokens": list(meta.tokens),
            "targets": sft_targets,
            "rope_idx": list(meta.rope_idx),
            "rope_max": max(meta.rope_idx) if meta.rope_idx else 0,
            "N": meta.N,
        })
    return pieces


def tokenizing_distributed_data_loader_with_state_sft_mt(
    tokenizer, B, T, dataset,
    kmax=16,
    rope_stride=256,
    rotary_seq_len=None,
    device="cuda",
    resume_state_dict=None,
    buffer_size=200,
    warmup_threshold=0,
):
    """SFT-MT training/eval dataloader.

    Yields `(inputs, targets, rope_idx, state_dict)` per micro-batch.

    Args:
        tokenizer: a RustBPETokenizer with `render_conversation`,
            `encode_special`, `get_bos_token_id`, `get_vocab_size`,
            `decode`.
        B: per-device batch size (number of rows per yield).
        T: max sequence length per row.
        dataset: list-indexable conversation source (e.g. a
            `tasks.common.TaskMixture`). Each item is a chat-conversation
            dict with a "messages" key (see `tasks/`).
        kmax: accepted for config/call-site compatibility but unused in
            the v4 layout (which has no K-waves or `<bot_K>` openers).
        rope_stride: per-paragraph rope stride (default 256, matches the
            pretrain MT loader). Per-piece rope_offset is rounded up to
            a multiple of this so rope-bands don't collide cross-piece.
        rotary_seq_len: optional safety bound; batches whose max
            `rope_idx` would exceed the model's RoPE cache are skipped
            with a warning rather than asserted in the GPU gather.
        device: target device for the yielded tensors.
        resume_state_dict: dict with `{conv_idx, epoch, consumed}` to
            restore from a checkpoint; None to start fresh at this rank.
        buffer_size: number of compiled turn-pieces to keep in the
            best-fit packing buffer. Larger = better packing efficiency,
            slightly more upfront compile work.
        warmup_threshold: v3 bot-1 warmup (see `build_turn_doc` docstring). 0 = v2 layout.
            When SFT-ing on top of a v3 base checkpoint, pass the same
            `mt_warmup_threshold` the base was trained with (read from
            base meta in `chat_sft_mt.py`).

    Padded slots (when no piece fits the remaining capacity) carry:
        inputs = `<bos>`, targets = -1, rope_idx = `rope_cursor + (k-pos)`
    so RoPE never sees a duplicate index inside a row. With target=-1,
    pad positions contribute no loss; causal attention can't leak info
    from them back into real content.

    `state_dict` semantics (mirrors `chat_sft.py`'s globals):
        conv_idx — next conv index this rank will fetch (rank-local cursor).
        epoch    — 1-indexed epoch counter; increments when conv_idx wraps.
        consumed — per-rank view of "total conversations consumed across
            all ranks"; increments by world_size per pop. Outer training
            loop uses `consumed >= len(dataset)` as the end-of-epoch trigger.
    """
    assert 1 <= kmax <= 16, f"kmax must be in 1..16, got {kmax}"
    assert warmup_threshold >= 0, f"warmup_threshold must be >= 0, got {warmup_threshold}"

    ddp, rank, local_rank, world_size = get_dist_info()

    # Resolve every MT-layout special from the tokenizer up front. If any
    # is missing, fail loud — silently falling back to the synth defaults
    # from `multithread_layout` would corrupt SFT supervision (the model
    # would learn to predict synth ids 2/3/11.. instead of the real
    # `<|sot|>`/`<|eot|>`/`<|bot_K|>`).
    def _require_special(name):
        tid = tokenizer.encode_special(name)
        assert tid is not None, (
            f"SFT-MT requires {name!r} in the tokenizer vocab; "
            f"{name!r} is missing. Use an MT tokenizer "
            f"(e.g. nanochat_64k_mt2)."
        )
        return tid

    sot_id = _require_special("<|sot|>")
    bos_id = tokenizer.get_bos_token_id()
    eot_id = _require_special("<|eot|>")
    sot_tail_id = _require_special("<|sot_tail|>")
    nl_ids = compute_nl_token_ids(tokenizer)
    special_ids = {
        "user_start":      tokenizer.encode_special("<|user_start|>"),
        "user_end":        tokenizer.encode_special("<|user_end|>"),
        "assistant_start": tokenizer.encode_special("<|assistant_start|>"),
        "assistant_end":   tokenizer.encode_special("<|assistant_end|>"),
        "bos":             bos_id,
    }

    if resume_state_dict is not None:
        conv_idx = int(resume_state_dict.get("conv_idx", rank))
        epoch    = int(resume_state_dict.get("epoch", 1))
        consumed = int(resume_state_dict.get("consumed", rank))
    else:
        conv_idx = rank
        epoch    = 1
        consumed = rank

    dataset_size = len(dataset)
    assert dataset_size > 0, "dataset is empty"

    pieces_buffer: List[dict] = []

    def refill_buffer():
        nonlocal conv_idx, epoch, consumed
        while len(pieces_buffer) < buffer_size:
            conv = dataset[conv_idx % dataset_size]
            pieces_buffer.extend(
                _compile_conv_to_pieces(
                    tokenizer, conv,
                    T=T, kmax=kmax, rope_stride=rope_stride,
                    nl_ids=nl_ids, special_ids=special_ids,
                    sot_id=sot_id, bos_id=bos_id,
                    sot_tail_id=sot_tail_id, eot_id=eot_id,
                    warmup_threshold=warmup_threshold,
                )
            )
            conv_idx += world_size
            consumed += world_size
            if conv_idx >= dataset_size:
                conv_idx = conv_idx % dataset_size
                epoch += 1

    use_cuda = device == "cuda"
    NUM_FIELDS = 3
    cpu_buffer = torch.empty(NUM_FIELDS * B * T, dtype=torch.long,
                             pin_memory=use_cuda)
    gpu_buffer = torch.empty(NUM_FIELDS * B * T, dtype=torch.long, device=device)
    cpu_inputs   = cpu_buffer[0 * B * T : 1 * B * T].view(B, T)
    cpu_targets  = cpu_buffer[1 * B * T : 2 * B * T].view(B, T)
    cpu_rope_idx = cpu_buffer[2 * B * T : 3 * B * T].view(B, T)
    inputs    = gpu_buffer[0 * B * T : 1 * B * T].view(B, T)
    targets   = gpu_buffer[1 * B * T : 2 * B * T].view(B, T)
    rope_idx  = gpu_buffer[2 * B * T : 3 * B * T].view(B, T)

    while True:
        cpu_inputs.fill_(bos_id)
        cpu_targets.fill_(-1)
        cpu_rope_idx.zero_()

        for row_idx in range(B):
            pos = 0
            rope_cursor = 0
            while pos < T:
                while len(pieces_buffer) < buffer_size:
                    refill_buffer()

                remaining = T - pos
                best_idx = -1
                best_len = 0
                for i, item in enumerate(pieces_buffer):
                    n = item["N"]
                    if n <= remaining and n > best_len:
                        best_idx = i
                        best_len = n

                if best_idx < 0:
                    for k in range(pos, T):
                        cpu_rope_idx[row_idx, k] = rope_cursor + (k - pos)
                    break

                item = pieces_buffer.pop(best_idx)
                N = item["N"]
                cpu_inputs[row_idx, pos:pos + N] = torch.tensor(
                    item["tokens"], dtype=torch.long)
                cpu_targets[row_idx, pos:pos + N] = torch.tensor(
                    item["targets"], dtype=torch.long)
                cpu_rope_idx[row_idx, pos:pos + N] = (
                    torch.tensor(item["rope_idx"], dtype=torch.long) + rope_cursor
                )

                doc_max_abs_rope = rope_cursor + item["rope_max"]
                rope_cursor = ((doc_max_abs_rope // rope_stride) + 1) * rope_stride
                pos += N

        state_dict = {
            "conv_idx": conv_idx,
            "epoch":    epoch,
            "consumed": consumed,
        }

        if rotary_seq_len is not None:
            max_rope = int(cpu_rope_idx.max().item())
            if max_rope >= rotary_seq_len:
                warnings.warn(
                    f"rope_idx max {max_rope} >= rotary_seq_len {rotary_seq_len}; "
                    f"skipping batch (would have asserted in RoPE gather).",
                    RuntimeWarning, stacklevel=2,
                )
                continue

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, rope_idx, state_dict


def tokenizing_distributed_data_loader_sft_mt(*args, **kwargs):
    """Convenience wrapper that yields `(inputs, targets, rope_idx)`
    without the state dict — for eval loops that don't need resume."""
    for inputs, targets, rope_idx, _state in (
        tokenizing_distributed_data_loader_with_state_sft_mt(*args, **kwargs)
    ):
        yield inputs, targets, rope_idx
