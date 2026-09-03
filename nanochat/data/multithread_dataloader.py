"""
Multithread-LM dataloader (run-preservation design — see
`design/blockmt/blockmt-newline-preserve-runs.md`).

Real-corpus path for the multithread row layout. Reuses `_document_batches`
from `nanochat.data.dataloader` for parquet streaming + DDP sharding +
resume, then adds multithread-specific block splitting, doc construction,
row compilation, target overrides, and rope-offset baking.

Attention is pure causal — the model dispatches to FA3 with
`causal=True`. Sibling tokens within a wave-step see each other only
via row-order causal (ordered, not mutual). Cross-doc handling reduces
to vanilla BOS-anchored causal, same as
`tokenizing_distributed_data_loader_bos_bestfit`. No mask construction
in the loader.

Block walking (Q1 rule): encode each doc's text whole, then walk the
token stream with a 3-state classifier so consecutive PURE_NL tokens
(bare `\n`/`\r`) are grouped into one block's trailing run rather than
each closing a block. TRAIL_NL tokens (merges like `.\n`, `?\n`,
` \n`) close the prior block IF a pending run was open (the token's
leading non-newline byte breaks the character-level run), then open a
new block with this token. NO_NL tokens close the prior block (if
pending) and accumulate as content.

Doc-start strip (Q1 follow-up): leading `\n`/`\r` chars are stripped
from raw doc text before encoding so the parent paragraph (stage 0)
is never an all-newline degenerate.

No doc-end fallback (Q7 relaxation): docs may end mid-content. The
last block of a doc can have a NO_NL c_L; the target-override at c_L
fires regardless of c_L's token class.

For each source doc:
  1. lstrip leading `\n`/`\r` from doc text.
  2. Encode the whole doc into a token stream.
  3. Walk tokens with the 3-state classifier → list of blocks.
  4. Drop empty docs.
  5. Group blocks into a multithread `DocV4` via `build_doc_v4`:
     sequential warmup prefix, one SOT-ramp middle, and a row-last tail.
  6. Compile to a row layout via `compile_doc_v4` and `compute_targets_v4`,
     resolving real tokenizer ids for `<bos>`, `<sot>`, `<sot_tail>`,
     `<eot>`.

Rows are packed BOS-bestfit-style: largest doc that fits is placed; if
nothing fits, the remainder is zero-padded with target=-1
(cross_entropy ignore_index). Cropping is intentionally NOT done — a
multithread row's interleaved layout doesn't survive truncation.

Per-doc rope offsets are baked into `rope_idx` at pack time so cross-doc
rope values don't collide. The offset is rounded up to the next
`rope_stride` boundary so paragraph anchors stay at stride multiples
globally across the row (not just within each doc).

Per-yield outputs (each shape `(B, T)` on `device` unless noted):
    inputs[s]    token at seq pos s; content tokens + `<sot>` + `<bos>`
                 + `<bot-K>`. No `<nl>` or `<eot>` row slots — `<eot>`
                 appears only as a target override (Q3).
    targets[s]   within-thread shift, with c_L overrides:
                   * decision c_L → `<bos>` / `<bot-K>`
                   * non-decision c_L (K≥2 only) → `<eot>`
                 -1 at pad / structural-opener slots.
    rope_idx[s]  1D RoPE index. Per-thread staggered within a doc;
                 per-doc offset baked at pack time.
    state_dict   resume cursor: {pq_idx, rg_idx, epoch}.
"""

import warnings

import torch

from nanochat.data.dataloader import _document_batches
from nanochat.data.multithread_layout import (
    DocV4, Paragraph,
    compile_doc_v4, compute_targets_v4,
)


# Token-class labels for the Q1 walk rule. Defined as module constants
# so the dict-valued classifier interns to fixed string identities
# (cheap equality checks in the per-token hot loop).
NL_CLASS_PURE = "PURE_NL"   # decoded bytes are entirely `\n`/`\r`
NL_CLASS_TRAIL = "TRAIL_NL"  # bytes end with `\n`/`\r`, start non-newline
NL_CLASS_NO = "NO_NL"        # no `\n`/`\r` in decoded bytes


def compute_nl_token_ids(tokenizer):
    """Return the set of token ids whose decoded bytes contain `\\n`
    or `\\r`.

    Under the run-preservation design, both `\\n` and `\\r` chars are
    members of the same newline-bearing class (Q5). Any token whose
    decoded bytes contain either char is "newline-bearing" and may
    appear in a block's trailing run. Precomputed once at dataloader /
    engine init by scanning the full vocab.
    """
    nl_ids = set()
    vocab_size = tokenizer.get_vocab_size()
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        if "\n" in s or "\r" in s:
            nl_ids.add(tid)
    return nl_ids


def classify_nl_tokens(tokenizer):
    """Return a dict mapping every token id to its newline class
    (PURE_NL / TRAIL_NL / NO_NL). The Q1 walk rule reads from this
    dict on the hot per-token loop.

    Classes:
      - PURE_NL: decoded bytes are entirely `\\n`/`\\r` (e.g. id 10,
        id 13 in nanochat_64k_mt2). Extends a pending run.
      - TRAIL_NL: bytes end with `\\n`/`\\r` and start with non-
        newline (e.g. `.\\n` id 307, `?\\n` id 791, ` \\n` id 12204).
        Closes any pending prior run and opens a new block with this
        token.
      - NO_NL: bytes contain no `\\n`/`\\r`. Closes any pending prior
        run and accumulates as content.

    LEAD_NL (bytes start with newline but end with non-newline) and
    MID_NL (newline strictly in the middle) are NOT allowed under the
    Q1 walk rule. `assert_walker_invariants` checks the vocab has
    neither.
    """
    nl_class = {}
    vocab_size = tokenizer.get_vocab_size()
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            nl_class[tid] = NL_CLASS_NO
            continue
        has_n = ("\n" in s) or ("\r" in s)
        if not has_n:
            nl_class[tid] = NL_CLASS_NO
        elif all(c in "\n\r" for c in s):
            nl_class[tid] = NL_CLASS_PURE
        elif s[-1] in "\n\r" and s[0] not in "\n\r":
            nl_class[tid] = NL_CLASS_TRAIL
        else:
            # LEAD_NL or MID_NL — Q1 walker rule doesn't handle these.
            # Caller should `assert_walker_invariants` to fail loud.
            nl_class[tid] = "INVALID"
    return nl_class


def assert_walker_invariants(tokenizer):
    """Assert the tokenizer's vocab is compatible with the Q1 walker
    rule (Q10). Specifically: no LEAD_NL or MID_NL tokens (any token
    whose decoded bytes have `\\n`/`\\r` not exclusively at the end).

    Tripped only if a future tokenizer extension introduces such a
    merge; `nanochat_64k_mt`/`nanochat_64k_mt2` both satisfy this
    today.
    """
    nl_class = classify_nl_tokens(tokenizer)
    bad = [tid for tid, c in nl_class.items() if c == "INVALID"]
    if bad:
        samples = [(tid, repr(tokenizer.decode([tid]))) for tid in bad[:5]]
        raise AssertionError(
            f"tokenizer has {len(bad)} LEAD_NL/MID_NL tokens (newline at "
            f"start or middle of decoded bytes); the Q1 walk rule assumes "
            f"every newline-bearing token ends in `\\n`/`\\r`. "
            f"First {min(5, len(bad))}: {samples}"
        )
    return nl_class


def split_tokens_into_blocks(tokens, nl_class):
    """Walk a token stream and split into blocks using the Q1 3-state
    walk rule.

    Args:
        tokens: iterable of token ids.
        nl_class: dict mapping token_id -> "PURE_NL" | "TRAIL_NL" | "NO_NL"
            (built once via `classify_nl_tokens`).

    The walker maintains a `pending_run` flag that's set after any
    newline-bearing token. State transitions:
      - PURE_NL: append to current block, set pending=True.
      - TRAIL_NL: if pending, close current block and open new with
        this token (its leading non-newline byte breaks the prior
        character-level run); else just append. Either way, set
        pending=True (the token ends in newline).
      - NO_NL: if pending, close current block, start new; either way
        append, clear pending.

    The trailing `current` block at end-of-stream is emitted even if
    its c_L is NO_NL (Q7: last block of a doc may have a non-newline-
    bearing tail).
    """
    blocks = []
    current = []
    pending = False
    for tok in tokens:
        cls = nl_class.get(tok, NL_CLASS_NO)
        if cls == NL_CLASS_PURE:
            current.append(tok)
            pending = True
        elif cls == NL_CLASS_TRAIL:
            if pending:
                # Prior run breaks at this token's leading non-newline
                # byte. Close the prior block, open a new one.
                blocks.append(current)
                current = [tok]
            else:
                current.append(tok)
            pending = True
        else:  # NO_NL (or unrecognized — assert_walker_invariants would
               # have tripped if INVALID).
            if pending:
                blocks.append(current)
                current = []
                pending = False
            current.append(tok)
    if current:
        blocks.append(current)
    return blocks


def build_doc_v4(block_token_lists, warmup_threshold=16):
    """Build the v4 topology: warmup prefix, one middle SOT ramp, final tail.

    The final source block is reserved as the sequential row-last tail when a
    document has at least two blocks. The prefix before that tail gets the v3
    warmup rule: include leading blocks until cumulative content tokens first
    exceed ``warmup_threshold``. Remaining prefix blocks become ordinary
    middle lines in one uncapped SOT ramp.
    """
    assert warmup_threshold >= 0, f"warmup_threshold must be >= 0, got {warmup_threshold}"
    paras = [Paragraph(content=toks) for toks in block_token_lists]
    n = len(paras)
    if n < 1:
        return None
    if n == 1:
        return DocV4(warmup=[paras[0]], middle=[], tail=None)

    prefix = paras[:-1]
    tail = paras[-1]

    if warmup_threshold > 0:
        acc = 0
        warmup_end = len(prefix) - 1
        for i, p in enumerate(prefix):
            acc += p.length
            if acc > warmup_threshold:
                warmup_end = i
                break
        n_warmup = warmup_end + 1
    else:
        n_warmup = 1

    warmup = prefix[:n_warmup]
    middle = prefix[n_warmup:]
    return DocV4(warmup=warmup, middle=middle, tail=tail)



def tokenizing_distributed_data_loader_with_state_multithread(
    tokenizer, B, T, split,
    kmax=16,
    rope_stride=256,
    rotary_seq_len=None,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
    warmup_threshold=0,
    layout_version="v4",
    v4_parallel_cap=None,
):
    """Multithread-LM dataloader (Interp 2, pure-causal).

    Yields `(inputs, targets, rope_idx, state_dict)`. See module
    docstring for tensor semantics. Each packed doc starts with `<bos>`,
    so vanilla causal `k <= q` across packed docs is exactly the
    BOS-anchored stance from
    `tokenizing_distributed_data_loader_bos_bestfit`.

    `warmup_threshold`: when > 0, each doc's leading blocks form the v4
    sequential warmup region until cumulative content first exceeds the
    threshold; the rest become SOT-ramp middle lines. See `build_doc_v4`.

    `v4_parallel_cap`: None keeps the original uncapped v4 SOT ramp;
    positive values cap simultaneously active ordinary middle lines.
    """
    assert split in ("train", "val")
    assert layout_version == "v4", (
        f"only layout_version='v4' is supported (v2/v3 were removed); "
        f"got {layout_version!r}"
    )
    if v4_parallel_cap == 0:
        v4_parallel_cap = None
    if v4_parallel_cap is not None:
        assert v4_parallel_cap >= 1, (
            f"v4_parallel_cap must be >= 1 or None, got {v4_parallel_cap}"
        )
    assert warmup_threshold >= 0, f"warmup_threshold must be >= 0, got {warmup_threshold}"
    sot_id = tokenizer.encode_special("<|sot|>")
    bos_id = tokenizer.get_bos_token_id()
    eot_id = tokenizer.encode_special("<|eot|>")
    sot_tail_id = tokenizer.encode_special("<|sot_tail|>")

    # Q1 walk rule: classify every token id once as PURE_NL / TRAIL_NL /
    # NO_NL. Also fail loud if the vocab has any LEAD_NL/MID_NL merge,
    # since the walker assumes newlines only ever appear at the tail of
    # a token (Q10).
    nl_class = assert_walker_invariants(tokenizer)
    nl_token_ids = {tid for tid, c in nl_class.items()
                    if c in (NL_CLASS_PURE, NL_CLASS_TRAIL)}
    assert nl_token_ids, "tokenizer has no newline-bearing tokens"
    # Sanity: bare `\n` must be a single PURE_NL token.
    bare_nl_ids = tokenizer.encode("\n")
    assert (len(bare_nl_ids) == 1
            and nl_class[bare_nl_ids[0]] == NL_CLASS_PURE), (
        f"tokenizer.encode('\\n') = {bare_nl_ids}; must be a single "
        f"PURE_NL token"
    )

    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    pq_idx, rg_idx, epoch = 0, 0, 1

    doc_buffer = []

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        prepped_texts = []
        for doc_text in doc_batch:
            if not doc_text or not doc_text.strip():
                continue
            # Q1 follow-up: strip leading `\n`/`\r` so the parent
            # paragraph (first block emitted by the walker) is never an
            # all-newline degenerate.
            doc_text = doc_text.lstrip("\n\r")
            if not doc_text:
                continue
            # No N1 fallback (Q7): doc text may end without a newline.
            # The walker emits the trailing `current` block as the last
            # block regardless of c_L's class.
            prepped_texts.append(doc_text)
        if not prepped_texts:
            return
        # Batched encode of whole doc texts.
        doc_token_lists = tokenizer.encode(
            prepped_texts, num_threads=tokenizer_threads
        )
        for doc_tokens in doc_token_lists:
            blocks = split_tokens_into_blocks(doc_tokens, nl_class)
            if not blocks:
                continue
            doc = build_doc_v4(blocks, warmup_threshold=warmup_threshold)
            if doc is None:
                continue
            meta = compile_doc_v4(
                doc, rope_stride=rope_stride,
                sot_id=sot_id, sot_tail_id=sot_tail_id, bos_id=bos_id,
                nl_token_ids=nl_token_ids,
                v4_parallel_cap=v4_parallel_cap,
            )
            if meta.N > T:
                continue  # multithread rows can't be cropped without breaking layout
            targets = compute_targets_v4(
                meta, doc, bos_id=bos_id, eot_id=eot_id,
            )
            doc_buffer.append({
                "tokens": meta.tokens,
                "targets": targets,
                "rope_idx": meta.rope_idx,
                "rope_max": max(meta.rope_idx),
                "N": meta.N,
            })

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
        cpu_inputs.zero_()
        cpu_targets.fill_(-1)
        cpu_rope_idx.zero_()
        for row_idx in range(B):
            pos = 0
            rope_cursor = 0  # stride-aligned absolute rope offset for the next doc
            while pos < T:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = T - pos
                # BOS-bestfit: largest doc that fits entirely.
                best_idx = -1
                best_len = 0
                for i, item in enumerate(doc_buffer):
                    n = item["N"]
                    if n <= remaining and n > best_len:
                        best_idx = i
                        best_len = n

                if best_idx < 0:
                    break  # nothing fits — leave remainder zero-padded.

                item = doc_buffer.pop(best_idx)
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

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Defensive guard: if any rope_idx would overflow the model's RoPE
        # cache (size = rotary_seq_len), skip this batch with a warning
        # rather than handing an out-of-bounds index to self.cos[0, rope_idx]
        # and crashing the GPU with a device-side assertion. Rare under the
        # default factor; activates only on pathologically dense packings.
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


def tokenizing_distributed_data_loader_multithread(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, rope_idx, _state in (
        tokenizing_distributed_data_loader_with_state_multithread(*args, **kwargs)
    ):
        yield inputs, targets, rope_idx
