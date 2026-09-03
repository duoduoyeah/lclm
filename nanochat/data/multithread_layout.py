"""
Multithread-LM row layout + target rules.

Production-side home for the data structures and pure-python helpers that
both the real dataloader (`nanochat/data/multithread_dataloader.py`) and the
preview / dump scripts under `attn_preview/` and `scripts/dump/` consume.

A doc (`DocV4`) is the v4 tail-SOT topology: a sequential `warmup` prefix,
one `middle` SOT-ramp of ordinary lines, and a row-last `tail` line. It is
flattened to a row via `compile_doc_v4`, with per-position targets from
`compute_targets_v4`.

NOTE: the v3 wave/`<bot-K>`/decision-thread model (and the `Doc`/`Stage`
types, `compile_doc`, `compute_targets`) was removed in the v4-only cleanup.
Some of the special-token-role prose below still references that retired
model for historical context.

Block tail — **run-preservation design** (see
`design/blockmt/blockmt-newline-preserve-runs.md`; supersedes the pre-2026-05-14
Interp 2 design). The dataloader's `split_tokens_into_blocks` walks the
token stream with a 3-state classifier (PURE_NL / TRAIL_NL / NO_NL) so
that maximal `\\n`/`\\r` runs are preserved verbatim as a block's
trailing tokens — `\\n\\n` is one boundary with a 2-token trailing run,
not two boundaries. The block's LAST content token (`c_L`) carries the
block-end role; no separate `<nl>` slot, and `<eot>` is target-only
(never a row slot).

  - **Non-decision** blocks end with `c_L` only (1-slot tail). The
    target at `c_L` is overridden to `<eot>` (engine retires the thread
    on sampling `<eot>`; never feeds it back). The natural-shift target
    that would have been -1 is replaced by `<eot>`. Only fires in K≥2
    waves — K=1 waves are decision-only.
  - **Decision** blocks end with `c_L` only (1-slot tail). The control
    signal — `<bot-K'>` (next wave's K) or `<bos>` (final stage) — is
    predicted at the `c_L` slot's logit (override; replaces the natural
    target=-1). The engine reads the control signal AT the c_L slot's
    logit and transitions directly to the next wave's `<bot-K>` opener
    (or TERMINATED).
  - Tail is uniform — both kinds end with just `c_L`; only the override
    target differs. `<eot>` does NOT consume a row slot.
  - **Q7 relaxation**: `c_L` may be NO_NL for the last block of a doc
    when the source text doesn't end with a newline-bearing char. The
    override at `c_L` still fires regardless of `c_L`'s token class.

Special-token roles — by mask class (target = -1 vs. target ≠ -1):

  Target-masked (`thread_idx = -1` in `compile_doc` → `compute_targets`
  leaves target at -1, fully excluded from CE loss):
  - `<bot-K>`: input-side structural opener emitted between stages.
    Also predicted at the prior decision thread's `c_L` (target
    override on the *previous* slot signaling "next wave has K
    threads"), so `<bot-K>` is dual-role: target = -1 where it sits
    in the input stream, but it IS a supervised target one slot
    earlier. The downstream `<sot>` sees it via row-order causal and
    recovers the wave size K.

  First-slot-of-a-thread (`thread_idx ≥ 0` → natural within-thread
  shift target → NOT mask-excluded, but BPB excludes specials via
  `token_bytes[id] = 0`):
  - `<bos>`: thread 0's anchor slot of stage 0 — emitted at the start
    of EVERY packed doc in the row (not only at row[0]; the dataloader
    packs multiple docs per row, each pre-emits its own `<bos>` via
    `compile_doc`). Target = the doc's first real content token, just
    like vanilla LM `BOS → first_token`.
  - `<sot>`: each non-parent thread's anchor slot in K≥2 waves
    (`local_step = 0`). Target = that thread's first content token.
    Never used as a target override anywhere — `<sot>` itself is
    never in the target stream (no position points back to a
    `<sot>` slot via `next_in_thread`).

  Target-only (appears in targets but never in inputs):
  - `<eot>`: target override at non-decision threads' `c_L` slots in
    K≥2 waves. Counted in CE loss; excluded from BPB byte denominator
    via `token_bytes[<eot>] = 0`, so it cannot move the BPB number.
  - `<bos>` (as override target): also used at the final stage's
    decision `c_L` to signal end-of-row. Same BPB exclusion applies.

  Absent from the layout entirely:
  - **No `<nl>` token** — the natural newline-bearing BPE merge fires
    inside `c_L`.

Practical consequence for eval/verify code: the only INPUT-side token
class with `target == -1` is `<bot-K>`. `<sot>` and `<bos>` are NOT
target-masked, even though they read like "openers" — they're the
first content slots of real threads.

The inference engine reads `<bot-K>` and `<bos>` predictions as control
signals (spawn / terminate) at the decision thread's `c_L` slot. For
`<bot-K>`, the engine ALSO feeds the predicted token back as the next
input (matching the row layout below).

Parent paragraph is a degenerate K=1 wave: stage 0 has a single thread
which is also that wave's decision thread, so its last content token
predicts `<bos>` or `<bot-K>` directly.

RoPE — uniform paragraph alignment. Every paragraph's anchor sits at
`paragraph_idx * stride` where `paragraph_idx` is a global counter across
the doc. Anchor = `<bos>` for parent, `<sot>` for stage-1+ threads. Content
starts at `anchor + 1`. Anchor-to-anchor rope distance is uniformly
`stride` (default 256). This gives the model a single uniform rule —
"paragraph boundary ↔ rope multiple of stride" — so thread identity is
encoded by rope band index alone.

    paragraph 0 (parent)        : <bos> at rope 0;        content at 1..L_parent
    paragraph 1 (stage-1 t0)    : <sot> at rope 1*stride; content at +1..
    paragraph 2 (stage-1 t1)    : <sot> at rope 2*stride; content at +1..
    paragraph 3 (stage-1 t2)    : <sot> at rope 3*stride; content at +1..
    paragraph 4 (stage-2 t0)    : <sot> at rope 4*stride; ...
    ...

Constraint (Interp 2): `stride > max_block_length + 1` (so blocks +
the next stage's `<bot-K>` opener don't overflow the stride window).
With `stride=256` and the D6 splitter's cap, max block length must be
<= 254. Caller's responsibility.

Layout — decision-thread-last is held to end-of-wave. The decision
thread's *last content token* (`c_L`, a newline-bearing token that
predicts `<bos>` / `<bot-K'>` at its logit) is held until every other
thread in the wave has emitted its content, then placed at the end of
the wave's row segment. Under row-order causal attention this is what
gives the decision token full sibling context (every sibling token
precedes it in the row, so causal sees them all). Train-time row order
= inference-time feed order, matching the postpone rule structurally.

For K=1 waves (parent paragraph), holding decision-last is a no-op since
there are no siblings — the held token emits at its natural row position.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import random


# Special token ids used by the synth/preview path. The real tokenizer
# allocates separate ids; production code translates between them via
# the tokenizer's `encode_special` calls (and resolves the set of
# newline-bearing token ids by scanning the vocab).
#
# Under the run-preservation design (2026-05-14), `<eot>` is used as a
# target-only override at non-decision c_L slots (not a row slot). The
# synth `EOT` constant below stands in for `<|eot|>` in compute_targets
# defaults. `NL` is legacy-only — no row slot uses it; the dataloader's
# Q1 walk rule classifies real newline-bearing tokens directly from
# decoded bytes.
BOS = 1
SOT = 2
EOT = 3
NL  = 4
SOT_TAIL = 5
BOT_BASE = 10        # BOT_K = BOT_BASE + K, so BOT_1 = 11, BOT_2 = 12, ...

SPECIAL_IDS = {BOS, SOT, EOT, NL, SOT_TAIL} | {BOT_BASE + k for k in range(1, 17)}


def bot_id(K: int) -> int:
    assert 1 <= K <= 16, f"only K in 1..16 supported, got {K}"
    return BOT_BASE + K


def token_name(tok: int) -> str:
    """Human-readable name for the visualizer / text dump."""
    if tok == BOS:
        return "<bos>"
    if tok == SOT:
        return "<sot>"
    if tok == EOT:
        return "<eot>"
    if tok == NL:
        return "<nl>"
    if tok == SOT_TAIL:
        return "<sot-tail>"
    if BOT_BASE + 1 <= tok <= BOT_BASE + 16:
        return f"<bot-{tok - BOT_BASE}>"
    return f"c{tok}"


def is_special(tok: int) -> bool:
    return tok in SPECIAL_IDS


@dataclass
class Paragraph:
    """A list of content token ids representing one block.

    Under the run-preservation design (2026-05-14), the LAST content
    token (`c_L`) is typically newline-bearing — either a PURE_NL token
    (`\\n` id 10, `\\r` id 13) or a TRAIL_NL merge (`.\\n` id 307,
    `?\\n` id 791, etc.) — produced by the dataloader's Q1 walk rule
    grouping runs of newline-bearing tokens as a block's trailing tail.
    Under the Q7 relaxation, `c_L` may also be a NO_NL token when this
    is the last block of a source doc that didn't end in a newline-
    bearing char. The target-override at `c_L` (`<eot>` for non-decision,
    `<bot-K>`/`<bos>` for decision) fires regardless of `c_L`'s class."""
    content: List[int]

    @property
    def length(self) -> int:
        return len(self.content)


@dataclass
class DocV4:
    """Block-MT v4 topology: sequential warmup, one SOT ramp, row-last tail."""
    warmup: List[Paragraph]
    middle: List[Paragraph]
    tail: Optional[Paragraph] = None

    def __post_init__(self):
        assert self.warmup, "v4 doc must have at least one warmup/parent block"


@dataclass
class RowMeta:
    """Compiled row layout. All lists have length N (= the row width)."""
    tokens: List[int]
    thread_idx: List[int]    # global paragraph idx (0..total_paragraphs-1)
    stage_idx: List[int]
    local_step: List[int]    # within-thread step
    rope_idx: List[int]

    @property
    def N(self) -> int:
        return len(self.tokens)


def compile_doc_v4(doc: DocV4, rope_stride: int = 256,
                   sot_id: int = SOT, sot_tail_id: int = SOT_TAIL,
                   bos_id: int = None, nl_token_ids=None,
                   v4_parallel_cap: Optional[int] = None) -> RowMeta:
    """Flatten a v4 doc into row order.

    v4 removes input-side <bot_K>. Ordinary <sot> anchors are layout
    inputs: they target the first content token for ordinary lines. The
    tail candidate is also an earlier <sot> anchor, but its same-thread
    next token is row-last <sot_tail>, then the final tail content.

    ``v4_parallel_cap`` optionally caps the number of simultaneously active
    ordinary middle lines. ``None`` preserves the original uncapped v4
    schedule. Positive values apply backpressure to new SOT injection while
    keeping the same target rules and final row-last tail.
    """
    _ = nl_token_ids
    if v4_parallel_cap is not None:
        assert v4_parallel_cap >= 1, (
            f"v4_parallel_cap must be >= 1 or None, got {v4_parallel_cap}"
        )
    for group_name, group in (("warmup", doc.warmup), ("middle", doc.middle)):
        for i, para in enumerate(group):
            assert para.content, f"empty v4 {group_name} block {i}"
    if doc.tail is not None:
        assert doc.tail.content, "empty v4 tail block"

    tokens, thread_idx, stage_idx, local_step, rope_idx = [], [], [], [], []

    def emit(tok, t, s, ls, rope):
        tokens.append(tok)
        thread_idx.append(t)
        stage_idx.append(s)
        local_step.append(ls)
        rope_idx.append(rope)

    thread = 0
    stage = 0

    # Parent block: BOS occupies the anchor slot. Parent content still starts
    # at local_step/rope +1 even if bos_id is omitted for previews.
    if bos_id is not None:
        emit(bos_id, thread, stage, 0, 0)
    for ls, tok in enumerate(doc.warmup[0].content, start=1):
        emit(tok, thread, stage, ls, ls)
    thread += 1
    stage += 1

    # Remaining warmup blocks are sequential K=1 lines opened by engine SOTs.
    for para in doc.warmup[1:]:
        base = thread * rope_stride
        emit(sot_id, thread, stage, 0, base)
        for ls, tok in enumerate(para.content, start=1):
            emit(tok, thread, stage, ls, base + ls)
        thread += 1
        stage += 1

    # Middle blocks run as an SOT-ramp waterfall. There is no decision thread
    # and no <bot_K> opener; every ordinary middle line ends by targeting EOT.
    # Uncapped v4 starts a new ordinary line every row step. Capped v4 emits
    # active content first, then starts at most one new line only if the active
    # ordinary count at the start of that row step is below the cap.
    middle_base_thread = thread
    middle_stage = stage
    K = len(doc.middle)
    if K:
        lengths = [p.length for p in doc.middle]
        rope_base_per_thread = [
            (middle_base_thread + i) * rope_stride for i in range(K)
        ]
        tail_thread = middle_base_thread + K if doc.tail is not None else None
        tail_stage = middle_stage + 1 if doc.tail is not None else middle_stage
        if v4_parallel_cap is None:
            max_w = max(t + lengths[t] for t in range(K))
            for w in range(max_w + 1):
                for t in range(K):
                    if w < t:
                        continue
                    ls = w - t
                    if ls > lengths[t]:
                        continue
                    gt = middle_base_thread + t
                    if ls == 0:
                        emit(sot_id, gt, middle_stage, 0,
                             rope_base_per_thread[t])
                    else:
                        emit(doc.middle[t].content[ls - 1], gt, middle_stage,
                             ls, rope_base_per_thread[t] + ls)
                if tail_thread is not None and w == K:
                    emit(sot_id, tail_thread, tail_stage, 0,
                         tail_thread * rope_stride)
        else:
            next_middle = 0
            active: List[int] = []
            next_ls = {}
            tail_candidate_emitted = False
            while (next_middle < K or active
                   or (tail_thread is not None and not tail_candidate_emitted)):
                active_at_step_start = list(active)
                finished = set()
                for t in active_at_step_start:
                    ls = next_ls[t]
                    gt = middle_base_thread + t
                    emit(doc.middle[t].content[ls - 1], gt, middle_stage,
                         ls, rope_base_per_thread[t] + ls)
                    if ls == lengths[t]:
                        finished.add(t)
                    else:
                        next_ls[t] = ls + 1

                if len(active_at_step_start) < v4_parallel_cap:
                    if next_middle < K:
                        t = next_middle
                        gt = middle_base_thread + t
                        emit(sot_id, gt, middle_stage, 0,
                             rope_base_per_thread[t])
                        active.append(t)
                        next_ls[t] = 1
                        next_middle += 1
                    elif tail_thread is not None and not tail_candidate_emitted:
                        emit(sot_id, tail_thread, tail_stage, 0,
                             tail_thread * rope_stride)
                        tail_candidate_emitted = True

                if finished:
                    active = [t for t in active if t not in finished]
        thread += K
        stage = tail_stage
    elif doc.tail is not None:
        # No ordinary middle lines. Reserve the tail immediately after warmup.
        emit(sot_id, thread, stage, 0, thread * rope_stride)

    # Tail anchor and final content are row-last. They share the tail thread
    # with the earlier tail-candidate SOT, so natural shift learns
    # <sot> -> <sot_tail> -> tail_c1.
    if doc.tail is not None:
        tail_thread = thread
        base = tail_thread * rope_stride
        emit(sot_tail_id, tail_thread, stage, 1, base + 1)
        for offset, tok in enumerate(doc.tail.content, start=2):
            emit(tok, tail_thread, stage, offset, base + offset)

    return RowMeta(tokens, thread_idx, stage_idx, local_step, rope_idx)


# ---------- targets / doc_idx ----------

def compute_targets_v4(meta: RowMeta, doc: DocV4,
                       bos_id: int = BOS, eot_id: int = EOT) -> List[int]:
    """Per-position targets for the v4 tail-SOT layout.

    Ordinary SOT anchors are engine/layout inputs and naturally target their
    line's first content token. The tail-candidate SOT naturally targets the
    row-last SOT_TAIL marker in the same tail thread. All non-tail line c_L
    positions target EOT; the final tail c_L targets BOS.
    """
    N = meta.N
    inputs = meta.tokens

    traj = defaultdict(list)
    for i in range(N):
        if meta.thread_idx[i] < 0:
            continue
        traj[meta.thread_idx[i]].append((meta.local_step[i], i))
    for gt in traj:
        traj[gt].sort()

    next_in_thread: List[int] = [-1] * N
    for entries in traj.values():
        for k in range(len(entries) - 1):
            _, pos = entries[k]
            _, npos = entries[k + 1]
            next_in_thread[pos] = npos

    targets: List[int] = [-1] * N
    for i in range(N):
        if meta.thread_idx[i] < 0:
            continue
        npos = next_in_thread[i]
        if npos != -1:
            targets[i] = inputs[npos]

    final_steps = {}
    gt = 0
    for para in doc.warmup:
        final_steps[gt] = (para.length, False)
        gt += 1
    for para in doc.middle:
        final_steps[gt] = (para.length, False)
        gt += 1
    if doc.tail is not None:
        # Tail thread local_step: 0=<sot>, 1=<sot_tail>, 2..=content.
        final_steps[gt] = (doc.tail.length + 1, True)
    else:
        # One-block document: parent is also the final line.
        final_gt = len(doc.warmup) - 1
        final_len = doc.warmup[-1].length
        final_steps[final_gt] = (final_len, True)

    for i in range(N):
        gt = meta.thread_idx[i]
        if gt < 0 or gt not in final_steps:
            continue
        final_ls, is_tail = final_steps[gt]
        if meta.local_step[i] != final_ls:
            continue
        targets[i] = bos_id if is_tail else eot_id

    return targets


def compute_doc_idx(meta: RowMeta) -> List[int]:
    """Single packed doc per row in stage-1 dumps."""
    return [0] * meta.N

