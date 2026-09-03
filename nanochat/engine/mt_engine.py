"""
Batched decode engine for nanochat.

Generic shell + two decode loops:
  - generate_ar             — vanilla AR; T_new=1 always; EOS termination.
  - generate_multithread_v4 — Block-MT v4 tail-SOT state machine with
                              per-row natural T_new.

Shared infrastructure (PagedKVCache + BlockManager + sampling) lives on
the BatchedEngine class. Sampling primitives (apply_top_k, apply_top_p,
sample_token) are lifted from WeDLM `engine/sampler.py:63-167`.

The class lives alongside `vanilla_engine.py` (single-row AR engine). The
package-level `Engine` export still points to the vanilla path; import this
module directly for the MT batched engine.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from nanochat.engine.cache import PagedKVCache, BlockManager
from nanochat.engine.context import Context
from nanochat.engine.trace_log import MTTraceLog, record_signal, record_decision_topk


# ============================================================================
# Sampling / caps configs
# ============================================================================

@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 42


@dataclass
class SafetyCaps:
    max_tokens: int = 1024  # hard cap per row, including prompt


# ============================================================================
# Sampling primitives — lifted from WeDLM engine/sampler.py:63-167
# ============================================================================

def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    to_remove = cumulative > top_p
    to_remove[..., 1:] = to_remove[..., :-1].clone()
    to_remove[..., 0] = False
    indices_to_remove = to_remove.scatter(dim=-1, index=sorted_indices, src=to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def sample_token(logits: torch.Tensor, sampling: SamplingConfig,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one token per row. logits: (N, V); returns (N,) int64.

    temperature == 0 → greedy argmax; otherwise top_k → top_p → softmax → multinomial.
    """
    if sampling.temperature == 0.0:
        return logits.argmax(dim=-1)
    logits = apply_top_k(logits, sampling.top_k)
    logits = apply_top_p(logits, sampling.top_p)
    probs = F.softmax(logits / sampling.temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


# ============================================================================
# AR per-row state
# ============================================================================

@dataclass
class ARRowState:
    block_table: list[int]
    tokens: list[int]              # all tokens emitted (prompt + generated)
    prompt_len: int
    cache_seqlens: int = 0         # current K length in the paged cache
    last_token: int | None = None  # token to feed next decode step
    terminated: bool = False
    stop_reason: str | None = None


# ============================================================================
# BatchedEngine
# ============================================================================

class BatchedEngine:
    """Batched decode engine with paged KV cache + varlen forward path."""

    def __init__(self, model, num_blocks: int, block_size: int = 256,
                 device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.model = model
        self.device = torch.device(device)
        self.block_size = block_size
        cfg = model.config
        head_dim = cfg.n_embd // cfg.n_head
        self.cache = PagedKVCache(
            num_layers=cfg.n_layer,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=cfg.n_kv_head,
            head_dim=head_dim,
            device=self.device,
            dtype=dtype,
        )
        self.cache.attach(model)
        self.block_manager = BlockManager(num_blocks)

    # ---- block / slot helpers --------------------------------------------

    def _allocate_row_blocks(self, max_total_tokens: int) -> list[int]:
        n = (max_total_tokens + self.block_size - 1) // self.block_size
        return self.block_manager.allocate(n)

    def _release_row_blocks(self, ids: list[int]) -> None:
        self.block_manager.release(ids)

    def _slots_for_row(self, block_table: list[int], start: int, n: int) -> list[int]:
        """Absolute paged-cache slots for n consecutive logical positions
        starting at `start` for a row with the given `block_table`."""
        slots = []
        for j in range(n):
            logical = start + j
            slots.append(
                block_table[logical // self.block_size] * self.block_size
                + logical % self.block_size
            )
        return slots

    def _block_tables_tensor(self, tables: list[list[int]]) -> torch.Tensor:
        max_len = max(len(t) for t in tables)
        padded = [t + [-1] * (max_len - len(t)) for t in tables]
        return torch.tensor(padded, dtype=torch.int32, device=self.device)

    def _signal_forbid_mask(self, specials: "MultithreadSpecials",
                            V: int, device: torch.device) -> torch.Tensor:
        """Cached `(V,)` bool mask: True everywhere EXCEPT `specials.signal_set`.
        Used at the DECISION_TAIL slot to force the sampled token into
        `{<bos>, <bot_1>..<bot_kmax>}`, matching the supervised target at
        the decision thread's c_L slot (multithread_layout:compute_targets).
        Without this mask, an off-distribution sample (e.g. content token at
        temp>0) would crash `bot_id_to_k[sampled_post]` in DECISION_TAIL."""
        key = (id(specials), V)
        if getattr(self, "_signal_mask_key", None) != key:
            forbid = torch.ones(V, dtype=torch.bool, device=device)
            for sig in specials.signal_set:
                forbid[sig] = False
            self._signal_forbid_cache = forbid
            self._signal_mask_key = key
        return self._signal_forbid_cache

    # ---- prefill ---------------------------------------------------------

    def _prefill_one(self, state: ARRowState, prompt: list[int]) -> torch.Tensor:
        """B=1 varlen prefill for one prompt; returns logits (L, V).

        Sets state.cache_seqlens to L on success.
        """
        L = len(prompt)
        idx = torch.tensor(prompt, dtype=torch.long, device=self.device)
        rope_idx = torch.arange(L, dtype=torch.long, device=self.device)
        cu_q = torch.tensor([0, L], dtype=torch.int32, device=self.device)
        cu_k = torch.tensor([0, L], dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(
            self._slots_for_row(state.block_table, 0, L),
            dtype=torch.int32, device=self.device,
        )
        block_tables = self._block_tables_tensor([state.block_table])
        ctx = Context(
            is_prefill=True,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=L,
            max_seqlen_k=L,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )
        logits = self.model(idx=idx, rope_idx=rope_idx, cache_ctx=ctx)
        state.cache_seqlens = L
        return logits  # (L, V)

    # ---- AR generation ---------------------------------------------------

    @torch.inference_mode()
    def generate_ar(self,
                    prompts: list[list[int]],
                    sampling: SamplingConfig,
                    caps: SafetyCaps,
                    eos_token_ids: set[int]) -> list[ARRowState]:
        """Vanilla AR generation, batched. Sequential B=1 prefill per row,
        then lockstep T_new=1 decode until each row hits EOS or max_tokens.

        Returns one `ARRowState` per input prompt with the full token stream
        (prompt + generation) on `state.tokens` and the termination reason.
        """
        N = len(prompts)
        rng = torch.Generator(device=self.device).manual_seed(sampling.seed)

        # 1) Allocate blocks per row.
        states = [
            ARRowState(
                block_table=self._allocate_row_blocks(caps.max_tokens),
                tokens=list(p),
                prompt_len=len(p),
            )
            for p in prompts
        ]

        # 2) Per-row prefill. Sample first decode token from last prefill logit.
        for state in states:
            logits = self._prefill_one(state, state.tokens)
            sampled = sample_token(logits[-1:], sampling, rng).item()
            if sampled in eos_token_ids:
                state.terminated = True
                state.stop_reason = "eos"
                continue
            if state.cache_seqlens + 1 > caps.max_tokens:
                state.terminated = True
                state.stop_reason = "max_tokens"
                continue
            state.last_token = sampled
            state.tokens.append(sampled)

        # 3) Lockstep T_new=1 decode. Terminated rows feed PAD; their logits
        #    are ignored but cache_seqlens advances uniformly (waste bounded
        #    by caps.max_tokens, since blocks are allocated for that horizon).
        while not all(s.terminated for s in states):
            tokens_list, rope_idx_list = [], []
            cu_q_list, cu_k_list = [0], [0]
            slot_map_list = []

            for state in states:
                tok = state.last_token if not state.terminated else 0  # 0 = PAD
                tokens_list.append(tok)
                rope_idx_list.append(state.cache_seqlens)
                cu_q_list.append(cu_q_list[-1] + 1)
                cu_k_list.append(cu_k_list[-1] + state.cache_seqlens + 1)
                slot_map_list.append(
                    self._slots_for_row(state.block_table, state.cache_seqlens, 1)[0]
                )

            idx = torch.tensor(tokens_list, dtype=torch.long, device=self.device)
            rope_idx = torch.tensor(rope_idx_list, dtype=torch.long, device=self.device)
            ctx = Context(
                is_prefill=False,
                cu_seqlens_q=torch.tensor(cu_q_list, dtype=torch.int32, device=self.device),
                cu_seqlens_k=torch.tensor(cu_k_list, dtype=torch.int32, device=self.device),
                max_seqlen_q=1,
                max_seqlen_k=max(s.cache_seqlens + 1 for s in states),
                slot_mapping=torch.tensor(slot_map_list, dtype=torch.int32, device=self.device),
                block_tables=self._block_tables_tensor([s.block_table for s in states]),
            )

            logits = self.model(idx=idx, rope_idx=rope_idx, cache_ctx=ctx)  # (N, V)
            sampled = sample_token(logits, sampling, rng).tolist()

            for state, tok in zip(states, sampled):
                state.cache_seqlens += 1  # advance even terminated (block waste bounded)
                if state.terminated:
                    continue
                if tok in eos_token_ids:
                    state.terminated = True
                    state.stop_reason = "eos"
                elif state.cache_seqlens >= caps.max_tokens:
                    state.terminated = True
                    state.stop_reason = "max_tokens"
                else:
                    state.last_token = tok
                    state.tokens.append(tok)

        # 4) Release blocks back to the pool.
        for state in states:
            self._release_row_blocks(state.block_table)

        return states

    @torch.inference_mode()
    def generate_multithread_v4(self,
                                prompts: list[list[int]],
                                specials: "MultithreadSpecials",
                                sampling: SamplingConfig,
                                caps: SafetyCaps,
                                rope_stride: int = 256,
                                decode_policy: str = "mirror_train",
                                target_paragraphs: int = 0,
                                warmup_threshold: int = 16,
                                v4_parallel_cap: int | None = None) -> list["MTRowState"]:
        """Block-MT v4 decode loop.

        v4 removes active <bot_K> control rows. The engine injects ordinary
        <sot> anchors. A tail-candidate <sot> may sample <|sot_tail|>; the
        engine holds that marker until all active ordinary middle lines retire,
        then feeds <|sot_tail|> row-last and decodes the final tail line
        sequentially.

        v4_parallel_cap optionally caps active ordinary middle lines. None
        preserves the original uncapped v4 SOT ramp; positive values delay new
        ordinary/tail-candidate <sot> injection until a slot opens.

        decode_policy:
          - mirror_train: use the model's online <sot> -> <|sot_tail|>
            decision directly.
          - chunked: once `target_paragraphs - 1` paragraphs are allocated,
            force the next newly injected <sot> to reserve the tail. This is
            a bounded smoke/diagnostic policy, not a v4 training semantic.
        """
        assert specials.sot_tail is not None, "v4 generation requires specials.sot_tail"
        assert decode_policy in ("chunked", "mirror_train")
        assert warmup_threshold >= 0, f"warmup_threshold must be >= 0, got {warmup_threshold}"
        if v4_parallel_cap == 0:
            v4_parallel_cap = None
        if v4_parallel_cap is not None:
            assert v4_parallel_cap >= 1, (
                f"v4_parallel_cap must be >= 1 or None, got {v4_parallel_cap}"
            )

        rng = torch.Generator(device=self.device).manual_seed(sampling.seed)
        states = []
        for prompt in prompts:
            L = len(prompt)
            block_table = self._allocate_row_blocks(caps.max_tokens)
            bos_offset = 1 if (L > 0 and prompt[0] == specials.bos) else 0
            states.append(MTRowState(
                block_table=block_table,
                tokens=list(prompt),
                token_paragraphs=[0] * L,
                token_wave_steps=[0] * L,
                prompt_len=L,
                cache_seqlens=0,
                phase="V4_SEQ",
                threads={
                    0: ThreadState(
                        paragraph_idx=0,
                        local_step=L,
                        is_decision=False,
                        alive=True,
                        queued=None,
                    ),
                },
                p_next=1,
                K=1,
                paragraphs_emitted=1,
                target_paragraphs=target_paragraphs,
                decode_policy=decode_policy,
                trace_log=MTTraceLog(),
                warmup_threshold=warmup_threshold,
                warmup_bos_offset=bos_offset,
                v4_parallel_cap=v4_parallel_cap,
            ))

        # Per-row prefill. The sampled token is the target of the prompt's
        # final input token, so handle it as a V4_SEQ sample without advancing
        # cache_seqlens (the prompt itself is already in cache).
        for state in states:
            logits = self._prefill_one(state, state.tokens)
            sampled = sample_token(logits[-1:], sampling, rng).item()
            handle_v4_seq_sample(state, sampled, specials, caps, rope_stride)

        for state in states:
            state.wave_step = 1
            while len(state.token_wave_steps) < len(state.tokens):
                state.token_wave_steps.append(state.wave_step)

        while not all(s.terminated for s in states):
            for state in states:
                if not state.terminated:
                    state.wave_step += 1

            all_tokens, all_ropes = [], []
            cu_q_list, cu_k_list = [0], [0]
            slot_map_list = []
            per_row_T_new = []
            per_row_owners = []

            for state in states:
                if state.terminated:
                    tok_chunk, rope_chunk, owners_chunk = [0], [0], [-1]
                    T_new_row = 1
                    row_slots = [state.block_table[0] * self.block_size]
                else:
                    tok_chunk, rope_chunk, owners_chunk = next_to_feed_multithread_v4(
                        state, specials, rope_stride,
                    )
                    T_new_row = len(tok_chunk)
                    row_slots = self._slots_for_row(
                        state.block_table, state.cache_seqlens, T_new_row,
                    )

                all_tokens.extend(tok_chunk)
                all_ropes.extend(rope_chunk)
                cu_q_list.append(cu_q_list[-1] + T_new_row)
                cu_k_list.append(cu_k_list[-1] + state.cache_seqlens + T_new_row)
                slot_map_list.extend(row_slots)
                per_row_T_new.append(T_new_row)
                per_row_owners.append(owners_chunk)

            idx = torch.tensor(all_tokens, dtype=torch.long, device=self.device)
            rope_idx = torch.tensor(all_ropes, dtype=torch.long, device=self.device)
            ctx = Context(
                is_prefill=False,
                cu_seqlens_q=torch.tensor(cu_q_list, dtype=torch.int32, device=self.device),
                cu_seqlens_k=torch.tensor(cu_k_list, dtype=torch.int32, device=self.device),
                max_seqlen_q=max(per_row_T_new),
                max_seqlen_k=max(s.cache_seqlens + per_row_T_new[r] for r, s in enumerate(states)),
                slot_mapping=torch.tensor(slot_map_list, dtype=torch.int32, device=self.device),
                block_tables=self._block_tables_tensor([s.block_table for s in states]),
            )
            logits = self.model(idx=idx, rope_idx=rope_idx, cache_ctx=ctx)

            for r, state in enumerate(states):
                if state.terminated:
                    continue
                row_start, row_end = cu_q_list[r], cu_q_list[r + 1]
                chunk_samples = sample_token(logits[row_start:row_end], sampling, rng).tolist()
                advance_multithread_v4(
                    state, per_row_owners[r], chunk_samples, per_row_T_new[r],
                    specials, caps, rope_stride,
                )

            for state in states:
                while len(state.token_wave_steps) < len(state.tokens):
                    state.token_wave_steps.append(state.wave_step)

        for state in states:
            self._release_row_blocks(state.block_table)
        return states


# ============================================================================
# Multithread per-row state + state-machine helpers
# ============================================================================

@dataclass
class ThreadState:
    """One thread inside a wave (or the parent paragraph as a degenerate K=1 wave)."""
    paragraph_idx: int
    local_step: int = 0           # next slot to feed for this thread
    is_decision: bool = False
    alive: bool = True
    alive_mid_wave: bool = True   # decision flips False after rollback
    queued: int | None = None     # sampled-but-not-yet-fed token
    # Under run-preservation (2026-05-14): non-decision threads sample
    # newline-bearing tokens as run-extension content (fed back) and
    # retire only when they sample `<eot>` (target-only override at c_L).
    # `in_run` flips True after the first newline-bearing sample so we
    # can detect off-distribution NO_NL samples mid-run and force-retire.
    in_run: bool = False


@dataclass
class MTRowState:
    block_table: list[int]
    tokens: list[int]                          # full row in linearized order
    prompt_len: int
    # Per-token paragraph_idx, parallel to `tokens` (same length, same order).
    # Used by `format_multithread_paragraphs` to un-interleave siblings back
    # into per-paragraph token lists for human-readable decoding.
    token_paragraphs: list[int] = field(default_factory=list)
    # Per-token wave_step, parallel to `tokens`. Prompt tokens are step 0;
    # the prefill sample is step 1; each outer-loop iteration of
    # `generate_multithread` bumps active rows' wave_step by 1, so every
    # token emitted during that iteration (including K parallel siblings)
    # shares the same value. Consumed by the gen-trace dumper to render
    # sibling tokens as one viewer-step.
    token_wave_steps: list[int] = field(default_factory=list)
    wave_step: int = 0
    cache_seqlens: int = 0
    phase: str = "PARENT"                      # PARENT|WAVE_START|WAVE_STEP|DECISION_TAIL|TERMINATED
    threads: dict[int, ThreadState] = field(default_factory=dict)
    p_next: int = 1                            # next paragraph_idx to allocate
    K: int = 1                                 # current wave width (1 during PARENT)
    # decision_buffer holds the decision thread's c_L (just-sampled when
    # the model samples a newline-bearing token mid-wave for K>=2).
    # DECISION_TAIL re-feeds it at end-of-wave; the sample at c_L's
    # logit is the control signal. Under Interpretation 2 the entire
    # DECISION_TAIL phase is 1 sub-step (no `<nl>` slot).
    decision_buffer: int | None = None
    # `<bot-K>` is dual-role: target at the prior decision's <nl> slot
    # AND input right before the next wave's first <sot>. When a signal
    # is sampled, we stash the bot token + its rope here so WAVE_START
    # can prepend it to sot_0 in the same forward chunk. See
    # multithread_layout.py.
    bot_pending: int | None = None
    bot_pending_rope: int | None = None
    # Waterfall ramp counter: number of threads that have already emitted
    # their <sot> (and are therefore producing content). Starts at 0 after
    # setup_wave; bumps to 1 after WAVE_START's sot_0; bumps once per
    # subsequent WAVE_STEP until reaching K (then full-width, no more sots).
    n_started: int = 0
    paragraphs_emitted: int = 1                # parent counts
    target_paragraphs: int = 0
    decode_policy: str = "chunked"
    kmax: int = 16                              # K_max for the chunked policy
    stop_reason: str | None = None
    # Optional diagnostic capture (signal events + decision logit
    # snapshots). Allocated by `generate_multithread`; helpers in
    # `nanochat.engine.trace_log` no-op when this is None.
    trace_log: MTTraceLog | None = None
    # v3 bot-1 warmup variant. `warmup_threshold` is
    # the per-doc warmup budget in CONTENT tokens (excludes a leading
    # BOS). While `warmup_remaining > 0` (computed dynamically from
    # `len(tokens) - warmup_bos_offset`), any sampled signal at a c_L
    # slot is post-hoc forced to `<bot-1>`, mirroring the training-side
    # bot-1 supervision. 0 = v2 behavior (no forcing).
    warmup_threshold: int = 0
    warmup_bos_offset: int = 0  # 1 if prompt starts with BOS, else 0
    # v4 tail-SOT state. `tail_thread` is the paragraph id whose earlier
    # <sot> sampled <|sot_tail|>; the engine feeds that <|sot_tail|> only
    # after all ordinary middle lines retire.
    tail_thread: int | None = None
    # v4 optional ablation cap for active ordinary middle lines. None keeps
    # the original uncapped SOT ramp.
    v4_parallel_cap: int | None = None

    @property
    def terminated(self) -> bool:
        return self.phase == "TERMINATED"

    @property
    def warmup_remaining(self) -> int:
        """How many more content tokens before warmup ends. 0 = past
        warmup (or warmup disabled). `tokens` is content-only after the
        optional leading BOS — specials (`<sot>`, `<bot-K>`, `<eot>`)
        are never appended to it, so subtracting `warmup_bos_offset`
        gives the true content count."""
        if self.warmup_threshold <= 0:
            return 0
        content_count = len(self.tokens) - self.warmup_bos_offset
        return max(0, self.warmup_threshold - content_count)


@dataclass
class MultithreadSpecials:
    bos: int
    sot: int
    eot: int                                   # target-only end-of-thread override
    bot_per_k: dict[int, int]                  # K -> token id, K in 1..16
    sot_tail: int | None = None                # v4 tail-line anchor
    # Under run-preservation (2026-05-14): `<eot>` is a target-only
    # override at non-decision c_L slots (never fed back as input).
    # `nl_token_ids` is the set of all newline-bearing tokens (`\n`/`\r`
    # in decoded bytes); engine uses it to detect run-extension content
    # for non-decision threads. Precomputed at engine init via
    # `compute_nl_token_ids(tokenizer)` on the production tokenizer.
    nl_token_ids: set[int] = field(default_factory=set)

    def __post_init__(self):
        self.bot_id_to_k: dict[int, int] = {tok: k for k, tok in self.bot_per_k.items()}
        # signal_set covers DECISION-thread transition signals only —
        # `<bot-K>` for non-final stages, `<bos>` for the final stage.
        # `<eot>` is NOT in signal_set (it's the NON-decision retire
        # signal, handled separately).
        self.signal_set: set[int] = set(self.bot_per_k.values()) | {self.bos}


def v4_active_ordinary_threads(state: MTRowState) -> list[int]:
    return [
        tid for tid, t in sorted(state.threads.items())
        if t.alive and tid != state.tail_thread
    ]


def v4_can_start_new_ordinary_thread(state: MTRowState,
                                     active_at_step_start: list[int]) -> bool:
    cap = state.v4_parallel_cap
    if cap == 0:
        cap = None
    if cap is not None:
        assert cap >= 1, f"v4_parallel_cap must be >= 1 or None, got {cap}"
    return cap is None or len(active_at_step_start) < cap


def setup_v4_seq_start(state: MTRowState) -> None:
    pid = state.p_next
    state.p_next += 1
    state.paragraphs_emitted += 1
    state.threads = {
        pid: ThreadState(
            paragraph_idx=pid,
            local_step=0,
            is_decision=False,
            alive=True,
            queued=None,
            in_run=False,
        )
    }
    state.K = 1
    state.phase = "V4_SEQ_START"


def setup_v4_ramp(state: MTRowState) -> None:
    state.threads = {}
    state.K = 0
    state.n_started = 0
    state.tail_thread = None
    state.phase = "V4_RAMP"


def handle_v4_line_end(state: MTRowState, specials: MultithreadSpecials,
                       rope_stride: int) -> None:
    if state.warmup_remaining > 0:
        setup_v4_seq_start(state)
    else:
        setup_v4_ramp(state)


def handle_v4_seq_sample(state: MTRowState, sampled: int,
                         specials: MultithreadSpecials,
                         caps: SafetyCaps, rope_stride: int) -> None:
    t = next(iter(state.threads.values()))
    if sampled == specials.bos:
        state.phase = "TERMINATED"
        state.stop_reason = "eos"
    elif sampled == specials.eot:
        handle_v4_line_end(state, specials, rope_stride)
    elif sampled == specials.sot_tail or sampled in specials.bot_id_to_k:
        # Off-distribution for a sequential content slot. Treat as a line end
        # so the engine can keep moving without feeding structural junk.
        handle_v4_line_end(state, specials, rope_stride)
    else:
        t.queued = sampled
        state.tokens.append(sampled)
        state.token_paragraphs.append(t.paragraph_idx)


def should_force_v4_tail(state: MTRowState) -> bool:
    return (
        state.decode_policy == "chunked"
        and state.target_paragraphs > 0
        and state.paragraphs_emitted >= max(1, state.target_paragraphs - 1)
    )


def next_to_feed_multithread_v4(state: MTRowState,
                                specials: MultithreadSpecials,
                                rope_stride: int):
    """Returns (tokens, ropes, owners) for v4 tail-SOT generation.

    Owners are paragraph ids. Structural anchors are real inputs to the model
    but are not appended to `state.tokens`, matching the v3 engine convention
    where dumpers reconstruct anchors from thread metadata.
    """
    if state.phase == "V4_SEQ":
        t = next(iter(state.threads.values()))
        return ([t.queued], [t.paragraph_idx * rope_stride + t.local_step], [t.paragraph_idx])

    if state.phase == "V4_SEQ_START":
        t = next(iter(state.threads.values()))
        return ([specials.sot], [t.paragraph_idx * rope_stride], [t.paragraph_idx])

    if state.phase in ("V4_RAMP", "V4_TAIL_WAIT"):
        tokens, ropes, owners = [], [], []
        active_at_step_start = v4_active_ordinary_threads(state)
        for tid in active_at_step_start:
            t = state.threads[tid]
            assert t.queued is not None, f"active v4 thread {tid} has no queued token"
            tokens.append(t.queued)
            ropes.append(t.paragraph_idx * rope_stride + t.local_step)
            owners.append(tid)

        if (state.phase == "V4_RAMP" and state.tail_thread is None
                and v4_can_start_new_ordinary_thread(state, active_at_step_start)):
            pid = state.p_next
            state.p_next += 1
            state.threads[pid] = ThreadState(
                paragraph_idx=pid,
                local_step=0,
                is_decision=False,
                alive=True,
                queued=None,
                in_run=False,
            )
            tokens.append(specials.sot)
            ropes.append(pid * rope_stride)
            owners.append(pid)

        if not tokens and state.tail_thread is not None:
            state.phase = "V4_TAIL_START"
            return next_to_feed_multithread_v4(state, specials, rope_stride)
        assert tokens, f"v4 state produced no tokens to feed in phase {state.phase}"
        return tokens, ropes, owners

    if state.phase == "V4_TAIL_START":
        t = state.threads[state.tail_thread]
        return ([specials.sot_tail], [t.paragraph_idx * rope_stride + t.local_step], [state.tail_thread])

    if state.phase == "V4_TAIL":
        t = state.threads[state.tail_thread]
        return ([t.queued], [t.paragraph_idx * rope_stride + t.local_step], [state.tail_thread])

    raise ValueError(f"unknown v4 phase: {state.phase}")


def advance_multithread_v4(state: MTRowState, owners: list[int], samples: list[int],
                           T_new: int, specials: MultithreadSpecials,
                           caps: SafetyCaps, rope_stride: int) -> None:
    assert len(owners) == len(samples) == T_new, (owners, samples, T_new)
    state.cache_seqlens += T_new

    if state.phase == "V4_SEQ_START":
        tid = owners[0]
        t = state.threads[tid]
        sampled = samples[0]
        t.local_step = 1
        if sampled == specials.bos:
            state.phase = "TERMINATED"
            state.stop_reason = "eos"
        elif sampled == specials.eot or sampled == specials.sot_tail or sampled in specials.bot_id_to_k:
            handle_v4_line_end(state, specials, rope_stride)
        else:
            t.queued = sampled
            state.tokens.append(sampled)
            state.token_paragraphs.append(t.paragraph_idx)
            state.phase = "V4_SEQ"

    elif state.phase == "V4_SEQ":
        tid = owners[0]
        t = state.threads[tid]
        sampled = samples[0]
        t.local_step += 1
        handle_v4_seq_sample(state, sampled, specials, caps, rope_stride)

    elif state.phase in ("V4_RAMP", "V4_TAIL_WAIT"):
        for tid, sampled in zip(owners, samples):
            t = state.threads[tid]
            is_new_anchor = t.local_step == 0 and t.queued is None
            if is_new_anchor:
                if should_force_v4_tail(state):
                    sampled = specials.sot_tail
                t.local_step = 1
                if sampled == specials.sot_tail:
                    state.tail_thread = tid
                    t.alive = False
                    state.paragraphs_emitted += 1
                    record_signal(state.trace_log, state.wave_step, t.paragraph_idx,
                                  sampled, None)
                elif sampled == specials.eot or sampled == specials.bos or sampled in specials.bot_id_to_k:
                    t.alive = False
                else:
                    t.queued = sampled
                    t.alive = True
                    state.paragraphs_emitted += 1
                    state.n_started += 1
                    state.tokens.append(sampled)
                    state.token_paragraphs.append(t.paragraph_idx)
                continue

            t.local_step += 1
            if sampled == specials.eot:
                t.alive = False
            elif sampled == specials.bos or sampled == specials.sot_tail or sampled in specials.bot_id_to_k:
                t.alive = False
            else:
                t.queued = sampled
                state.tokens.append(sampled)
                state.token_paragraphs.append(t.paragraph_idx)

        if state.tail_thread is not None:
            if v4_active_ordinary_threads(state):
                state.phase = "V4_TAIL_WAIT"
            else:
                state.phase = "V4_TAIL_START"
        else:
            state.phase = "V4_RAMP"

    elif state.phase == "V4_TAIL_START":
        tid = owners[0]
        t = state.threads[tid]
        sampled = samples[0]
        t.local_step = 2
        if sampled == specials.bos or sampled == specials.eot:
            state.phase = "TERMINATED"
            state.stop_reason = "eos"
        elif sampled == specials.sot_tail or sampled in specials.bot_id_to_k:
            state.phase = "TERMINATED"
            state.stop_reason = "invalid_tail"
        else:
            t.queued = sampled
            state.tokens.append(sampled)
            state.token_paragraphs.append(t.paragraph_idx)
            state.phase = "V4_TAIL"

    elif state.phase == "V4_TAIL":
        tid = owners[0]
        t = state.threads[tid]
        sampled = samples[0]
        t.local_step += 1
        if sampled == specials.bos or sampled == specials.eot:
            state.phase = "TERMINATED"
            state.stop_reason = "eos"
        elif sampled == specials.sot_tail or sampled in specials.bot_id_to_k:
            state.phase = "TERMINATED"
            state.stop_reason = "invalid_tail"
        else:
            t.queued = sampled
            state.tokens.append(sampled)
            state.token_paragraphs.append(t.paragraph_idx)

    else:
        raise ValueError(f"unknown v4 phase: {state.phase}")

    if state.cache_seqlens >= caps.max_tokens and not state.terminated:
        state.phase = "TERMINATED"
        state.stop_reason = "max_tokens"


# ============================================================================
# Multithread row -> human-readable paragraphs
# ============================================================================

def format_multithread_paragraphs(state: MTRowState, tokenizer) -> list[str]:
    """Un-interleave a multithread row by paragraph_idx and decode each paragraph.

    `state.tokens` is the row in ls-major linearized order (siblings of a
    K>=2 wave appear interleaved token-by-token). `state.token_paragraphs`
    is the per-token paragraph_idx companion list. This helper groups
    tokens by paragraph_idx (preserving emission order within each paragraph),
    decodes each group with the tokenizer, and returns the list of
    paragraph strings in paragraph_idx order.

    Joining the result with "\\n\\n" produces a human-readable rendering of
    what the model emitted, with sibling paragraphs separated cleanly.
    """
    assert len(state.tokens) == len(state.token_paragraphs), (
        f"tokens / token_paragraphs length mismatch: "
        f"{len(state.tokens)} vs {len(state.token_paragraphs)}"
    )
    paras: dict[int, list[int]] = {}
    for tok, pid in zip(state.tokens, state.token_paragraphs):
        paras.setdefault(pid, []).append(tok)
    return [tokenizer.decode(paras[pid]) for pid in sorted(paras)]


def format_multithread_readable(state: MTRowState, tokenizer,
                                separator: str = "\n\n") -> str:
    """Convenience: paragraphs joined into a single readable string."""
    return separator.join(format_multithread_paragraphs(state, tokenizer))
