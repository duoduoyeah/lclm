"""
CPU-only tests for nanochat.engine.mt_engine.

Covers BlockManager pool, sampling primitives, and the Block-MT v4
tail-SOT state machine via a synthetic trace. Doesn't load a model or
touch GPU. End-to-end forwards on a real checkpoint live in
`scripts/smoke_batched_engine.py` and need a GPU+triton+FA2 install.

The legacy v3 wave-chaining engine (generate_multithread,
next_to_feed_multithread, advance_multithread, setup_wave,
enforce_policy) was removed in the v4-only cleanup; its state-machine
tests were dropped with it.

Run:
    python -m pytest tests/test_batched_engine_state.py -v
"""

import pytest

pytest.importorskip("triton")

import torch

from nanochat.engine.cache import BlockManager
from nanochat.engine.mt_engine import (
    apply_top_k, apply_top_p, sample_token,
    SamplingConfig, SafetyCaps,
    MTRowState, ThreadState, MultithreadSpecials,
    next_to_feed_multithread_v4, advance_multithread_v4,
)


# --------------------------------------------------------------------------
# BlockManager
# --------------------------------------------------------------------------

class TestBlockManager:
    def test_allocate_release_roundtrip(self):
        bm = BlockManager(num_blocks=8)
        assert bm.num_free() == 8
        ids = bm.allocate(3)
        assert len(ids) == 3 and bm.num_free() == 5
        bm.release(ids)
        assert bm.num_free() == 8

    def test_allocate_raises_when_empty(self):
        bm = BlockManager(num_blocks=4)
        bm.allocate(4)
        with pytest.raises(RuntimeError):
            bm.allocate(1)

    def test_partial_release(self):
        bm = BlockManager(num_blocks=10)
        ids = bm.allocate(7)
        bm.release(ids[:3])
        assert bm.num_free() == 6


# --------------------------------------------------------------------------
# Sampling primitives
# --------------------------------------------------------------------------

class TestSampling:
    def test_greedy_temperature_zero(self):
        logits = torch.randn(3, 100)
        out = sample_token(logits, SamplingConfig(temperature=0.0))
        assert torch.equal(out, logits.argmax(dim=-1))

    def test_top_k_masks_correctly(self):
        logits = torch.randn(3, 50)
        masked = apply_top_k(logits, top_k=10)
        kept = (masked != float("-inf")).sum(dim=-1)
        assert torch.all(kept == 10)

    def test_top_k_zero_passthrough(self):
        logits = torch.randn(3, 50)
        masked = apply_top_k(logits, top_k=0)
        assert torch.equal(masked, logits)

    def test_top_p_passthrough_at_one(self):
        logits = torch.randn(3, 50)
        masked = apply_top_p(logits, top_p=1.0)
        assert torch.equal(masked, logits)

    def test_sample_shape(self):
        logits = torch.randn(5, 100)
        rng = torch.Generator().manual_seed(42)
        out = sample_token(logits, SamplingConfig(temperature=0.7, top_k=20, top_p=0.95), rng)
        assert out.shape == (5,) and out.dtype == torch.int64


# --------------------------------------------------------------------------
# Multithread v4 state machine
# --------------------------------------------------------------------------

def _make_specials():
    # Interp 2 (2026-05-11): no `eot` / `nl` scalar fields; instead
    # `nl_token_ids` is the set of any-token-decoding-to-contains-`\n`.
    # The test fixture uses arbitrary ids 10 + 99 as stand-ins; bot_per_k
    # bumped to 200+ to avoid collisions with the nl set.
    return MultithreadSpecials(
        bos=100, sot=101, eot=102,
        nl_token_ids={10, 99},
        bot_per_k={k: 200 + k for k in range(1, 17)},
        sot_tail=103,
    )


class TestV4TailSotStateMachine:
    def test_tail_waits_for_active_lines_then_decodes_row_last(self):
        specials = _make_specials()
        caps = SafetyCaps(max_tokens=1024)
        s = MTRowState(
            block_table=[0, 1, 2, 3],
            tokens=[1, 2, 3, 4],
            token_paragraphs=[0, 0, 0, 0],
            prompt_len=4,
            cache_seqlens=4,
            phase="V4_RAMP",
            threads={},
            p_next=1,
            K=0,
            paragraphs_emitted=1,
            target_paragraphs=4,
            decode_policy="mirror_train",
            warmup_threshold=0,
        )

        toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
        assert toks == [specials.sot]
        assert ropes == [256]
        assert owners == [1]
        advance_multithread_v4(s, owners, [11], len(toks), specials, caps, 256)
        assert s.phase == "V4_RAMP"
        assert s.tokens[-1] == 11
        assert s.paragraphs_emitted == 2

        toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
        assert toks == [11, specials.sot]
        assert ropes == [257, 512]
        assert owners == [1, 2]
        advance_multithread_v4(s, owners, [12, 21], len(toks), specials, caps, 256)
        assert s.phase == "V4_RAMP"
        assert s.tokens[-2:] == [12, 21]
        assert s.paragraphs_emitted == 3

        toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
        assert toks == [12, 21, specials.sot]
        assert ropes == [258, 513, 768]
        assert owners == [1, 2, 3]
        advance_multithread_v4(
            s, owners, [specials.eot, 22, specials.sot_tail],
            len(toks), specials, caps, 256,
        )
        assert s.phase == "V4_TAIL_WAIT"
        assert s.tail_thread == 3
        assert s.paragraphs_emitted == 4

        toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
        assert toks == [22]
        assert ropes == [514]
        assert owners == [2]
        advance_multithread_v4(s, owners, [specials.eot], len(toks), specials, caps, 256)
        assert s.phase == "V4_TAIL_START"

        toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
        assert toks == [specials.sot_tail]
        assert ropes == [769]
        assert owners == [3]
        advance_multithread_v4(s, owners, [31], len(toks), specials, caps, 256)
        assert s.phase == "V4_TAIL"
        assert s.tokens[-1] == 31

        toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
        assert toks == [31]
        assert ropes == [770]
        assert owners == [3]
        advance_multithread_v4(s, owners, [specials.bos], len(toks), specials, caps, 256)
        assert s.phase == "TERMINATED"
        assert s.stop_reason == "eos"
        assert s.tokens == [1, 2, 3, 4, 11, 12, 21, 22, 31]
        assert s.token_paragraphs == [0, 0, 0, 0, 1, 1, 2, 2, 3]
