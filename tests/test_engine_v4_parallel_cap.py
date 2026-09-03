"""CPU-only tests for Block-MT v4 runtime parallel-cap state."""

from nanochat.engine.mt_engine import (
    MTRowState,
    MultithreadSpecials,
    SafetyCaps,
    advance_multithread_v4,
    next_to_feed_multithread_v4,
)


def _make_specials():
    return MultithreadSpecials(
        bos=100,
        sot=101,
        eot=102,
        nl_token_ids={10, 99},
        bot_per_k={k: 200 + k for k in range(1, 17)},
        sot_tail=103,
    )


def _state(cap, target=0, policy="mirror_train"):
    return MTRowState(
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
        target_paragraphs=target,
        decode_policy=policy,
        warmup_threshold=0,
        v4_parallel_cap=cap,
    )


def test_cap_one_delays_new_sot_until_active_line_retires():
    specials = _make_specials()
    caps = SafetyCaps(max_tokens=1024)
    s = _state(cap=1)

    toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
    assert toks == [specials.sot]
    assert ropes == [256]
    assert owners == [1]
    advance_multithread_v4(s, owners, [11], len(toks), specials, caps, 256)

    toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
    assert toks == [11]
    assert ropes == [257]
    assert owners == [1]
    advance_multithread_v4(s, owners, [specials.eot], len(toks), specials, caps, 256)

    toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
    assert toks == [specials.sot]
    assert ropes == [512]
    assert owners == [2]


def test_cap_two_allows_two_then_waits_for_slot():
    specials = _make_specials()
    caps = SafetyCaps(max_tokens=1024)
    s = _state(cap=2)

    toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
    assert toks == [specials.sot]
    advance_multithread_v4(s, owners, [11], len(toks), specials, caps, 256)

    toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
    assert toks == [11, specials.sot]
    assert ropes == [257, 512]
    assert owners == [1, 2]
    advance_multithread_v4(s, owners, [12, 21], len(toks), specials, caps, 256)

    toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
    assert toks == [12, 21]
    assert ropes == [258, 513]
    assert owners == [1, 2]
    advance_multithread_v4(
        s, owners, [specials.eot, 22], len(toks), specials, caps, 256,
    )

    toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
    assert toks == [22, specials.sot]
    assert ropes == [514, 768]
    assert owners == [2, 3]


def test_uncapped_and_zero_preserve_original_sot_ramp():
    specials = _make_specials()
    caps = SafetyCaps(max_tokens=1024)
    for cap in (None, 0):
        s = _state(cap=cap)

        toks, _, owners = next_to_feed_multithread_v4(s, specials, 256)
        advance_multithread_v4(s, owners, [11], len(toks), specials, caps, 256)

        toks, _, owners = next_to_feed_multithread_v4(s, specials, 256)
        advance_multithread_v4(s, owners, [12, 21], len(toks), specials, caps, 256)

        toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
        assert toks == [12, 21, specials.sot]
        assert ropes == [258, 513, 768]
        assert owners == [1, 2, 3]


def test_tail_candidate_is_delayed_until_cap_slot_opens():
    specials = _make_specials()
    caps = SafetyCaps(max_tokens=1024)
    s = _state(cap=2, target=4, policy="chunked")

    toks, _, owners = next_to_feed_multithread_v4(s, specials, 256)
    advance_multithread_v4(s, owners, [11], len(toks), specials, caps, 256)

    toks, _, owners = next_to_feed_multithread_v4(s, specials, 256)
    advance_multithread_v4(s, owners, [12, 21], len(toks), specials, caps, 256)
    assert s.paragraphs_emitted == 3

    toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
    assert toks == [12, 21]
    assert ropes == [258, 513]
    assert owners == [1, 2]
    advance_multithread_v4(
        s, owners, [specials.eot, 22], len(toks), specials, caps, 256,
    )
    assert s.tail_thread is None

    toks, ropes, owners = next_to_feed_multithread_v4(s, specials, 256)
    assert toks == [22, specials.sot]
    assert ropes == [514, 768]
    assert owners == [2, 3]
    advance_multithread_v4(
        s, owners, [specials.eot, 999], len(toks), specials, caps, 256,
    )
    assert s.tail_thread == 3
    assert s.phase == "V4_TAIL_START"
