import pytest

from nanochat.data.multithread_layout import (
    BOS,
    EOT,
    SOT,
    SOT_TAIL,
    DocV4,
    Paragraph,
    compile_doc_v4,
    compute_targets_v4,
)


def _doc():
    return DocV4(
        warmup=[Paragraph([100])],
        middle=[
            Paragraph([200, 201, 202, 203]),
            Paragraph([210, 211, 212]),
            Paragraph([220, 221]),
        ],
        tail=Paragraph([300, 301]),
    )


def _assert_cap_respected(meta, doc, cap):
    first_middle = len(doc.warmup)
    after_middle = first_middle + len(doc.middle)
    active = set()
    max_active = 0
    for tid, ls in zip(meta.thread_idx, meta.local_step):
        if first_middle <= tid < after_middle:
            middle_idx = tid - first_middle
            if ls == 0:
                assert len(active) < cap
                active.add(tid)
            max_active = max(max_active, len(active))
            if ls == doc.middle[middle_idx].length:
                active.remove(tid)
        else:
            max_active = max(max_active, len(active))
    assert max_active <= cap
    assert not active


def test_compile_doc_v4_cap_above_middle_matches_uncapped():
    doc = _doc()
    uncapped = compile_doc_v4(doc, bos_id=BOS)
    capped = compile_doc_v4(doc, bos_id=BOS, v4_parallel_cap=4)

    assert capped.tokens == uncapped.tokens
    assert capped.thread_idx == uncapped.thread_idx
    assert capped.stage_idx == uncapped.stage_idx
    assert capped.local_step == uncapped.local_step
    assert capped.rope_idx == uncapped.rope_idx


def test_compile_doc_v4_cap_two_delays_new_sot_until_slot_opens():
    doc = _doc()
    meta = compile_doc_v4(doc, bos_id=BOS, v4_parallel_cap=2)

    assert meta.tokens == [
        BOS, 100,
        SOT, 200, SOT, 201, 210, 202, 211, 203, 212,
        SOT, 220, SOT, 221,
        SOT_TAIL, 300, 301,
    ]
    assert meta.thread_idx == [
        0, 0,
        1, 1, 2, 1, 2, 1, 2, 1, 2,
        3, 3, 4, 3,
        4, 4, 4,
    ]
    assert meta.local_step == [
        0, 1,
        0, 1, 0, 2, 1, 3, 2, 4, 3,
        0, 1, 0, 2,
        1, 2, 3,
    ]
    assert meta.stage_idx[13] == 2  # tail-candidate SOT
    assert meta.stage_idx[15:] == [2, 2, 2]
    _assert_cap_respected(meta, doc, cap=2)


def test_compile_doc_v4_cap_one_serializes_middle_threads():
    doc = _doc()
    meta = compile_doc_v4(doc, bos_id=BOS, v4_parallel_cap=1)

    assert meta.tokens == [
        BOS, 100,
        SOT, 200, 201, 202, 203,
        SOT, 210, 211, 212,
        SOT, 220, 221,
        SOT, SOT_TAIL, 300, 301,
    ]
    assert meta.thread_idx == [
        0, 0,
        1, 1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3,
        4, 4, 4, 4,
    ]
    _assert_cap_respected(meta, doc, cap=1)


@pytest.mark.parametrize("cap", [1, 2, 4, 8])
def test_compile_doc_v4_planned_caps_respect_active_middle_bound(cap):
    doc = DocV4(
        warmup=[Paragraph([100])],
        middle=[
            Paragraph([200, 201, 202, 203]),
            Paragraph([210, 211, 212]),
            Paragraph([220, 221]),
            Paragraph([230, 231, 232, 233, 234]),
            Paragraph([240]),
        ],
        tail=Paragraph([300, 301]),
    )
    meta = compile_doc_v4(doc, bos_id=BOS, v4_parallel_cap=cap)

    _assert_cap_respected(meta, doc, cap=cap)
    assert meta.tokens.count(SOT_TAIL) == 1
    assert meta.tokens[-3:] == [SOT_TAIL, 300, 301]


def test_compute_targets_v4_unchanged_for_capped_schedule():
    doc = _doc()
    meta = compile_doc_v4(doc, bos_id=BOS, v4_parallel_cap=2)
    targets = compute_targets_v4(meta, doc, bos_id=BOS, eot_id=EOT)

    assert targets[0] == 100          # BOS -> first parent content
    assert targets[2] == 200          # ordinary SOT -> first middle token
    assert targets[4] == 210
    assert targets[11] == 220
    assert targets[13] == SOT_TAIL    # tail-candidate SOT -> row-last anchor
    assert targets[15] == 300         # SOT_TAIL -> first tail token
    assert targets[9] == EOT          # middle c_L targets are target-only EOT
    assert targets[10] == EOT
    assert targets[14] == EOT
    assert targets[17] == BOS         # final tail c_L terminates the row


def test_compile_doc_v4_rejects_nonpositive_cap():
    with pytest.raises(AssertionError):
        compile_doc_v4(_doc(), bos_id=BOS, v4_parallel_cap=0)

class TinyV4Tokenizer:
    def __init__(self):
        self.specials = {
            "<|bos|>": BOS,
            "<|sot|>": SOT,
            "<|eot|>": EOT,
            "<|sot_tail|>": SOT_TAIL,
        }
        for k in range(1, 17):
            self.specials[f"<|bot_{k}|>"] = 1000 + k

    def get_vocab_size(self):
        return 400

    def encode_special(self, text):
        return self.specials[text]

    def get_bos_token_id(self):
        return BOS

    def get_special_tokens(self):
        return list(self.specials)

    def decode(self, ids):
        if len(ids) != 1:
            return "".join(self.decode([i]) for i in ids)
        tid = ids[0]
        if tid == 10:
            return "\n"
        return f"x{tid}"

    def encode(self, text, num_threads=1):
        def one(_text):
            if _text == "\n":
                return [10]
            return [200, 10, 210, 10, 220, 10, 230, 10, 240, 10]
        if isinstance(text, list):
            return [one(t) for t in text]
        return one(text)


def test_v4_dataloader_threads_parallel_cap_into_training_compiler(monkeypatch):
    import nanochat.data.multithread_dataloader as mdl

    seen_caps = []
    real_compile = mdl.compile_doc_v4

    def spy_compile_doc_v4(*args, **kwargs):
        seen_caps.append(kwargs.get("v4_parallel_cap"))
        return real_compile(*args, **kwargs)

    def fake_batches(split, resume_state_dict, tokenizer_batch_size):
        while True:
            yield ["doc"], (0, 0, 1)

    monkeypatch.setattr(mdl, "compile_doc_v4", spy_compile_doc_v4)
    monkeypatch.setattr(mdl, "_document_batches", fake_batches)

    loader = mdl.tokenizing_distributed_data_loader_with_state_multithread(
        TinyV4Tokenizer(), B=1, T=64, split="train", device="cpu",
        warmup_threshold=0, layout_version="v4", v4_parallel_cap=2,
        buffer_size=1,
    )
    inputs, targets, rope_idx, state = next(loader)

    assert seen_caps and all(cap == 2 for cap in seen_caps)
    assert SOT_TAIL in inputs[0].tolist()
    assert state == {"pq_idx": 0, "rg_idx": 0, "epoch": 1}


def test_v4_parallel_cap_rejected_for_non_v4_loader():
    import nanochat.data.multithread_dataloader as mdl

    loader = mdl.tokenizing_distributed_data_loader_with_state_multithread(
        TinyV4Tokenizer(), B=1, T=64, split="train", device="cpu",
        layout_version="v3", v4_parallel_cap=2,
    )
    with pytest.raises(AssertionError):
        next(loader)
