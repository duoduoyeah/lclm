import torch

from nanochat.eval import core
from nanochat.eval.core_mt import evaluate_example_mt


class TinyTokenizer:
    def __init__(self):
        self.specials = {
            "<|bos|>": 120,
            "<|sot|>": 121,
            "<|eot|>": 122,
        }
        for k in range(1, 17):
            self.specials[f"<|bot_{k}|>"] = 122 + k
        self.reverse_specials = {v: k for k, v in self.specials.items()}

    def get_vocab_size(self):
        return 160

    def encode_special(self, text):
        return self.specials[text]

    def get_bos_token_id(self):
        return self.specials["<|bos|>"]

    def encode(self, text, prepend=None, append=None, num_threads=8):
        def one(s):
            ids = [ord(ch) for ch in s]
            if prepend is not None:
                ids.insert(0, prepend if isinstance(prepend, int) else self.encode_special(prepend))
            if append is not None:
                ids.append(append if isinstance(append, int) else self.encode_special(append))
            return ids

        if isinstance(text, str):
            return one(text)
        return [one(t) for t in text]

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        out = []
        for tid in ids:
            if tid in self.reverse_specials:
                out.append(self.reverse_specials[tid])
            else:
                out.append(chr(tid))
        return "".join(out)


class TinyModel:
    def __init__(self):
        self.cos = torch.zeros(1, 512, 1, 1)


def test_evaluate_example_mt_pads_rope_idx_with_zero(monkeypatch):
    tokenizer = TinyTokenizer()
    model = TinyModel()
    data = [{"query": "Q", "choices": ["a", "bb"], "gold": 0}]
    task_meta = {
        "task_type": "multiple_choice",
        "num_fewshot": 0,
        "continuation_delimiter": " ",
    }
    captured = {}

    def fake_forward_model(model, input_ids, rope_idx=None):
        captured["input_ids"] = input_ids.detach().cpu()
        captured["rope_idx"] = rope_idx.detach().cpu()
        losses = torch.ones_like(input_ids, dtype=torch.float32)
        predictions = input_ids.clone()
        return losses, predictions

    monkeypatch.setattr(core, "forward_model", fake_forward_model)

    assert evaluate_example_mt(0, model, tokenizer, data, "cpu", task_meta) is True

    input_ids = captured["input_ids"]
    rope_idx = captured["rope_idx"]
    assert input_ids[0, -1].item() == tokenizer.get_bos_token_id()
    assert input_ids[1, -1].item() == ord("b")
    assert rope_idx[0, -1].item() == 0
    assert rope_idx[1, -1].item() != 0


def test_evaluate_task_arange_keeps_multiline_candidates(monkeypatch):
    import nanochat.models.multithread_lm as mt_mod

    class FakeMT:
        pass

    monkeypatch.setattr(mt_mod, "MultithreadLM", FakeMT)

    seen = []

    def fake_evaluate_example(idx, model, tokenizer, data, device, task_meta, mt_encoding="chain"):
        seen.append(idx)
        return True

    monkeypatch.setattr(core, "evaluate_example", fake_evaluate_example)

    data = [
        {"continuation": "ok"},
        {"continuation": "bad\ncontinuation"},
        {"continuation": "also ok"},
    ]
    task_meta = {
        "task_type": "language_modeling",
        "num_fewshot": 0,
        "continuation_delimiter": " ",
    }

    accuracy = core.evaluate_task(
        FakeMT(), TinyTokenizer(), data, "cpu", task_meta, mt_encoding="arange"
    )

    assert accuracy == 1.0
    assert seen == [0, 1, 2]
