"""Per-row diagnostic capture for block-MT generation.

Companion to `MTRowState` in `mt_engine.py`. Holds two kinds of state that
the production decode loop doesn't need but trace dumpers do:

  - `signal_events`: every control-signal sampled at a decision slot
    (parent c_L, K=1-wave decision, DECISION_TAIL). The engine consumes
    these as state-machine inputs (bot_pending → fed to WAVE_START as
    input; bos → terminate) so they never land in `state.tokens`.
    Recording them lets dumpers reconstruct the per-wave K choice for
    visualization.

  - `decision_topk`: pre-mask top-K snapshot of the c_L logit
    distribution at each DECISION_TAIL slot. Populated only when
    `BatchedEngine.generate_multithread` is called with
    `collect_decision_logits > 0`. Useful for analyzing what the model
    "wanted" to emit at the masked positions vs what the mask forced.

Recording helpers short-circuit cheaply when `trace_log is None`, so
production callers that don't attach a log pay no cost in the hot loop.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class MTTraceLog:
    # {wave_step, after_paragraph_idx, signal_id, K_next}; K_next None ⇔ bos.
    signal_events: list[dict] = field(default_factory=list)
    # {wave_step, ids: [int*K], logits: [float*K], probs: [float*K]}; full-vocab
    # softmax marginals (not local-renormalized).
    decision_topk: list[dict] = field(default_factory=list)


def record_signal(trace_log: MTTraceLog | None, wave_step: int,
                  after_paragraph_idx: int, signal_id: int,
                  K_next: int | None) -> None:
    if trace_log is None:
        return
    trace_log.signal_events.append({
        "wave_step": int(wave_step),
        "after_paragraph_idx": int(after_paragraph_idx),
        "signal_id": int(signal_id),
        "K_next": int(K_next) if K_next is not None else None,
    })


def record_decision_topk(trace_log: MTTraceLog | None, wave_step: int,
                         logits_row: torch.Tensor, top_k: int) -> None:
    """Snapshot top-K from a (V,) logit vector. No-op if trace_log is None
    or top_k <= 0."""
    if trace_log is None or top_k <= 0:
        return
    K_topk = min(top_k, logits_row.size(-1))
    probs = torch.softmax(logits_row.float(), dim=-1)
    vals, ids = torch.topk(probs, k=K_topk)
    trace_log.decision_topk.append({
        "wave_step": int(wave_step),
        "ids": ids.tolist(),
        "probs": vals.tolist(),
        "logits": logits_row[ids].float().tolist(),
    })
