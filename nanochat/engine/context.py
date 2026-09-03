"""
Per-call forward-pass context for the batched engine.

Before each model forward, the engine populates a global Context with the
metadata that attention layers need for varlen + paged KV (cu_seqlens,
slot_mapping, block_tables). The model's attention modules read it via
`get_context()` rather than receiving these as forward args — keeps the
model forward signature clean and matches the WeDLM/vLLM convention.

Lifted (and trimmed of wedlm-specific fields) from WeDLM
`utils/context.py`.
"""

from dataclasses import dataclass

import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill: bool,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
) -> None:
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
    )


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = Context()
