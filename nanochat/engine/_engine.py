"""Compatibility shim for the renamed vanilla generation engine."""

from nanochat.engine.vanilla_engine import (
    Engine,
    KVCache,
    RowState,
    sample_next_token,
)
