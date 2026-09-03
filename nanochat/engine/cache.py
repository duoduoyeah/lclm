"""
Paged KV cache machinery for batched varlen decode.

Three things live here:
  - Triton `store_kvcache` kernel — scatter per-token K, V into cache slots.
    Lifted from WeDLM (`wedlm/layers/attention.py`, Tencent, Apache 2.0).
  - `PagedKVCache` + `BlockManager` — tensor pool + free/used block tracker.
  - `varlen_attn_with_paged_cache` — store-then-attend wrapper (the only
    caller of the kernel + the varlen attention primitive).

The model's attention layer calls `varlen_attn_with_paged_cache(...)` in
its decode dispatch arm; the engine pre-populates `Context` (see
`engine/context.py`) and allocates blocks via `BlockManager`.
"""

from collections import deque

import torch

from nanochat.ops.attention import flash_attn_varlen_func


# triton is required for the kernel below; on CPU-only setups (no GPU,
# no triton install) we still want the rest of this module to import
# successfully — model classes that only use the meta-device path for
# param counting / shape inspection don't actually call into triton.
# Eager imports of `triton` / `triton.language` failed module load and
# blocked things like `scripts/param_count.py`. We defer until needed.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
        D_POW2: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        offs = tl.arange(0, D_POW2)
        mask = offs < D
        key_offsets = idx * key_stride + offs
        value_offsets = idx * value_stride + offs
        key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
        value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
        cache_offsets = slot * D + offs
        tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
        tl.store(v_cache_ptr + cache_offsets, value, mask=mask)
else:
    store_kvcache_kernel = None


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """Scatter per-token K, V into the paged cache.

    key, value:        (N, n_kv_head, head_dim)
    k_cache, v_cache:  (num_blocks, block_size, n_kv_head, head_dim) — viewed
                       flat as (num_blocks * block_size, n_kv_head, head_dim).
    slot_mapping:      (N,) int32. slot_mapping[i] is the absolute flat slot
                       for token i. Pass -1 to skip (padding).
    """
    if not _HAS_TRITON:
        raise RuntimeError(
            "store_kvcache requires triton, which is not installed. "
            "Engine paged-KV decode is unavailable on this machine."
        )
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
        _next_pow2(D),
    )


class PagedKVCache:
    """Pool of paged KV blocks shared across all attention layers.

    Layout: a single contiguous tensor of shape
        (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
    where dim 0 selects K vs V. `attach(model)` wires each attention
    module's `k_cache` / `v_cache` buffers to its layer's slice
    `pool[0, layer_idx]` / `pool[1, layer_idx]`, both of shape
        (num_blocks, block_size, num_kv_heads, head_dim)
    — the FA2 paged-KV expected shape.

    Each layer's cache shares the same block-id space; allocation lives
    in `BlockManager`.
    """

    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads,
                 head_dim, device, dtype):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.pool = torch.empty(
            2, num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            device=device, dtype=dtype,
        )

    def attach(self, model: torch.nn.Module) -> None:
        """Wire the model's attention modules to per-layer slices of the pool.

        Iterates `model.modules()` in registration order; any module with
        `k_cache` and `v_cache` attributes gets the next layer's slice.
        Asserts the count matches `num_layers` so a wiring miss surfaces
        immediately.
        """
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.pool[0, layer_id]
                module.v_cache = self.pool[1, layer_id]
                layer_id += 1
        assert layer_id == self.num_layers, (
            f"PagedKVCache.attach: model has {layer_id} k_cache/v_cache "
            f"modules, expected {self.num_layers}"
        )


class BlockManager:
    """Free/used block pool for the paged KV cache.

    Simpler than vLLM/WeDLM's full manager — no hash-based prefix cache,
    no reference counting. Just a deque of free block ids. Suitable for
    v1 where rows are independent and blocks aren't shared.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free: deque[int] = deque(range(num_blocks))
        self.used: set[int] = set()

    def num_free(self) -> int:
        return len(self.free)

    def allocate(self, n: int) -> list[int]:
        """Pop `n` ids off the free pool. Raises if not enough are free."""
        if len(self.free) < n:
            raise RuntimeError(
                f"BlockManager: requested {n} blocks, only {len(self.free)} free"
            )
        ids = [self.free.popleft() for _ in range(n)]
        self.used.update(ids)
        return ids

    def release(self, ids: list[int]) -> None:
        """Return blocks to the free pool."""
        for bid in ids:
            self.used.discard(bid)
            self.free.append(bid)


def varlen_attn_with_paged_cache(
    q, k, v,
    k_cache, v_cache,
    slot_mapping,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    block_table,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
):
    """Store new K, V into the paged cache, then varlen-attend over the full cache.

    The two-step that FA2's varlen API leaves to the caller — the kernel
    doesn't insert into the cache itself. After the scatter, every k_cache
    /v_cache slot named by `slot_mapping` holds this step's new K/V; the
    subsequent varlen call reads the full per-row prefix + new range via
    `cu_seqlens_k` and `block_table`.

    Shapes:
        q                    (sum_T_new, n_head, head_dim)
        k, v                 (sum_T_new, n_kv_head, head_dim)
        k_cache, v_cache     (num_blocks, block_size, n_kv_head, head_dim)
        slot_mapping         (sum_T_new,) int32, absolute slot per token
        cu_seqlens_q         (B+1,) int32 — cumulative new-token offsets
        cu_seqlens_k         (B+1,) int32 — cumulative full-K offsets (prefix + new)
        block_table          (B, max_blocks_per_seq) int32

    Returns:
        (sum_T_new, n_head, head_dim) attention output.
    """
    store_kvcache(k, v, k_cache, v_cache, slot_mapping)
    return flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )
