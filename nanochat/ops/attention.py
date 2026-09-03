"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanochat.ops.attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Window size computation
# =============================================================================
def compute_window_sizes(sequence_len, n_layer, window_pattern):
    """
    Compute per-layer window sizes for sliding window attention.

    Returns list of (left, right) tuples for flash attention's window_size parameter.
    Pattern string is tiled across layers. Final layer always gets L (full context).
    Characters: L=long (full context), S=short (quarter context, ceiled to 128-token tile).
    """
    pattern = window_pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
    long_window = sequence_len
    short_window = -(-long_window // 4 // 128) * 128  # ceil to 128 (safe multiple of FA2/FA3 tile sizes: 32, 64, 128)
    char_to_window = {
        "L": (long_window, 0),
        "S": (short_window, 0),
    }
    window_sizes = []
    for layer_idx in range(n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    window_sizes[-1] = (long_window, 0)
    return window_sizes


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs, FA2 on Ampere/Ada (sm80–sm89)
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


def _load_flash_attention_2():
    """Try to load Flash Attention 2 (Ampere/Ada/Hopper, sm80+).

    Used on hardware where FA3 isn't available (e.g. ada6000 sm89). Requires
    the PyPI `flash-attn` package — install via the `gpu-hpcc` extras with
    `--no-build-isolation` so it compiles against the local torch.
    """
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:  # FA2 needs sm80+ (Ampere or newer)
            return None
        import flash_attn
        if not hasattr(flash_attn, "flash_attn_func"):
            return None
        return flash_attn
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
_fa2 = _load_flash_attention_2()
HAS_FA2 = _fa2 is not None

# Override for testing: set to 'fa3', 'fa2', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl in ('fa2', 'sdpa'):
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False


def _resolve_use_fa2():
    """Decide whether to use FA2 (only when FA3 is not in use)."""
    if _override_impl == 'fa2':
        assert HAS_FA2, "Cannot override to FA2: not available on this hardware"
        return True
    if _override_impl in ('fa3', 'sdpa'):
        return False
    if USE_FA3:
        return False
    if HAS_FA2:
        # FA2 supports bf16 and fp16 (not fp32); fall back to SDPA otherwise
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE in (torch.bfloat16, torch.float16):
            return True
    return False

USE_FA3 = _resolve_use_fa3()
USE_FA2 = _resolve_use_fa2()

import os as _os
if _os.environ.get("RANK", "0") == "0":
    _impl = "FA3" if USE_FA3 else "FA2" if USE_FA2 else "SDPA"
    print(f"[nanochat.ops.attention] HAS_FA3={HAS_FA3} HAS_FA2={HAS_FA2} -> using {_impl}")


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa, causal):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    
    #TODO: this is the left side window, later we also need to consider right side for the full-attn model
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    
    if causal:
        mask = col_idx <= row_idx
    else:
        mask = torch.ones(Tq, Tk, dtype=torch.bool, device=device)

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        #TODO: when full attn, the window_size is double direction or only effect left
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    if USE_FA2:
        return _fa2.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa, causal)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    if USE_FA2:
        return _fa2.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa, causal)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k,
                           block_table=None, softmax_scale=None,
                           causal=False, window_size=(-1, -1)):
    """Varlen flash attention. Packs heterogeneous per-row sequence lengths
    via `cu_seqlens_q` / `cu_seqlens_k` cumulative offsets.

    With `block_table` set, k and v are paged caches of shape
    `(num_blocks, block_size, n_kv_head, head_dim)`; `cu_seqlens_k`
    describes the full per-row K length (prefix + new), and the kernel
    indirects through `block_table[r, page_idx]` to read pages.

    Used for both varlen prefill (no block_table, k/v are flat packed) and
    varlen decode against a paged KV cache (block_table set).

    No SDPA fallback — varlen requires FA2 or FA3.
    """
    if USE_FA3:
        return _fa3.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            block_table=block_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )

    if USE_FA2:
        return _fa2.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=block_table,
        )

    raise NotImplementedError(
        "flash_attn_varlen_func requires FA2 or FA3 (no SDPA fallback in v1)"
    )


# =============================================================================
# Flex attention: arbitrary mask via torch.nn.attention.flex_attention
# =============================================================================
from torch.nn.attention.flex_attention import flex_attention as _flex_attention
from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask

# flex_attention is most efficient under torch.compile; the model is wrapped by
# torch.compile in scripts/base_train.py, which subsumes this.
_flex_attention_compiled = _flex_attention


def create_flex_block_mask(mask_mod, Q_LEN, KV_LEN, device, B=None, H=None):
    """Thin wrapper around `torch.nn.attention.flex_attention.create_block_mask`.

    `mask_mod(b, h, q_idx, kv_idx) -> bool` is the per-position predicate.
    Returns a `BlockMask` to pass to `flex_attn_func`.
    """
    return _create_block_mask(mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device)


def flex_attn_func(q, k, v, block_mask, enable_gqa=False):
    """Flex attention call site, matching the (B, T, H, D) layout used here.

    Internally transposes to flex's (B, H, T, D) layout, dispatches, and
    transposes back. No score_mod (pure block mask).
    """
    # (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y = _flex_attention_compiled(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)
    return y.transpose(1, 2)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
    flash_attn_varlen_func=flash_attn_varlen_func,
)
