"""
Attention mask builder for ILM's [noisy | clean] 2L layout.

Quadrants (True = attend, SDPA-with-bool convention):
  UL (noisy q, noisy k) — within-block bidir:          block_q == block_kv
  UR (noisy q, clean k) — offset block-causal, diag masked:  block_q > block_kv
  LL (clean q, noisy k) — blocked:                      all False
  LR (clean q, clean k) — block-causal:                 block_q >= block_kv

Under scheme B (batch-uniform noisy length, per-block survival varies),
block_ids differ per sample; output shape is (B, 1, T_total, T_total).
"""

import torch
from torch import Tensor


def _ul(noisy_block_ids: Tensor) -> Tensor:
    return noisy_block_ids.unsqueeze(-1) == noisy_block_ids.unsqueeze(-2)


def _ur(noisy_block_ids: Tensor, clean_block_ids: Tensor) -> Tensor:
    return noisy_block_ids.unsqueeze(-1) > clean_block_ids.unsqueeze(-2)


def _ll(B: int, T_clean: int, T_noisy: int, device) -> Tensor:
    return torch.zeros(B, T_clean, T_noisy, device=device, dtype=torch.bool)


def _lr(clean_block_ids: Tensor) -> Tensor:
    return clean_block_ids.unsqueeze(-1) >= clean_block_ids.unsqueeze(-2)


def _stitch(ul: Tensor, ur: Tensor, ll: Tensor, lr: Tensor) -> Tensor:
    top = torch.cat([ul, ur], dim=-1)
    bottom = torch.cat([ll, lr], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def build_block_attn_mask(
    noisy_block_ids: Tensor,   # (B, T_noisy) long
    clean_block_ids: Tensor,   # (B, T_clean) long
) -> Tensor:
    """Return (B, 1, T_total, T_total) bool mask for the block variant."""
    assert noisy_block_ids.dim() == 2 and clean_block_ids.dim() == 2
    assert noisy_block_ids.size(0) == clean_block_ids.size(0)

    B = noisy_block_ids.size(0)
    T_n = noisy_block_ids.size(1)
    T_c = clean_block_ids.size(1)
    device = noisy_block_ids.device

    ul = _ul(noisy_block_ids)                      # (B, T_n, T_n)
    ur = _ur(noisy_block_ids, clean_block_ids)     # (B, T_n, T_c)
    ll = _ll(B, T_c, T_n, device)                  # (B, T_c, T_n)
    lr = _lr(clean_block_ids)                      # (B, T_c, T_c)

    mask = _stitch(ul, ur, ll, lr)                 # (B, T, T)
    return mask.unsqueeze(1)                       # (B, 1, T, T)
