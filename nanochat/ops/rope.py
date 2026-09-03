import torch

from nanochat.common import COMPUTE_DTYPE


def precompute_rotary_embeddings(seq_len, head_dim, device, base=100000):
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    # Works for both 4D (B, T, n_head, head_dim) — training/batched —
    # and 3D (N, n_head, head_dim) — flat varlen decode. cos/sin broadcast
    # over the head dim either way.
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], -1)
