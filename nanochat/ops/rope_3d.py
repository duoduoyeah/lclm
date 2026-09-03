"""
3D Rotary Position Encoding (3D-RPE).

Standard RoPE rotation + a per-chunk additive phase offset that shifts all
head_dim/2 channel-pair angles uniformly. Reuses `apply_rotary_emb` from
rope.py unchanged.

Four variants are supported:

  variant="paper"   (original 3D-RPE paper, Eq. 4/5; Bloch sphere)
      angle(p, l) = m · θ_l + (π/2 − base_chunk^{−j})
      Per-pair RoPE on within-chunk index m; uniform additive offset on j.
      φ_j = base_chunk^{−j} saturates to ~0 fast (φ_1 ≈ 1e-5 for
      base_chunk = 100000), so chunks ≥ 1 collapse to a near-identical π/2
      offset and become mutually indistinguishable at the chunk level.
      Asymmetric: translation-invariant in m, but j gives an absolute-position
      bias.

  variant="linear"  (symmetric "RoPE on a second axis")
      angle(p, l) = m · θ_l + j · theta_phi
      Per-pair RoPE on m; uniform linear offset on j with a single scalar
      chunk frequency. Translation-invariant in both m and j (relative-only).

  variant="paper_rev"   (paper variant with m and j swapped)
      angle(p, l) = j · θ_l + (π/2 − base_chunk^{−m})
      Per-pair RoPE on chunk index j; uniform additive offset on m.
      Mirror image of "paper": within-chunk relative position is now encoded
      only by the asymmetric scalar offset on m (which itself saturates after
      m=1 with base_chunk = 100000), while cross-chunk relative position
      gets the per-pair RoPE structure.

  variant="dual_rope"   (RoPE on both axes with separate frequency bases)
      angle(p, l) = m · θ_l + j · θ_φ_l
      θ_φ_l = base_chunk^{−(2l+1)/d}    (note the +1 exponent shift)
      Per-pair RoPE on m AND per-pair RoPE on j, with a *different* base on
      each axis. Translation-invariant in both m and j; multi-frequency on
      both.

      Two design choices keep `θ_l ≠ θ_φ_l` at every pair l so the slope-1
      diagonal collapse (Δm=1,Δj=0) ≡ (Δm=0,Δj=1) is avoided:
        (a) base ≠ base_chunk, and
        (b) the +1 exponent shift on θ_φ_l (without it, l=0 would still
            give θ_0 = θ_φ_0 = 1 regardless of base, leaving a residual
            collapse on the dominant low-frequency pair).

      For the diagonal to actually be distinguishable on the *full* 64-pair
      vector (not just on l=0), the differences θ_l − θ_φ_l must be non-
      tiny across many l. With base ≫ base_chunk both with the same shape,
      the differences stay small — you generally want base_chunk to be
      well below base (e.g. base=100000, base_chunk=10..100), and chunk
      count ≪ within-chunk count anyway makes this natural.

Common notation:
    j = p // c                 (chunk index)
    m = p % c                  (within-chunk position)
    θ_l = base^{−2l/head_dim}  (per-pair RoPE frequencies)
    c = chunk_size

Implementation
--------------
All four variants decompose as `angle(m, j, l) = m_part(m, l) + j_part(j, l)`,
so the math lives in one helper per variant (`_angle_parts_<variant>`); each
returns `(m_part, j_part)` with shapes `(N_m, head_dim/2)` and
`(N_j, head_dim/2)` (uniform-across-l components are broadcast to per-pair
shape via `expand`, which is a view — no copy).

The public entry point, `precompute_3d_rotary_embeddings`, uses fixed
positions: m and j are both length-`seq_len` (paired by `pos % c` /
`pos // c`), so `freqs = m_part + j_part` (elementwise, shape
`(seq_len, half)`). It is used by `gpt` and `simple_gpt`.
"""

import torch

from nanochat.common import COMPUTE_DTYPE


def _inv_freq(head_dim, base, device):
    """Per-pair RoPE frequencies θ_l = base^(−2l/head_dim), shape (head_dim/2,)."""
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (channel_range / head_dim))


def _expand_uniform(part_1d, half):
    """Promote a (N,)-shape uniform-across-l offset to a (N, half) view via `expand`."""
    return part_1d[:, None].expand(-1, half)


def _angle_parts_paper(m_vals, j_vals, head_dim, base, base_chunk):
    """paper:  angle = m·θ_l + (π/2 − base_chunk^{−j})

    Per-pair RoPE on m; uniform additive offset on j. Returns
    `(m_part, j_part)` both of shape `(N_m, half)` / `(N_j, half)`.
    """
    half = head_dim // 2
    if base_chunk is None:
        base_chunk = base
    m_part = torch.outer(m_vals, _inv_freq(head_dim, base, m_vals.device))   # (N_m, half)
    j_uniform = (torch.pi / 2) - base_chunk ** (-j_vals)                     # (N_j,)
    j_part = _expand_uniform(j_uniform, half)                                # (N_j, half)
    return m_part, j_part


def _angle_parts_linear(m_vals, j_vals, head_dim, base, theta_phi):
    """linear:  angle = m·θ_l + j·theta_phi

    Per-pair RoPE on m; uniform linear offset on j with a single scalar
    chunk frequency.
    """
    half = head_dim // 2
    m_part = torch.outer(m_vals, _inv_freq(head_dim, base, m_vals.device))   # (N_m, half)
    j_uniform = j_vals * theta_phi                                           # (N_j,)
    j_part = _expand_uniform(j_uniform, half)                                # (N_j, half)
    return m_part, j_part


def _angle_parts_paper_rev(m_vals, j_vals, head_dim, base, base_chunk):
    """paper_rev:  angle = j·θ_l + (π/2 − base_chunk^{−m})

    Per-pair RoPE on j; uniform additive offset on m. Mirror of `paper`.
    """
    half = head_dim // 2
    if base_chunk is None:
        base_chunk = base
    j_part = torch.outer(j_vals, _inv_freq(head_dim, base, j_vals.device))   # (N_j, half)
    m_uniform = (torch.pi / 2) - base_chunk ** (-m_vals)                     # (N_m,)
    m_part = _expand_uniform(m_uniform, half)                                # (N_m, half)
    return m_part, j_part


def _angle_parts_dual_rope(m_vals, j_vals, head_dim, base, base_chunk):
    """dual_rope:  angle = m·θ_l + j·θ_φ_l   with θ_φ_l = base_chunk^{−(2l+1)/d}

    Per-pair RoPE on BOTH axes; the +1 exponent shift on θ_φ_l ensures
    θ_l ≠ θ_φ_l even at l=0.
    """
    if base_chunk is None:
        base_chunk = base
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=m_vals.device)
    inv_freq      = 1.0 / (base       ** (channel_range / head_dim))         # (half,)
    inv_freq_chunk = 1.0 / (base_chunk ** ((channel_range + 1) / head_dim))  # (half,)
    m_part = torch.outer(m_vals, inv_freq)                                   # (N_m, half)
    j_part = torch.outer(j_vals, inv_freq_chunk)                             # (N_j, half)
    return m_part, j_part


def _angle_parts(variant, m_vals, j_vals, head_dim, base, base_chunk, theta_phi):
    """Dispatch to the per-variant kernel. Returns `(m_part, j_part)`."""
    if variant == "paper":
        return _angle_parts_paper(m_vals, j_vals, head_dim, base, base_chunk)
    if variant == "linear":
        return _angle_parts_linear(m_vals, j_vals, head_dim, base, theta_phi)
    if variant == "paper_rev":
        return _angle_parts_paper_rev(m_vals, j_vals, head_dim, base, base_chunk)
    if variant == "dual_rope":
        return _angle_parts_dual_rope(m_vals, j_vals, head_dim, base, base_chunk)
    raise ValueError(f"unknown variant: {variant!r}; expected 'paper', 'linear', 'paper_rev', or 'dual_rope'")


def precompute_3d_rotary_embeddings(
    seq_len, head_dim, chunk_size, device,
    base=100000,
    variant="paper",
    base_chunk=None,
    theta_phi=1.0,
):
    """Fixed-position 3D-RPE: `j = pos // c`, `m = pos % c`, paired."""
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    m_vals = pos % chunk_size
    j_vals = pos // chunk_size
    m_part, j_part = _angle_parts(variant, m_vals, j_vals, head_dim,
                                   base, base_chunk, theta_phi)
    freqs = m_part + j_part   # both (seq_len, head_dim/2), elementwise pair
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]   # (1, seq_len, 1, head_dim/2)
    return cos, sin
