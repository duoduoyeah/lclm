"""
LegacyGPT -- the GPT architecture used by karpathy/nanochat-d34.

This module is the 2c4473d snapshot of nanochat/gpt.py with four mechanical
adaptations from design/paper/open-source-nanochat.md:

  1. Inline norm/apply_rotary_emb replaced with imports from
     nanochat.ops.{norm, rope}.
  2. Old nanochat.muon / nanochat.adamw imports removed.
     setup_optimizer is a stub -- LegacyGPT is load-only.
  3. CausalSelfAttention.forward rewritten to use the current flash_attn
     API (nanochat.ops.attention). Kernel swap, not parameter swap;
     state_dict keys are unchanged.
  4. Top-level classes renamed GPT* -> LegacyGPT*.

The parameter tree (state_dict keys) is bit-for-bit compatible with the
d34 checkpoint at karpathy/nanochat-d34. Predates aa530cd (resid_lambdas,
x0_lambdas), so none of those keys exist.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.models.gpt import Linear  # dtype-casting nn.Linear subclass; needed for bf16 inference over fp32 master weights (d34 checkpoint is fp32)
from nanochat.ops.attention import flash_attn
from nanochat.ops.norm import norm
from nanochat.ops.rope import apply_rotary_emb, precompute_rotary_embeddings


@dataclass
class LegacyGPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 65536
    n_layer: int = 34
    n_head: int = 17
    n_kv_head: int = 17
    n_embd: int = 2176


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Deliberate departure from 2c4473d's plain nn.Linear: the dtype-casting
        # subclass is needed because d34's checkpoint is fp32 but activations
        # are bf16 (COMPUTE_DTYPE). State_dict keys are identical either way.
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # (B, T, H, D) -- flash_attn's native layout, no transpose needed.
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        # d34 has no sliding-window pattern: full causal context everywhere.
        window_size = (-1, -1)

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class LegacyGPT(nn.Module):
    ATTENTION_BACKEND = "flash_attn"

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        # base=10000 matches the 2c4473d snapshot; current ops default to 100000.
        cos, sin = precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim,
            device=self.transformer.wte.weight.device, base=10000,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Matches the 2c4473d snapshot init recipe (no resid/x0/ve/smear)."""
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim,
            device=self.transformer.wte.weight.device, base=10000,
        )
        self.cos, self.sin = cos, sin

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def setup_optimizer(self, *args, **kwargs):
        raise NotImplementedError(
            "LegacyGPT is load-only; train against nanochat.models.gpt instead."
        )

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                rope_idx=None):
        B, T = idx.size()
        assert idx.device == self.cos.device, f"idx on {idx.device}, rotary on {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"rotary dtype {self.cos.dtype} != COMPUTE_DTYPE {COMPUTE_DTYPE}"
        if rope_idx is None:
            # Position-sliced cos/sin (default path; preserves existing eyeball-baseline behavior).
            assert T <= self.cos.size(1), f"Sequence length {T} > rotary cache {self.cos.size(1)}"
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            cos_sin = self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T]
        else:
            # Gather path for rope-perturbation eval. Mirrors multithread_lm.py:311.
            assert kv_cache is None, "rope_idx + kv_cache not supported"
            assert rope_idx.shape == idx.shape, f"rope_idx {tuple(rope_idx.shape)} != idx {tuple(idx.shape)}"
            cos = self.cos[0, rope_idx]   # (B, T, 1, head_dim/2)
            sin = self.sin[0, rope_idx]
            cos_sin = (cos, sin)

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Naive single-sequence autoregressive sampler (kept for parity with
        the 2c4473d snapshot). BatchedEngine.generate_ar is the production
        path; this one only sees use if someone instantiates LegacyGPT in a
        notebook."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()
