"""
Simple GPT — clean fork of gpt.py with experimental features stripped.

Removed (vs gpt.py):
- smear (previous-token embedding mixing)
- backout (mid-layer residual subtraction)
- sliding window attention (window_pattern); always full causal
- value embeddings (ResFormer)

Kept:
- rotary embeddings (rope + rope3d_* variants)
- QK norm
- untied wte / lm_head
- relu^2 MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA)
- Flash Attention 3 / SDPA fallback
- per-layer resid_lambdas / x0_lambdas residual scaling

This is the reusable base for small architectural experiments.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.train.optim import MuonAdamW, DistMuonAdamW

from nanochat.engine.cache import varlen_attn_with_paged_cache
from nanochat.ops.attention import flash_attn
from nanochat.ops.norm import norm
from nanochat.ops.rope import apply_rotary_emb, precompute_rotary_embeddings
from nanochat.ops.rope_3d import precompute_3d_rotary_embeddings


@dataclass
class SimpleGPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Rotary embedding variant. "rope" = standard RoPE; rope3d_* = 3D-RPE
    # variants (see nanochat/ops/rope_3d.py).
    rope_variant: str = "rope"
    rope_chunk_size: int = 128
    rope_base_chunk: Optional[float] = None  # rope3d_paper / rope3d_paper_rev
    rope_theta_phi: float = 1.0              # rope3d_linear


def _build_rope_cos_sin(config, seq_len, head_dim, device):
    variant = config.rope_variant
    if variant == "rope":
        return precompute_rotary_embeddings(seq_len, head_dim, device)
    if variant == "rope3d_paper":
        return precompute_3d_rotary_embeddings(
            seq_len, head_dim, config.rope_chunk_size, device,
            variant="paper", base_chunk=config.rope_base_chunk,
        )
    if variant == "rope3d_paper_rev":
        return precompute_3d_rotary_embeddings(
            seq_len, head_dim, config.rope_chunk_size, device,
            variant="paper_rev", base_chunk=config.rope_base_chunk,
        )
    if variant == "rope3d_linear":
        return precompute_3d_rotary_embeddings(
            seq_len, head_dim, config.rope_chunk_size, device,
            variant="linear", theta_phi=config.rope_theta_phi,
        )
    if variant == "rope3d_dual_rope":
        return precompute_3d_rotary_embeddings(
            seq_len, head_dim, config.rope_chunk_size, device,
            variant="dual_rope", base_chunk=config.rope_base_chunk,
        )
    raise ValueError(f"unknown rope_variant: {variant!r}")


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


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
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # Placeholders for the new BatchedEngine's paged KV cache; rewired
        # by `nanochat.engine.cache.PagedKVCache.attach` at engine setup.
        # Stay empty during training and during the legacy FA3 KVCache path.
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, x, cos_sin, kv_cache, cache_ctx=None):
        if cache_ctx is not None:
            return self._forward_decode(x, cos_sin, cache_ctx)
        return self._forward_train(x, cos_sin, kv_cache)

    def _forward_train(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K)
        k = k * 1.2

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=(-1, 0),
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

    def _forward_decode(self, x, cos_sin, cache_ctx):
        # Flat (N, C) varlen + paged-KV decode path used by the new
        # BatchedEngine. Mirrors `_forward_train` but skips the (B, T) view
        # and dispatches to the store-then-attend wrapper, which scatters
        # new K/V into the paged cache before attending.
        N, C = x.size()
        q = self.c_q(x).view(N, self.n_head, self.head_dim)
        k = self.c_k(x).view(N, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(N, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        y = varlen_attn_with_paged_cache(
            q, k, v,
            self.k_cache, self.v_cache,
            slot_mapping=cache_ctx.slot_mapping,
            cu_seqlens_q=cache_ctx.cu_seqlens_q,
            cu_seqlens_k=cache_ctx.cu_seqlens_k,
            max_seqlen_q=cache_ctx.max_seqlen_q,
            max_seqlen_k=cache_ctx.max_seqlen_k,
            block_table=cache_ctx.block_tables,
            causal=True,
            window_size=(-1, 0),
        )
        y = y.contiguous().view(N, -1)
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

    def forward(self, x, cos_sin, kv_cache, cache_ctx=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache, cache_ctx)
        x = x + self.mlp(norm(x))
        return x


class SimpleGPT(nn.Module):
    ATTENTION_BACKEND = "flash_attn"

    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE major footgun: this __init__ runs in meta device context.
        Any calculations here are shapes/dtypes only, no actual data.
        => Real init happens in init_weights().
        """
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
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Over-compute rotary cache by 10X; cheap.
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = _build_rope_cos_sin(config, self.rotary_seq_len, head_dim, device=self.transformer.wte.weight.device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = _build_rope_cos_sin(self.config, self.rotary_seq_len, head_dim, device=self.transformer.wte.weight.device)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. fp16 requires fp32 because GradScaler
        # cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        FLOPs per token (forward + backward).
        Each matmul weight contributes 6 FLOPs (2 forward + 4 backward).
        Plus 12 * h * q * seq_len for attention QK matmul.
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = (self.transformer.wte.weight.numel() +
                           self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = self.config.n_layer * 12 * h * q * t
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params)

        # Scale the LR for AdamW params by ∝1/√dmodel (tuned for 768 dim).
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, rope_idx=None, kv_cache=None,
                cache_ctx=None, loss_reduction='mean'):
        # Three input modes:
        #   training/eval — idx is (B, T); rope_idx and cache_ctx are None.
        #   legacy decode — idx is (B, T); kv_cache is the FA3 contiguous KVCache.
        #   batched decode — idx and rope_idx are flat (N,); cache_ctx is set
        #                    (new BatchedEngine, paged-KV + varlen).
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"

        if cache_ctx is not None:
            assert idx.dim() == 1, "cache_ctx (batched decode) requires flat (N,) input"
            assert rope_idx is not None and rope_idx.shape == idx.shape, \
                "cache_ctx (batched decode) requires rope_idx of the same shape as idx"
            assert kv_cache is None, "cache_ctx and kv_cache are mutually exclusive"
            assert targets is None, "cache_ctx (batched decode) does not compute loss"
            assert self.config.rope_variant == "rope", (
                "BatchedEngine path supports rope_variant='rope' only; "
                f"got {self.config.rope_variant!r}. The 3D-RPE variants stash "
                "cos/sin in a different shape and need their own gather."
            )
            # Gather cos/sin by absolute rope index. self.cos shape is
            # (1, rotary_seq_len, 1, head_dim/2); rope_idx shape (N,) →
            # (N, 1, head_dim/2), which apply_rotary_emb broadcasts over n_head.
            cos = self.cos[0, rope_idx]
            sin = self.sin[0, rope_idx]
            cos_sin = (cos, sin)
        else:
            B, T = idx.size()
            assert T <= self.cos.size(1), f"Sequence length grew beyond rotary embeddings cache: {T} > {self.cos.size(1)}"
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        x0 = x  # initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin, kv_cache, cache_ctx)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference. Assumes batch size 1; tokens
        and yielded values are simple Python lists / ints.
        """
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
            token = next_ids.item()
            yield token
