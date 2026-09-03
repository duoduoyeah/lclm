"""
Multithread-LM — clean fork of simple_gpt.py for the multithread row layout.

Key differences vs simple_gpt:
- Pure-causal attention via FA3 `flash_attn_func(causal=True)`. Sibling
  tokens within a wave-step see each other only via row-order causal
  (ordered, not mutual). The multithread row layout (decision-last held
  to end of wave; see `multithread_layout.py`) is what gives the decision
  token full sibling context structurally.
- forward signature is `forward(idx, targets, rope_idx)`. RoPE is consumed
  via `rope_idx` (B, T) — cos/sin are gathered by index rather than sliced
  by sequence position. The dataloader has already baked per-doc rope
  offsets into `rope_idx` at pack time.
- `rotary_seq_len = sequence_len * 40`. Multithread RoPE is staggered with
  stride=256 + uniform paragraph alignment, so rope values can comfortably
  exceed `sequence_len`. The 40x headroom absorbs the worst-case packed
  row (cheap memory cost — cos/sin tables are small).
- Decode path uses `varlen_attn_with_paged_cache` (`causal=True`) — same
  KV-cache plumbing as the rest of the engine.
- Only the standard "rope" variant is supported. The 3D-RPE variants
  defined in `rope_3d.py` are vanilla-LM style and don't compose with
  per-position rope-idx gather without further design.

Otherwise mirrors simple_gpt: QK norm, untied wte/lm_head, relu^2 MLP,
GQA, per-layer resid_lambdas / x0_lambdas, the full optimizer setup.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.train.optim import MuonAdamW, DistMuonAdamW

from nanochat.engine.cache import varlen_attn_with_paged_cache
from nanochat.ops.attention import flash_attn
from nanochat.ops.norm import norm
from nanochat.ops.rope import apply_rotary_emb, precompute_rotary_embeddings


@dataclass
class MultithreadLMConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # RoPE cache headroom factor. Multithread rope values can exceed
    # sequence_len because of per-paragraph stride + per-doc offset baked
    # at pack time. The absolute worst case per row is `T * rope_stride`
    # (every doc 1 token, each advancing rope_cursor by rope_stride),
    # but real packings reach a small fraction of that. 128x gives 2x
    # headroom over the old 64x default and comfortably covers typical
    # packings; the multithread_dataloader installs a python-side guard
    # (skip-and-warn) as a second safety net for the rare pathological
    # case that exceeds the cache. Was 64 prior to 2026-05-13; bumped
    # to 128 after the d12 T=1024 run hit `< 65536` device-side asserts.
    # At T=1024 this gives 131072 rope slots, cos/sin ~16 MB each at
    # bf16 — trivial vs the model's GB-scale weights.
    rotary_seq_len_factor: int = 128
    # RoPE base frequency. Default 100000 matches the production v3 from-
    # scratch path. The d34→MT continued-pretrain line (see
    # design/posttrain/d34-rope-postpretrain.md) overrides to 10000 to preserve
    # d34's pretrained rope behavior at surgery init.
    rope_base: int = 100000
    # Post-QK-norm scalar applied to q, k. Default 1.2 matches the v3 from-
    # scratch architecture. d34-warmstart sets this to 1.0 (LegacyGPT has no
    # such scaling); pre-dividing c_q/c_k weights cannot compensate because
    # QK norm is scale-invariant and absorbs the pre-divide. See
    # design/posttrain/d34-rope-postpretrain.md T7.
    qk_scale: float = 1.2


def _build_rope_cos_sin(config, seq_len, head_dim, device):
    return precompute_rotary_embeddings(seq_len, head_dim, device, base=config.rope_base)


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class MultithreadAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.qk_scale = config.qk_scale
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # Placeholders for paged KV cache slices. PagedKVCache.attach() rewires
        # these to per-layer views of the shared pool when the engine is set
        # up; they stay empty during training.
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, x, cos_sin, cache_ctx=None):
        if cache_ctx is not None:
            return self._forward_decode(x, cos_sin, cache_ctx)
        return self._forward_train(x, cos_sin)

    def _forward_train(self, x, cos_sin):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q = q * self.qk_scale
        k = k * self.qk_scale

        # Sibling tokens within a wave-step see each other only via row-order
        # causal (ordered, not mutual). The decision-last layout in
        # multithread_layout.compile_doc gives the decision token full sibling
        # context structurally.
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

    def _forward_decode(self, x, cos_sin, cache_ctx):
        # Flat (N, C) input — varlen + paged-KV decode path. Mirrors _forward_train
        # but skips the (B, T) batched view and dispatches to the store-then-attend
        # wrapper, which scatters new K/V into the paged cache before attending.
        N, C = x.size()
        q = self.c_q(x).view(N, self.n_head, self.head_dim)
        k = self.c_k(x).view(N, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(N, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * self.qk_scale
        k = k * self.qk_scale

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
        self.attn = MultithreadAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, cache_ctx=None):
        x = x + self.attn(norm(x), cos_sin, cache_ctx)
        x = x + self.mlp(norm(x))
        return x


class MultithreadLM(nn.Module):
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
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.rotary_seq_len = config.sequence_len * config.rotary_seq_len_factor
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

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
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

    def forward(self, idx, targets=None, rope_idx=None,
                cache_ctx=None, loss_reduction='mean'):
        # Two input modes:
        #   training/eval — idx, rope_idx are (B, T); cache_ctx is None.
        #   decode        — idx, rope_idx are (N,) flat-packed; cache_ctx is set.
        if rope_idx is None:
            # Forward-only inference: treat as single-doc with positions 0..T-1.
            assert cache_ctx is None, "rope_idx is required in decode (cache_ctx) mode"
            B, T = idx.shape
            rope_idx = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        assert rope_idx.shape == idx.shape
        if cache_ctx is not None:
            assert idx.dim() == 1, "cache_ctx (decode) requires flat (N,) input"
            assert targets is None, "cache_ctx (decode) does not compute loss"
        else:
            assert idx.dim() == 2, "training/eval mode requires (B, T) input"
        assert idx.device == self.cos.device, f"idx and cos device mismatch: {idx.device} vs {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE

        # Gather cos/sin by rope_idx. self.cos is (1, rotary_seq_len, 1, head_dim/2);
        # rope_idx is (B, T) for training or (N,) for decode; gather yields
        # (B, T, 1, head_dim/2) or (N, 1, head_dim/2) respectively, which
        # apply_rotary_emb broadcasts over the head dim.
        cos = self.cos[0, rope_idx]
        sin = self.sin[0, rope_idx]
        cos_sin = (cos, sin)

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin, cache_ctx)
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
