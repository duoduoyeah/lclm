"""Per-K breakdown of MT block-mt val loss.

Like `scripts/eval_blockmt_val_loss.py`, but segments per-position
cross-entropy and bytes by **which K-wave each position belongs to**.
The K of a position is recovered by scanning the input stream:

  - `<bos>` → K=1 (parent anchor of a new doc)
  - `<bot-K>` → K=K_value (opener of a new K-wide wave)
  - otherwise → K stays the same as the most recent opener/anchor

Outputs per-K aggregates and a global aggregate (which should match
`evaluate_bpb` exactly on the same loader / steps):

  - bpb_K, ce_per_token_K, ce_per_non_special_K, n_*_K, total_bytes_K

Companion to `scripts/eval/measure_mt_tokens_per_step.py` — together they
give you (quality, throughput) per K, the load-bearing axis for the
quality-vs-parallelism Pareto.

Usage:
    NANOCHAT_DATASET=climbmix_split \\
    NANOCHAT_TOKENIZER=nanochat_64k_mt2 \\
        python -m scripts.eval.blockmt_val_loss_per_K \\
            --model-tag d24_r20_climbmix_blockmt_v3 \\
            --step 14800 \\
            --device-batch-size 16 \\
            --eval-tokens 5242880
"""
import argparse
import json
import math
import os

import torch
import torch.distributed as dist

from nanochat.common import autodetect_device_type, compute_init, get_base_dir, print0
from nanochat.train.checkpoint import load_model_from_dir
from nanochat.tokenizer import get_token_bytes
from nanochat.data.multithread_dataloader import (
    tokenizing_distributed_data_loader_multithread,
)
from nanochat.eval.evaluate_bpb_sliced import evaluate_bpb_sliced
from nanochat.eval.mt_v4_diagnostics import (
    tokenizing_distributed_data_loader_multithread_v4_slices,
)


@torch.no_grad()
def evaluate_per_K(model, loader, steps, token_bytes, bos_id, bot_K_ids, kmax,
                   device):
    """Per-position eval, bucket by K of the containing wave.

    Returns dict: K -> {nats_all, nats_non_special, bytes, n_valid, n_non_special}.
    K=0 is reserved as the "global" aggregate.
    """
    # Buckets indexed by K (0 = global aggregate, 1..kmax = per-K).
    n_buckets = kmax + 1
    nats_all = torch.zeros(n_buckets, dtype=torch.float64, device=device)
    nats_ns = torch.zeros(n_buckets, dtype=torch.float64, device=device)
    bytes_ = torch.zeros(n_buckets, dtype=torch.int64, device=device)
    n_valid = torch.zeros(n_buckets, dtype=torch.int64, device=device)
    n_ns = torch.zeros(n_buckets, dtype=torch.int64, device=device)

    # Build a vocab-wide K-lookup tensor: K_lookup[token_id] = K for bot
    # tokens, bos_id → "reset to 1" sentinel, else 0 (no change).
    # We encode bos_id as -1 (reset sentinel) since 0 means "no change".
    vocab_size = int(token_bytes.shape[0])
    K_lookup = torch.zeros(vocab_size, dtype=torch.int32, device=device)
    K_lookup[bos_id] = -1  # reset sentinel
    for tok_id, K in bot_K_ids.items():
        K_lookup[tok_id] = K

    batch_iter = iter(loader)
    for _ in range(steps):
        batch = next(batch_iter)
        x, y, *extras = batch                # (B, T)
        loss2d = model(x, y, *extras, loss_reduction='none')  # (B, T) nats

        B, T = x.shape
        # Per-position K — scan inputs row-by-row. Vectorized:
        #   step_K[i] = K_lookup[x[i]]   (in {-1, 0, 1..kmax})
        #   if step_K[i] == 0 → keep prev K
        #   if step_K[i] == -1 → reset to 1
        #   else → step_K[i]
        step_K = K_lookup[x]  # (B, T)
        # Per-row scan: a small Python loop over T is fine (T=2048; B=16
        # → ~32k positions per batch, scan is cheap on CPU).
        # We do it on CPU once and ship K_per_pos back to device.
        step_K_cpu = step_K.to("cpu", non_blocking=True)
        K_per_pos = torch.empty((B, T), dtype=torch.int32)
        for b in range(B):
            K = 1
            row = step_K_cpu[b].numpy()
            out = K_per_pos[b].numpy()
            for i in range(T):
                v = row[i]
                if v == -1:
                    K = 1
                elif v > 0:
                    K = int(v)
                out[i] = K
        K_per_pos = K_per_pos.to(device)

        # Build per-position byte counts.
        y_safe = torch.where(y >= 0, y, torch.zeros_like(y))
        bytes2d = torch.where(
            y >= 0,
            token_bytes[y_safe],
            torch.zeros_like(y, dtype=token_bytes.dtype),
        )
        valid_mask = (y >= 0)
        non_special_mask = bytes2d > 0

        # Flatten everything for bucket aggregation.
        loss_flat = loss2d.view(-1).to(torch.float64)
        K_flat = K_per_pos.view(-1).to(torch.int64)
        valid_flat = valid_mask.view(-1)
        ns_flat = non_special_mask.view(-1)
        bytes_flat = bytes2d.view(-1).to(torch.int64)

        # Per-K scatter-add.
        nats_all.scatter_add_(0, K_flat, loss_flat * valid_flat)
        nats_ns.scatter_add_(0, K_flat, loss_flat * ns_flat)
        bytes_.scatter_add_(0, K_flat, bytes_flat)
        n_valid.scatter_add_(0, K_flat, valid_flat.to(torch.int64))
        n_ns.scatter_add_(0, K_flat, ns_flat.to(torch.int64))

        # Also accumulate to global bucket (index 0).
        nats_all[0] += (loss_flat * valid_flat).sum()
        nats_ns[0] += (loss_flat * ns_flat).sum()
        bytes_[0] += bytes_flat.sum()
        n_valid[0] += valid_flat.sum()
        n_ns[0] += ns_flat.sum()

    # DDP reduce: sum each per-K bucket across ranks. Buckets are
    # disjoint per rank because the distributed loader hands each rank a
    # disjoint shard of docs, so this is just sum-of-disjoint-counts.
    if dist.is_initialized() and dist.get_world_size() > 1:
        for t in (nats_all, nats_ns, bytes_, n_valid, n_ns):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    out = {}
    for K in range(n_buckets):
        nv = int(n_valid[K].item())
        nns = int(n_ns[K].item())
        nat_all = float(nats_all[K].item())
        nat_ns = float(nats_ns[K].item())
        byt = int(bytes_[K].item())
        if nv == 0 and K != 0:
            continue
        out[K] = {
            "n_valid": nv,
            "n_non_special": nns,
            "nats_all": nat_all,
            "nats_non_special": nat_ns,
            "total_bytes": byt,
            "bpb": (nat_ns / (math.log(2) * byt)) if byt else float("nan"),
            "ce_per_token": (nat_all / nv) if nv else float("nan"),
            "ce_per_non_special_token": (nat_ns / nns) if nns else float("nan"),
            "ppl_per_token": math.exp(nat_all / nv) if nv else float("nan"),
            "ppl_per_non_special": math.exp(nat_ns / nns) if nns else float("nan"),
        }
    return out


def per_wave_width_from_v4_report(report):
    """Convert v4 sliced wave-width buckets to the historical per_K shape."""
    aggregate = report["aggregate"]
    per_K = {
        0: {
            "n_valid": int(aggregate["n_valid_targets"]),
            "n_non_special": int(aggregate["n_non_special_targets"]),
            "nats_all": float(aggregate["ce_per_token"] * aggregate["n_valid_targets"]),
            "nats_non_special": float(aggregate["ce_per_non_special_token"]
                                      * aggregate["n_non_special_targets"]),
            "total_bytes": int(aggregate["total_bytes"]),
            "bpb": float(aggregate["bpb"]),
            "ce_per_token": float(aggregate["ce_per_token"]),
            "ce_per_non_special_token": float(aggregate["ce_per_non_special_token"]),
            "ppl_per_token": math.exp(float(aggregate["ce_per_token"])),
            "ppl_per_non_special": math.exp(float(aggregate["ce_per_non_special_token"])),
        }
    }

    wave = report["bucket"]["wave_width"]
    for label, bpb, ce, n_tok, total_bytes in zip(
        wave["labels"], wave["bpb"], wave["ce_per_token"],
        wave["n_tokens"], wave["total_bytes"],
    ):
        if not n_tok:
            continue
        K = int(label)
        per_K[K] = {
            "n_valid": int(n_tok),
            "n_non_special": int(n_tok),
            "nats_all": float(ce * n_tok),
            "nats_non_special": float(ce * n_tok),
            "total_bytes": int(total_bytes),
            "bpb": float(bpb),
            "ce_per_token": float(ce),
            "ce_per_non_special_token": float(ce),
            "ppl_per_token": math.exp(float(ce)),
            "ppl_per_non_special": math.exp(float(ce)),
        }
    return per_K


def print_per_K_table(per_K):
    print0("=" * 95)
    print0(f"{'K':>4} {'n_valid':>10} {'n_nonsp':>10} {'bpb':>9} "
           f"{'ce_tok':>9} {'ppl_tok':>9} {'ce_nonsp':>10} {'ppl_nonsp':>10}")
    print0("-" * 95)
    for K in sorted(per_K.keys()):
        d = per_K[K]
        tag = "GLOBAL" if K == 0 else str(K)
        print0(f"{tag:>4} {d['n_valid']:>10,} {d['n_non_special']:>10,} "
               f"{d['bpb']:>9.4f} {d['ce_per_token']:>9.4f} "
               f"{d['ppl_per_token']:>9.3f} {d['ce_per_non_special_token']:>10.4f} "
               f"{d['ppl_per_non_special']:>10.3f}")
    print0("=" * 95)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-tag", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--eval-tokens", type=int, default=5 * 1024 * 1024)
    p.add_argument("--override-kmax", type=int, default=None)
    p.add_argument("--override-warmup-threshold", type=int, default=None)
    p.add_argument("--out", default=None,
                   help="optional JSON output path. Default writes next to checkpoint.")
    args = p.parse_args()

    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    checkpoints_dir = os.path.join(get_base_dir(), "base_checkpoints")
    print0(f"Loading {args.model_tag} step={args.step}")
    model, tokenizer, meta = load_model_from_dir(
        checkpoints_dir, device, phase="eval",
        model_tag=args.model_tag, step=args.step,
    )
    model.eval()

    uc = meta.get("user_config", {})
    if uc.get("model") != "multithread_lm":
        raise SystemExit(f"This script is for multithread_lm only; got "
                         f"{uc.get('model')!r}")

    mt_layout_version = uc.get("mt_layout_version", "v4")
    mt_kmax_train = int(uc["mt_kmax"])
    mt_rope_stride = int(uc["mt_rope_stride"])
    mt_warmup_threshold_train = int(uc.get("mt_warmup_threshold", 0))
    mt_kmax = (args.override_kmax if args.override_kmax is not None
               else mt_kmax_train)
    mt_warmup_threshold = (args.override_warmup_threshold
                           if args.override_warmup_threshold is not None
                           else mt_warmup_threshold_train)
    if mt_kmax != mt_kmax_train or mt_warmup_threshold != mt_warmup_threshold_train:
        print0(f"OVERRIDE: kmax {mt_kmax_train}->{mt_kmax}, "
               f"warmup_threshold {mt_warmup_threshold_train}->{mt_warmup_threshold}")

    model_config = meta["model_config"]
    max_seq_len = int(meta.get("max_seq_len", model_config["sequence_len"]))
    rotary_seq_len = (model_config["sequence_len"]
                      * model_config.get("rotary_seq_len_factor", 1))

    token_bytes = get_token_bytes(name=meta.get("tokenizer_name"), device=device)

    tokens_per_step = args.device_batch_size * max_seq_len * ddp_world_size
    eval_steps = max(1, args.eval_tokens // tokens_per_step)
    eval_tokens_actual = eval_steps * tokens_per_step
    print0(f"eval_steps={eval_steps} -> {eval_tokens_actual:,} tokens")

    if mt_layout_version == "v4":
        mt_v4_parallel_cap_train = int(uc.get("mt_v4_parallel_cap", 0))
        if mt_v4_parallel_cap_train > 0:
            raise SystemExit(
                "v4 per-K/wave-width eval currently supports uncapped v4 only; "
                "cap rows should use aggregate BPB plus token-yield accounting."
            )
        loader = tokenizing_distributed_data_loader_multithread_v4_slices(
            tokenizer, args.device_batch_size, max_seq_len, split="val",
            kmax=mt_kmax, rope_stride=mt_rope_stride,
            rotary_seq_len=rotary_seq_len,
            device=device,
            warmup_threshold=mt_warmup_threshold,
        )
        print0("Running v4 wave-width eval via sliced diagnostics...")
        report = evaluate_bpb_sliced(model, loader, eval_steps, token_bytes)
        per_K = per_wave_width_from_v4_report(report)
        print_per_K_table(per_K)
        if ddp_rank == 0:
            out_path = (args.out or os.path.join(
                checkpoints_dir, args.model_tag,
                f"val_loss_per_K_step{meta['step']:06d}.json"))
            payload = {
                "model_tag": args.model_tag,
                "step": meta["step"],
                "tokenizer_name": meta.get("tokenizer_name"),
                "mt_layout_version": "v4",
                "mt_kmax": mt_kmax,
                "mt_kmax_train": mt_kmax_train,
                "mt_rope_stride": mt_rope_stride,
                "mt_warmup_threshold": mt_warmup_threshold,
                "mt_warmup_threshold_train": mt_warmup_threshold_train,
                "eval_tokens": eval_tokens_actual,
                "bucket_definition": "v4 wave_width (active ordinary middle lines at row position)",
                "per_K": {str(k): v for k, v in per_K.items()},
            }
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print0(f"Wrote {out_path}")
        return

    loader = tokenizing_distributed_data_loader_multithread(
        tokenizer, args.device_batch_size, max_seq_len, split="val",
        kmax=mt_kmax, rope_stride=mt_rope_stride,
        rotary_seq_len=rotary_seq_len,
        device=device,
        warmup_threshold=mt_warmup_threshold,
    )

    bos_id = tokenizer.encode_special("<|bos|>")
    bot_K_ids = {tokenizer.encode_special(f"<|bot_{k}|>"): k
                 for k in range(1, mt_kmax + 1)}
    print0(f"bos_id={bos_id}  bot_K_ids={ {bot_K_ids[t]: t for t in bot_K_ids} }")

    print0("Running per-K eval...")
    per_K = evaluate_per_K(model, loader, eval_steps, token_bytes,
                           bos_id, bot_K_ids, mt_kmax, device)

    print_per_K_table(per_K)

    if ddp_rank == 0:
        suffix = ("" if mt_kmax == mt_kmax_train
                  and mt_warmup_threshold == mt_warmup_threshold_train
                  else f"_K{mt_kmax}_W{mt_warmup_threshold}")
        out_path = (args.out or os.path.join(
            checkpoints_dir, args.model_tag,
            f"val_loss_per_K_step{meta['step']:06d}{suffix}.json"))
        payload = {
            "model_tag": args.model_tag,
            "step": meta["step"],
            "tokenizer_name": meta.get("tokenizer_name"),
            "mt_layout_version": mt_layout_version,
            "mt_kmax": mt_kmax,
            "mt_kmax_train": mt_kmax_train,
            "mt_warmup_threshold": mt_warmup_threshold,
            "mt_warmup_threshold_train": mt_warmup_threshold_train,
            "eval_tokens": eval_tokens_actual,
            "per_K": {str(k): v for k, v in per_K.items()},
        }
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print0(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
