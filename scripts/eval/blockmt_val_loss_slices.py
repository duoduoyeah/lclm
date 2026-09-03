"""Sliced re-eval of validation BPB on a block-MT v4 checkpoint.

Mirrors `scripts.eval.blockmt_val_loss`, but instead of one aggregate BPB
the report breaks it down by per-position layout slices computed by
the eval-side v4 diagnostics loader:

  - line_role         warmup / middle_sibling / tail
  - depth_in_line     1 / 2 / 3-4 / 5-8 / 9-16 / 17+
  - line_start_mask   first-content-byte-of-line (on/off)
  - is_anchor_target  target is a special anchor (sot/sot_tail/eot/bos)
  - wave_width        # of concurrent middle-thread emissions at this row pos

The eval-side diagnostics loader mirrors the production v4 dataloader
topology, so the aggregate BPB should match `scripts.eval.blockmt_val_loss`
on the same eval-token budget while keeping slice metadata out of the core
training dataloader. The script hard-fails on non-v4 checkpoints with a clear
message.

Usage:
    NANOCHAT_DATASET=climbmix_split \\
        NANOCHAT_TOKENIZER=nanochat_64k_mt_v4 \\
        torchrun --standalone --nproc_per_node=4 \\
            -m scripts.eval.blockmt_val_loss_slices -- \\
            --model-tag d24_r8_climbmix_blockmt_v4 \\
            --step 5952 \\
            --device-batch-size 16 \\
            --eval-tokens 20971520
"""
import argparse
import json
import math
import os

import torch

from nanochat.common import autodetect_device_type, compute_init, get_base_dir, print0
from nanochat.train.checkpoint import load_model_from_dir
from nanochat.tokenizer import get_token_bytes
from nanochat.eval.evaluate_bpb_sliced import evaluate_bpb_sliced
from nanochat.eval.mt_v4_diagnostics import (
    tokenizing_distributed_data_loader_multithread_v4_slices,
)


def _fmt_count(n):
    return f"{n:,}" if isinstance(n, int) else f"{n}"


def _print_bucket_table(name, breakdown):
    labels = breakdown["labels"]
    bpb = breakdown["bpb"]
    ce = breakdown["ce_per_token"]
    counts = breakdown["n_tokens"]
    bytes_ = breakdown["total_bytes"]
    rows = [(labels[i], counts[i], bytes_[i], bpb[i], ce[i])
            for i in range(len(labels)) if counts[i] > 0]
    if not rows:
        print0(f"  [{name}] (no samples)")
        return
    print0(f"  [{name}]")
    print0(f"    {'label':>14}  {'n_tokens':>12}  {'bytes':>12}  "
           f"{'bpb':>9}  {'ce':>9}")
    for lbl, c, b, p, e in rows:
        print0(f"    {lbl:>14}  {_fmt_count(c):>12}  {_fmt_count(b):>12}  "
               f"{p:>9.6f}  {e:>9.6f}")


def _print_mask_table(name, breakdown):
    print0(f"  [{name}]  on vs off")
    print0(f"    {'side':>5}  {'n_tokens':>12}  {'bytes':>12}  "
           f"{'bpb':>9}  {'ce':>9}")
    for side in ("on", "off"):
        d = breakdown[side]
        c, b, p, e = d["n_tokens"], d["total_bytes"], d["bpb"], d["ce_per_token"]
        if c == 0:
            print0(f"    {side:>5}  (no samples)")
            continue
        print0(f"    {side:>5}  {_fmt_count(c):>12}  {_fmt_count(b):>12}  "
               f"{p:>9.6f}  {e:>9.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-tag", required=True,
                        help="checkpoint dir name under $NANOCHAT_BASE_DIR/base_checkpoints/")
    parser.add_argument("--step", type=int, default=None,
                        help="checkpoint step (default: largest in dir)")
    parser.add_argument("--device-batch-size", type=int, default=16)
    parser.add_argument("--eval-tokens", type=int, default=10 * 1024 * 1024,
                        help="number of tokens per rank to eval (rounded down to a "
                             "multiple of device_batch_size * max_seq_len)")
    parser.add_argument("--out", default=None,
                        help="optional JSON output path. Default writes next to checkpoint.")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    checkpoints_dir = os.path.join(get_base_dir(), "base_checkpoints")
    print0(f"Loading model_tag={args.model_tag} step={args.step} from {checkpoints_dir}")
    model, tokenizer, meta = load_model_from_dir(
        checkpoints_dir, device, phase="eval",
        model_tag=args.model_tag, step=args.step,
    )
    model.eval()
    print0(f"Loaded. step={meta['step']} config={meta.get('model_config', {})}")
    print0(f"Tokenizer={meta.get('tokenizer_name', '(unknown)')}")

    user_config = meta.get("user_config", {})
    if user_config.get("model") != "multithread_lm":
        raise SystemExit(
            f"eval_blockmt_val_loss_slices is for multithread_lm only; "
            f"checkpoint reports model={user_config.get('model')!r}"
        )
    mt_layout_version = user_config.get("mt_layout_version", "v4")
    if mt_layout_version != "v4":
        raise SystemExit(
            f"Sliced eval supports layout_version='v4' only; checkpoint reports "
            f"mt_layout_version={mt_layout_version!r}. Use scripts.eval.blockmt_val_loss "
            f"for v2/v3 aggregate BPB."
        )

    mt_kmax = int(user_config["mt_kmax"])
    mt_rope_stride = int(user_config["mt_rope_stride"])
    mt_warmup_threshold = int(
        user_config.get("mt_warmup_threshold",
                        meta.get("mt_warmup_threshold", 0))
    )
    model_config = meta["model_config"]
    max_seq_len = int(meta.get("max_seq_len", model_config["sequence_len"]))
    rotary_seq_len = model_config["sequence_len"] * model_config.get("rotary_seq_len_factor", 1)
    print0(f"MT config: layout=v4 kmax={mt_kmax} rope_stride={mt_rope_stride} "
           f"warmup_threshold={mt_warmup_threshold} "
           f"max_seq_len={max_seq_len} rotary_seq_len={rotary_seq_len}")

    token_bytes = get_token_bytes(name=meta.get("tokenizer_name"), device=device)

    tokens_per_step = args.device_batch_size * max_seq_len * ddp_world_size
    eval_steps = max(1, args.eval_tokens // tokens_per_step)
    eval_tokens_actual = eval_steps * tokens_per_step
    print0(f"eval_steps={eval_steps} -> {eval_tokens_actual:,} tokens "
           f"(dev_bs={args.device_batch_size}, world={ddp_world_size})")

    val_loader = tokenizing_distributed_data_loader_multithread_v4_slices(
        tokenizer, args.device_batch_size, max_seq_len, split="val",
        kmax=mt_kmax, rope_stride=mt_rope_stride,
        rotary_seq_len=rotary_seq_len,
        device=device,
        warmup_threshold=mt_warmup_threshold,
    )

    print0("Running evaluate_bpb_sliced...")
    report = evaluate_bpb_sliced(model, val_loader, eval_steps, token_bytes)

    if ddp_rank != 0:
        return

    agg = report["aggregate"]
    print0("=" * 78)
    print0(f"model_tag={args.model_tag}  step={meta['step']}  layout=v4")
    print0(f"  val/bpb                      : {agg['bpb']:.6f}")
    print0(f"  val/ce_per_non_special_token : {agg['ce_per_non_special_token']:.6f}")
    print0(f"  val/ce_per_token             : {agg['ce_per_token']:.6f}")
    print0(f"  n_non_special_targets        : {agg['n_non_special_targets']:,}")
    print0(f"  n_valid_targets              : {agg['n_valid_targets']:,}")
    print0(f"  total_bytes                  : {agg['total_bytes']:,}")
    print0("-" * 78)
    print0("Sliced (content-only — bpb and ce computed over non-special targets):")
    for name in ("line_role", "depth_in_line", "wave_width"):
        _print_bucket_table(name, report["bucket"][name])
    for name in ("line_start_mask", "is_anchor_target"):
        _print_mask_table(name, report["mask"][name])
    print0("=" * 78)

    out_path = (args.out or os.path.join(
        checkpoints_dir, args.model_tag,
        f"val_loss_eval_slices_step{meta['step']:06d}.json"))
    payload = {
        "model_tag": args.model_tag,
        "step": meta["step"],
        "tokenizer_name": meta.get("tokenizer_name"),
        "mt_layout_version": "v4",
        "mt_kmax": mt_kmax,
        "mt_rope_stride": mt_rope_stride,
        "mt_warmup_threshold": mt_warmup_threshold,
        "eval_tokens": eval_tokens_actual,
        "report": report,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print0(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
