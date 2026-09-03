"""Sliced re-eval of validation BPB on a vanilla simple_gpt checkpoint.

Apples-to-apples companion to `scripts.eval.blockmt_val_loss_slices`.
Same per-bucket BPB breakdown but on the vanilla `bos_bestfit` val
loader, with slice meta derived inline from the input/target stream.

Reports the three slices that have a clean vanilla analog:

  - depth_in_line     1 / 2 / 3-4 / 5-8 / 9-16 / 17+
                      position of the target within its line, counted
                      from the most recent line-break input (<bos> or a
                      newline-bearing token).
  - line_start_mask   on/off — target is the first content byte of a new
                      line (predicted from <bos> or a newline token).
                      Direct vanilla equivalent of the v4 metric.
  - is_anchor_target  on/off — target is a non-byte-bearing special
                      (effectively <bos> for vanilla; the only special
                      vanilla training data contains).

Layout-specific v4 slices (line_role, wave_width) are dropped — they
have no vanilla analog.

Usage:
    NANOCHAT_DATASET=climbmix_split \\
        NANOCHAT_TOKENIZER=nanochat_64k_mt2 \\
        torchrun --standalone --nproc_per_node=4 \\
            -m scripts.eval.simple_val_loss_slices -- \\
            --model-tag d24_r8_climbmix_simple \\
            --step 5952 \\
            --device-batch-size 16 \\
            --eval-tokens 20971520
"""
import argparse
import json
import os

import torch

from nanochat.common import autodetect_device_type, compute_init, get_base_dir, print0
from nanochat.train.checkpoint import load_model_from_dir
from nanochat.tokenizer import get_token_bytes
from nanochat.eval.evaluate_bpb_sliced import (
    evaluate_bpb_sliced, derive_vanilla_slice_meta, VANILLA_SCHEMA,
)
from nanochat.data.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.data.multithread_dataloader import (
    assert_walker_invariants, NL_CLASS_PURE, NL_CLASS_TRAIL,
)


def _build_nl_mask(tokenizer, vocab_size, device):
    """Boolean (vocab_size,) tensor: True for PURE_NL / TRAIL_NL token ids.

    Source of truth is the same walker classifier the v4 dataloader uses,
    so vanilla line-start positions match v4's notion of "line boundary"
    exactly (modulo packing).
    """
    nl_class = assert_walker_invariants(tokenizer)
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for tid, cls in nl_class.items():
        if cls in (NL_CLASS_PURE, NL_CLASS_TRAIL):
            mask[tid] = True
    return mask


def _sliced_batches(vanilla_loader, token_bytes, nl_mask, bos_id):
    """Wrap a vanilla 2-tuple loader so each yield matches the sliced
    evaluator's 4-tuple contract: (inputs, targets, None, slice_meta).
    rope_idx=None signals the simple_gpt forward path (no rope_idx arg)."""
    for inputs, targets in vanilla_loader:
        slice_meta = derive_vanilla_slice_meta(
            inputs, targets, token_bytes, nl_mask, bos_id,
        )
        yield inputs, targets, None, slice_meta


def _fmt_count(n):
    return f"{n:,}" if isinstance(n, int) else f"{n}"


def _print_bucket_table(name, breakdown):
    labels = breakdown["labels"]
    rows = [(labels[i], breakdown["n_tokens"][i], breakdown["total_bytes"][i],
             breakdown["bpb"][i], breakdown["ce_per_token"][i])
            for i in range(len(labels)) if breakdown["n_tokens"][i] > 0]
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
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-tag", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--eval-tokens", type=int, default=10 * 1024 * 1024)
    p.add_argument("--out", default=None,
                   help="optional JSON output path. Default writes next to checkpoint.")
    args = p.parse_args()

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
    declared_model = user_config.get("model", "?")
    if declared_model not in ("simple_gpt", "gpt"):
        print0(f"WARN: checkpoint declares model={declared_model!r}; this script "
               f"targets vanilla simple_gpt/gpt — for multithread_lm use "
               f"`scripts.eval.blockmt_val_loss_slices`.")

    model_config = meta["model_config"]
    max_seq_len = int(meta.get("max_seq_len", model_config["sequence_len"]))
    vocab_size = int(model_config.get("vocab_size", tokenizer.get_vocab_size()))
    token_bytes = get_token_bytes(name=meta.get("tokenizer_name"), device=device)
    nl_mask = _build_nl_mask(tokenizer, vocab_size=token_bytes.shape[0], device=device)
    bos_id = tokenizer.get_bos_token_id()
    print0(f"vanilla slice config: bos_id={bos_id} n_nl_token_ids={int(nl_mask.sum())} "
           f"max_seq_len={max_seq_len}")

    tokens_per_step = args.device_batch_size * max_seq_len * ddp_world_size
    eval_steps = max(1, args.eval_tokens // tokens_per_step)
    eval_tokens_actual = eval_steps * tokens_per_step
    print0(f"eval_steps={eval_steps} -> {eval_tokens_actual:,} tokens "
           f"(dev_bs={args.device_batch_size}, world={ddp_world_size})")

    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, max_seq_len, "val",
        device=device,
    )
    sliced_loader = _sliced_batches(val_loader, token_bytes, nl_mask, bos_id)

    print0("Running evaluate_bpb_sliced (vanilla schema)...")
    report = evaluate_bpb_sliced(model, sliced_loader, eval_steps, token_bytes,
                                 schema=VANILLA_SCHEMA)

    if ddp_rank != 0:
        return

    agg = report["aggregate"]
    print0("=" * 78)
    print0(f"model_tag={args.model_tag}  step={meta['step']}  layout=vanilla")
    print0(f"  val/bpb                      : {agg['bpb']:.6f}")
    print0(f"  val/ce_per_non_special_token : {agg['ce_per_non_special_token']:.6f}")
    print0(f"  val/ce_per_token             : {agg['ce_per_token']:.6f}")
    print0(f"  n_non_special_targets        : {agg['n_non_special_targets']:,}")
    print0(f"  n_valid_targets              : {agg['n_valid_targets']:,}")
    print0(f"  total_bytes                  : {agg['total_bytes']:,}")
    print0("-" * 78)
    print0("Sliced (content-only — bpb and ce computed over non-special targets):")
    for name in ("depth_in_line",):
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
        "model_family": declared_model,
        "eval_tokens": eval_tokens_actual,
        "report": report,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print0(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
