"""Re-eval validation cross-entropy on a vanilla (simple_gpt) checkpoint.

Companion to `scripts/eval_blockmt_val_loss.py`. Uses the same evaluator
(`nanochat.eval.evaluate_bpb`) and the vanilla `bos_bestfit` val loader
(same as `scripts/base_train.py` and `scripts/base_eval.py` use for
simple_gpt).

Reports:
  - val/bpb                     bits per byte over non-special targets
                                (specials have token_bytes=0; for vanilla
                                training this is mostly just <bos>).
  - val/ce_per_non_special      mean nats per non-special target. For
                                simple_gpt this matches the MT model's
                                `ce_per_non_special` apples-to-apples
                                (both are mean CE over content tokens).
  - val/ce_per_token            mean nats per ALL valid targets. Mirrors
                                train/loss.

Usage:
    NANOCHAT_DATASET=climbmix_split \\
    NANOCHAT_TOKENIZER=nanochat_64k_mt2 \\
        python -m scripts.eval.simple_val_loss \\
            --model-tag d24_r8_climbmix_simple \\
            --step 5952 \\
            --device-batch-size 16 \\
            --eval-tokens 5242880
"""
import argparse
import json
import os

import torch

from nanochat.common import autodetect_device_type, compute_init, get_base_dir, print0
from nanochat.train.checkpoint import load_model_from_dir
from nanochat.tokenizer import get_token_bytes
from nanochat.eval.evaluate_bpb import evaluate_bpb
from nanochat.data.dataloader import tokenizing_distributed_data_loader_bos_bestfit


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-tag", required=True,
                   help="checkpoint dir under $NANOCHAT_BASE_DIR/base_checkpoints/")
    p.add_argument("--step", type=int, default=None,
                   help="checkpoint step (default: largest in dir)")
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
        print0(f"WARN: checkpoint declares model={declared_model!r} — this script "
               f"is for vanilla simple_gpt/gpt; for multithread_lm use "
               f"`scripts.eval.blockmt_val_loss`.")

    model_config = meta["model_config"]
    max_seq_len = int(meta.get("max_seq_len", model_config["sequence_len"]))
    token_bytes = get_token_bytes(name=meta.get("tokenizer_name"), device=device)

    tokens_per_step = args.device_batch_size * max_seq_len * ddp_world_size
    eval_steps = max(1, args.eval_tokens // tokens_per_step)
    eval_tokens_actual = eval_steps * tokens_per_step
    print0(f"eval_steps={eval_steps} -> {eval_tokens_actual:,} tokens "
           f"(dev_bs={args.device_batch_size}, world={ddp_world_size}, T={max_seq_len})")

    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, max_seq_len, "val",
        device=device,
    )

    print0("Running evaluate_bpb (return_dict=True)...")
    metrics = evaluate_bpb(model, val_loader, eval_steps, token_bytes, return_dict=True)
    if ddp_rank == 0:
        print0("=" * 72)
        print0(f"model_tag={args.model_tag}  step={meta['step']}")
        print0(f"  val/bpb                      : {metrics['bpb']:.6f}")
        print0(f"  val/ce_per_non_special_token : {metrics['ce_per_non_special_token']:.6f}")
        print0(f"  val/ce_per_token             : {metrics['ce_per_token']:.6f}")
        print0(f"  n_non_special_targets        : {metrics['n_non_special_targets']:,}")
        print0(f"  n_valid_targets              : {metrics['n_valid_targets']:,}")
        print0(f"  total_bytes                  : {metrics['total_bytes']:,}")
        print0("=" * 72)
        out_path = (args.out or os.path.join(
            checkpoints_dir, args.model_tag,
            f"val_loss_eval_step{meta['step']:06d}.json"))
        payload = {
            "model_tag": args.model_tag,
            "step": meta["step"],
            "tokenizer_name": meta.get("tokenizer_name"),
            "model_family": declared_model,
            "eval_tokens": eval_tokens_actual,
            "metrics": metrics,
        }
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print0(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
