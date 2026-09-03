"""Re-eval validation cross-entropy on a block-MT (multithread_lm) checkpoint.

Uses the same multithread val loader as train (so the layout includes c_L
target overrides — `<eot>`, `<bot_K>`, `<bos>` — at decision and
non-decision thread boundaries). Reports both:

  - val/bpb                     bits per byte over non-special targets.
                                Special tokens have token_bytes=0 so they
                                are auto-excluded; this matches v1's
                                in-loop val/bpb (0.8471 at step 5161).

  - val/ce_per_non_special      mean nats per non-special target. This is
                                what train/loss would be if the c_L
                                override positions (<eot>/<bot_K>/<bos>)
                                were ignored — useful for testing whether
                                v2's lower train/loss is driven by easy
                                eot prediction at non-decision thread c_L.

  - val/ce_per_token            mean nats per ALL valid targets (incl.
                                c_L overrides). Mirrors train/loss
                                directly: should be near v2's last
                                logged train/loss=2.7485 if held-out
                                generalization is comparable to train.

Reads loader layout knobs from the checkpoint's `meta_*.json`
(`user_config`) so it matches the training-time loader exactly, including
the v4 parallel cap when present.

Usage:
    NANOCHAT_DATASET=climbmix_split \\
        python -m scripts.eval.blockmt_val_loss \\
            --model-tag d12_r20_climbmix_blockmt_v2 \\
            --device-batch-size 16 \\
            --eval-tokens 10485760
"""
import argparse
import json
import os

import torch

from nanochat.common import autodetect_device_type, compute_init, get_base_dir, print0
from nanochat.train.checkpoint import load_model_from_dir
from nanochat.tokenizer import get_token_bytes
from nanochat.eval.evaluate_bpb import evaluate_bpb
from nanochat.data.multithread_dataloader import (
    tokenizing_distributed_data_loader_multithread,
)


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
    parser.add_argument("--override-kmax", type=int, default=None,
                        help="hold the model fixed but force a different "
                             "max wave width on the loader. Use for the "
                             "quality-vs-parallelism sweep (K=1 → pure "
                             "sequential rows; K=16 → training default).")
    parser.add_argument("--override-warmup-threshold", type=int, default=None,
                        help="override mt_warmup_threshold (default: meta value).")
    parser.add_argument("--mt-v4-parallel-cap", type=int, default=None,
                        help="v4-only eval override for mt_v4_parallel_cap. "
                             "Default None = use checkpoint meta; 0 = force "
                             "uncapped.")
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
        raise SystemExit(f"This script is for multithread_lm only; checkpoint reports "
                         f"model={user_config.get('model')!r}")

    # The eot-aware multithread_dataloader requires the tokenizer to have
    # `<|eot|>` as a special token. Pre-v2 checkpoints (tokenizer
    # `nanochat_64k_mt`) predate this mechanism — skip cleanly so a batch
    # run over mixed tags continues with the remaining ones.
    try:
        tokenizer.encode_special("<|eot|>")
    except Exception:
        if ddp_rank == 0:
            print0(f"Skipping {args.model_tag}: tokenizer "
                   f"{meta.get('tokenizer_name', '?')!r} has no <|eot|> token "
                   f"(pre-v2 checkpoint, no eot-target mechanism). The "
                   f"in-loop val/bpb from training (eot-free by construction) "
                   f"is the right comparison number — fetch it from wandb.")
        return

    mt_kmax_train = int(user_config["mt_kmax"])
    mt_rope_stride = int(user_config["mt_rope_stride"])
    # v3 dataloader knob: warmup_threshold > 0 promotes leading blocks to
    # K=1 stages. Read from user_config (or top-level meta as fallback) so
    # eval-time layout matches train-time. Default 0 keeps v2 behavior.
    mt_warmup_threshold_train = int(
        user_config.get("mt_warmup_threshold",
                        meta.get("mt_warmup_threshold", 0))
    )
    # v4 layout has its own topology (sequential warmup + single middle SOT
    # ramp + sot_tail). Match the train-time layout exactly so eval data
    # carries the same anchors the model was trained on. Default "v3" for
    # back-compat with v2/v3 checkpoints that predate this key.
    mt_layout_version = user_config.get("mt_layout_version", "v4")
    mt_v4_parallel_cap_train = int(user_config.get("mt_v4_parallel_cap", 0))
    mt_v4_parallel_cap_train_resolved = (
        mt_v4_parallel_cap_train if mt_v4_parallel_cap_train > 0 else None
    )
    if args.mt_v4_parallel_cap is not None and args.mt_v4_parallel_cap < 0:
        raise SystemExit("--mt-v4-parallel-cap must be >= 0")
    if mt_layout_version == "v4":
        if args.mt_v4_parallel_cap is None:
            mt_v4_parallel_cap = mt_v4_parallel_cap_train_resolved
        else:
            mt_v4_parallel_cap = (
                args.mt_v4_parallel_cap if args.mt_v4_parallel_cap > 0 else None
            )
    else:
        if args.mt_v4_parallel_cap not in (None, 0):
            raise SystemExit("--mt-v4-parallel-cap is only valid for v4 checkpoints")
        mt_v4_parallel_cap = None
    mt_v4_parallel_cap_is_override = (
        args.mt_v4_parallel_cap is not None
        and mt_v4_parallel_cap != mt_v4_parallel_cap_train_resolved
    )
    mt_kmax = (args.override_kmax
               if args.override_kmax is not None else mt_kmax_train)
    mt_warmup_threshold = (args.override_warmup_threshold
                           if args.override_warmup_threshold is not None
                           else mt_warmup_threshold_train)
    if mt_kmax != mt_kmax_train or mt_warmup_threshold != mt_warmup_threshold_train:
        print0(f"OVERRIDE: kmax {mt_kmax_train}->{mt_kmax}, "
               f"warmup_threshold {mt_warmup_threshold_train}->{mt_warmup_threshold}")
    model_config = meta["model_config"]
    max_seq_len = int(meta.get("max_seq_len", model_config["sequence_len"]))
    rotary_seq_len = model_config["sequence_len"] * model_config.get("rotary_seq_len_factor", 1)
    print0(f"MT config: layout={mt_layout_version} kmax={mt_kmax} "
           f"rope_stride={mt_rope_stride} "
           f"warmup_threshold={mt_warmup_threshold} "
           f"v4_parallel_cap={mt_v4_parallel_cap if mt_v4_parallel_cap is not None else 'uncapped'} "
           f"max_seq_len={max_seq_len} rotary_seq_len={rotary_seq_len}")

    token_bytes = get_token_bytes(name=meta.get("tokenizer_name"), device=device)

    tokens_per_step = args.device_batch_size * max_seq_len * ddp_world_size
    eval_steps = max(1, args.eval_tokens // tokens_per_step)
    eval_tokens_actual = eval_steps * tokens_per_step
    print0(f"eval_steps={eval_steps} -> {eval_tokens_actual:,} tokens "
           f"(dev_bs={args.device_batch_size}, world={ddp_world_size})")

    val_loader = tokenizing_distributed_data_loader_multithread(
        tokenizer, args.device_batch_size, max_seq_len, split="val",
        kmax=mt_kmax, rope_stride=mt_rope_stride,
        rotary_seq_len=rotary_seq_len,
        device=device,
        warmup_threshold=mt_warmup_threshold,
        layout_version=mt_layout_version,
        v4_parallel_cap=mt_v4_parallel_cap,
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
        # Persist a sidecar JSON next to the checkpoint dir for plotting later.
        # K-override sweep: suffix the filename so we don't clobber the
        # canonical (matches-training) sidecar.
        is_override = (mt_kmax != mt_kmax_train
                       or mt_warmup_threshold != mt_warmup_threshold_train
                       or mt_v4_parallel_cap_is_override)
        suffix_bits = []
        if mt_kmax != mt_kmax_train or mt_warmup_threshold != mt_warmup_threshold_train:
            suffix_bits.append(f"K{mt_kmax}_W{mt_warmup_threshold}")
        if mt_v4_parallel_cap_is_override:
            cap = (f"v4cap{mt_v4_parallel_cap}"
                   if mt_v4_parallel_cap is not None else "v4uncap")
            suffix_bits.append(cap)
        suffix = ("_" + "_".join(suffix_bits)) if is_override else ""
        out_path = (args.out or os.path.join(
            checkpoints_dir, args.model_tag,
            f"val_loss_eval_step{meta['step']:06d}{suffix}.json"))
        payload = {
            "model_tag": args.model_tag,
            "step": meta["step"],
            "tokenizer_name": meta.get("tokenizer_name"),
            "mt_layout_version": mt_layout_version,
            "mt_kmax": mt_kmax,
            "mt_kmax_train": mt_kmax_train,
            "mt_rope_stride": mt_rope_stride,
            "mt_warmup_threshold": mt_warmup_threshold,
            "mt_warmup_threshold_train": mt_warmup_threshold_train,
            "mt_v4_parallel_cap": mt_v4_parallel_cap,
            "mt_v4_parallel_cap_train": mt_v4_parallel_cap_train_resolved,
            "eval_tokens": eval_tokens_actual,
            "metrics": metrics,
        }
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print0(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
