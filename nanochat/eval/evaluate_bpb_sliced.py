"""Sliced BPB evaluator.

Same forward pass as `evaluate_bpb`, but bucketizes the per-token
cross-entropy + per-target byte counts by per-position slice labels.
Returns per-bucket BPB + CE for each label stream so callers can print
breakdowns like:

    line_role:        warmup / middle_sibling / tail        (v4 only)
    depth_in_line:    1 / 2 / 3-4 / 5-8 / 9-16 / 17+
    line_start_mask:  on / off
    is_anchor_target: on / off
    wave_width:       1 / 2 / ... / 32                       (v4 only)

Aggregate `val/bpb`, `val/ce_per_non_special_token`, `val/ce_per_token`
mirror the unsliced eval and are returned alongside the breakdowns.

Layouts:
  V4_SCHEMA       — block-MT v4 (5 slices, fed by
                    nanochat.eval.mt_v4_diagnostics).
  VANILLA_SCHEMA  — vanilla simple_gpt (3 shared slices; meta derived
                    inline by `derive_vanilla_slice_meta`).
"""
import math

import torch
import torch.distributed as dist


# Schemas drive which slices the evaluator accumulates. Each schema is:
#   {"buckets": {name: [label0, label1, ...]}, "masks": [name0, name1, ...]}
# The evaluator silently skips any slice whose meta key isn't present in the
# batch's slice_meta dict — so a schema can be a superset of what some loaders
# emit (within reason) without crashing.

V4_SCHEMA = {
    "buckets": {
        "line_role": ["warmup", "middle_sibling", "tail"],
        "depth_in_line": ["1", "2", "3-4", "5-8", "9-16", "17+"],
        "wave_width": [str(i) for i in range(33)],   # widen if a doc ever ramps deeper
    },
    "masks": ["line_start_mask", "is_anchor_target"],
}

VANILLA_SCHEMA = {
    "buckets": {
        "depth_in_line": ["1", "2", "3-4", "5-8", "9-16", "17+"],
    },
    "masks": ["line_start_mask", "is_anchor_target"],
}

DEPTH_BUCKET_EDGES = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10**9))


def _depth_bucketize(depth_1based: torch.Tensor) -> torch.Tensor:
    """Map 1-based depths to bucket indices in {0..5}, -1 for non-positive."""
    out = torch.full_like(depth_1based, -1, dtype=torch.int64)
    for b, (lo, hi) in enumerate(DEPTH_BUCKET_EDGES):
        in_range = (depth_1based >= lo) & (depth_1based <= hi)
        out = torch.where(in_range, torch.full_like(out, b), out)
    return out


def derive_vanilla_slice_meta(inputs: torch.Tensor, targets: torch.Tensor,
                              token_bytes: torch.Tensor,
                              nl_mask: torch.Tensor, bos_id: int) -> dict:
    """Derive vanilla-loader slice meta from the input/target stream.

    Vanilla packs `<bos> doc1_tokens <bos> doc2_tokens ...` per row; the only
    in-stream "special" is `<bos>`. A "line break input" is either `<bos>`
    or a newline-bearing token, and the target at such a position is the
    first content of a new line.

    Args
        inputs, targets: (B, T) int64 tensors. targets[i] is the next token.
        token_bytes:     (vocab_size,) int — 0 for specials (mask), positive for content.
        nl_mask:         (vocab_size,) bool — True for PURE_NL / TRAIL_NL token ids.
        bos_id:          int — vanilla doc-anchor token id.

    Returns dict with three (B, T) int8 tensors:
        line_start_mask, depth_in_line, is_anchor_target
    """
    device = inputs.device
    B, T = inputs.shape

    valid = targets >= 0
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    target_bytes = torch.where(valid, token_bytes[safe_targets],
                               torch.zeros_like(targets, dtype=token_bytes.dtype))
    target_is_content = target_bytes > 0
    target_is_anchor = valid & (target_bytes == 0)

    # Line-break input: <bos> OR newline-bearing token at the predicting position.
    break_input = (inputs == bos_id) | nl_mask[inputs]

    # depth-in-line via cummax of "position when last break was seen".
    arange_t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    pos_or_neg1 = torch.where(break_input, arange_t,
                              torch.full_like(arange_t, -1))
    last_break_pos, _ = pos_or_neg1.cummax(dim=1)
    # depth_target = (i - last_break_pos) + 1.
    #   at i = break_pos:     target predicted is first content of new line  -> depth=1
    #   at i = break_pos + 1: target predicted is the second content          -> depth=2
    #   ... and so on. Positions before the first break (last_break_pos==-1)
    # don't occur in practice — vanilla rows always start with <bos>, so
    # position 0 is itself a break — but defensively the +1 still places
    # them in the "17+" tail bucket.
    depth_target_1based = (arange_t - last_break_pos) + 1

    depth_in_line = _depth_bucketize(depth_target_1based)
    # Restrict to valid + content (anchor targets get -1 since "depth-in-line"
    # for a special target isn't meaningful — these positions are predicting
    # an inter-doc <bos>).
    depth_in_line = torch.where(valid & target_is_content,
                                depth_in_line,
                                torch.full_like(depth_in_line, -1))

    line_start_mask = (break_input & target_is_content).to(torch.int8)
    is_anchor_target = target_is_anchor.to(torch.int8)

    return {
        "line_start_mask": line_start_mask,
        "depth_in_line": depth_in_line.to(torch.int8),
        "is_anchor_target": is_anchor_target,
    }


@torch.no_grad()
def evaluate_bpb_sliced(model, batches, steps, token_bytes,
                        schema=V4_SCHEMA):
    """Sliced BPB / CE eval over `steps` batches.

    `batches` must yield 4-tuples `(inputs, targets, rope_idx, slice_meta_dict)`.
    For vanilla simple_gpt (no rope_idx), the orchestrator should wrap the
    2-tuple loader with a generator that adds `rope_idx=None` and a derived
    slice_meta. Model is invoked as `model(x, y, rope_idx, ...)` when
    `rope_idx` is not None, otherwise as `model(x, y, ...)`.

    Returns a dict:
        {
          "aggregate": {bpb, ce_per_non_special_token, ce_per_token,
                        n_non_special_targets, n_valid_targets, total_bytes},
          "bucket": {<slice_name>: {labels, bpb, ce_per_token, n_tokens, total_bytes}},
          "mask":   {<mask_name>: {"on": {...}, "off": {...}}},
        }
    """
    device = model.get_device()
    bucket_slices = schema["buckets"]
    mask_slices = schema["masks"]

    agg_nats_non_special = torch.tensor(0.0, dtype=torch.float32, device=device)
    agg_nats_all = torch.tensor(0.0, dtype=torch.float32, device=device)
    agg_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    agg_count_non_special = torch.tensor(0, dtype=torch.int64, device=device)
    agg_count_all = torch.tensor(0, dtype=torch.int64, device=device)

    bucket_nats = {name: torch.zeros(len(labels), dtype=torch.float32, device=device)
                   for name, labels in bucket_slices.items()}
    bucket_bytes = {name: torch.zeros(len(labels), dtype=torch.int64, device=device)
                    for name, labels in bucket_slices.items()}
    bucket_count = {name: torch.zeros(len(labels), dtype=torch.int64, device=device)
                    for name, labels in bucket_slices.items()}
    mask_nats = {name: torch.zeros(2, dtype=torch.float32, device=device)
                 for name in mask_slices}
    mask_bytes = {name: torch.zeros(2, dtype=torch.int64, device=device)
                  for name in mask_slices}
    mask_count = {name: torch.zeros(2, dtype=torch.int64, device=device)
                  for name in mask_slices}

    batch_iter = iter(batches)
    for _ in range(steps):
        batch = next(batch_iter)
        if len(batch) != 4:
            raise ValueError(
                f"evaluate_bpb_sliced expects 4-tuple "
                f"(inputs, targets, rope_idx_or_None, slice_meta_dict); "
                f"got {len(batch)}-tuple"
            )
        x, y, rope_idx, slice_meta = batch
        if rope_idx is None:
            loss2d = model(x, y, loss_reduction='none')
        else:
            loss2d = model(x, y, rope_idx, loss_reduction='none')
        loss = loss2d.view(-1)
        y_flat = y.view(-1)

        valid = y_flat >= 0
        y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
        num_bytes = torch.where(
            valid, token_bytes[y_safe],
            torch.zeros_like(y_flat, dtype=token_bytes.dtype),
        )
        non_special = num_bytes > 0

        agg_nats_non_special += (loss * non_special).sum()
        agg_nats_all += (loss * valid).sum()
        agg_bytes += num_bytes.sum()
        agg_count_non_special += non_special.sum()
        agg_count_all += valid.sum()

        nonspec_long = non_special.to(torch.int64)
        loss_nonspec = loss * non_special
        bytes_nonspec = num_bytes

        for name, labels in bucket_slices.items():
            if name not in slice_meta:
                continue
            n_buckets = len(labels)
            label_tensor = slice_meta[name].view(-1).to(torch.int64)
            in_range = (label_tensor >= 0) & (label_tensor < n_buckets) & valid
            safe_labels = torch.where(in_range, label_tensor,
                                      torch.zeros_like(label_tensor))
            bucket_nats[name].scatter_add_(
                0, safe_labels,
                (loss_nonspec * in_range).to(torch.float32),
            )
            bucket_bytes[name].scatter_add_(
                0, safe_labels,
                (bytes_nonspec * in_range).to(torch.int64),
            )
            bucket_count[name].scatter_add_(
                0, safe_labels,
                (nonspec_long * in_range).to(torch.int64),
            )

        for name in mask_slices:
            if name not in slice_meta:
                continue
            mask = (slice_meta[name].view(-1) == 1) & valid
            off_mask = (~mask) & valid
            mask_nats[name][1] += (loss_nonspec * mask).sum()
            mask_bytes[name][1] += (bytes_nonspec * mask).sum()
            mask_count[name][1] += (nonspec_long * mask).sum()
            mask_nats[name][0] += (loss_nonspec * off_mask).sum()
            mask_bytes[name][0] += (bytes_nonspec * off_mask).sum()
            mask_count[name][0] += (nonspec_long * off_mask).sum()

    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        tensors = [agg_nats_non_special, agg_nats_all, agg_bytes,
                   agg_count_non_special, agg_count_all]
        for name in bucket_slices:
            tensors += [bucket_nats[name], bucket_bytes[name], bucket_count[name]]
        for name in mask_slices:
            tensors += [mask_nats[name], mask_bytes[name], mask_count[name]]
        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    ln2 = math.log(2)
    aggregate = {
        "bpb": (agg_nats_non_special.item() / (ln2 * agg_bytes.item())
                if agg_bytes.item() else float("inf")),
        "ce_per_non_special_token": (agg_nats_non_special.item()
                                     / agg_count_non_special.item()
                                     if agg_count_non_special.item()
                                     else float("inf")),
        "ce_per_token": (agg_nats_all.item() / agg_count_all.item()
                         if agg_count_all.item() else float("inf")),
        "n_non_special_targets": agg_count_non_special.item(),
        "n_valid_targets": agg_count_all.item(),
        "total_bytes": agg_bytes.item(),
    }

    bucket = {}
    for name, labels in bucket_slices.items():
        nats = bucket_nats[name].tolist()
        bytes_ = bucket_bytes[name].tolist()
        counts = bucket_count[name].tolist()
        bucket[name] = {
            "labels": labels,
            "bpb": [n / (ln2 * b) if b else float("nan")
                    for n, b in zip(nats, bytes_)],
            "ce_per_token": [n / c if c else float("nan")
                             for n, c in zip(nats, counts)],
            "n_tokens": counts,
            "total_bytes": bytes_,
        }

    masks = {}
    for name in mask_slices:
        nats = mask_nats[name].tolist()
        bytes_ = mask_bytes[name].tolist()
        counts = mask_count[name].tolist()
        out = {}
        for key, idx in (("off", 0), ("on", 1)):
            out[key] = {
                "bpb": (nats[idx] / (ln2 * bytes_[idx])
                        if bytes_[idx] else float("nan")),
                "ce_per_token": (nats[idx] / counts[idx]
                                 if counts[idx] else float("nan")),
                "n_tokens": counts[idx],
                "total_bytes": bytes_[idx],
            }
        masks[name] = out

    return {
        "aggregate": aggregate,
        "bucket": bucket,
        "mask": masks,
    }
