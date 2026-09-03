"""Eval/dump-only diagnostics for Block-MT v4 validation rows.

This module intentionally lives above ``nanochat.data``. The production
multithread layout and dataloader should only build training/eval rows; sliced
analysis metadata such as line role, line depth, and wave width is used by
offline diagnostics and paper tables, so it belongs in the eval layer.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import List

import torch

from nanochat.data.dataloader import _document_batches
from nanochat.data.multithread_dataloader import (
    NL_CLASS_PURE,
    NL_CLASS_TRAIL,
    assert_walker_invariants,
    build_doc_v4,
    split_tokens_into_blocks,
)
from nanochat.data.multithread_layout import (
    DocV4,
    RowMeta,
    compile_doc_v4,
    compute_targets_v4,
)


LINE_ROLE_WARMUP = 0
LINE_ROLE_MIDDLE_SIBLING = 1
LINE_ROLE_TAIL = 2

# depth-in-line buckets are indexed by the 1-based ordinal position of the
# target token within its line. These labels match evaluate_bpb_sliced.V4_SCHEMA.
DEPTH_BUCKET_EDGES = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10**9))
DEPTH_BUCKET_LABELS = ("1", "2", "3-4", "5-8", "9-16", "17+")

SLICE_FIELDS = (
    "line_role",
    "depth_in_line",
    "line_start_mask",
    "is_anchor_target",
    "wave_width",
)


def _depth_bucket(local_step: int) -> int:
    for b, (lo, hi) in enumerate(DEPTH_BUCKET_EDGES):
        if lo <= local_step <= hi:
            return b
    return -1


def compute_slice_meta_v4(meta: RowMeta, doc: DocV4, targets: List[int],
                          sot_id: int, sot_tail_id: int,
                          eot_id: int, bos_id: int) -> dict:
    """Per-position slice labels for v4 layout diagnostics.

    The returned lists all have length ``meta.N``. Positions that do not apply
    to a slice use -1 for bucket labels and 0 for masks.

    The metadata is aligned to the *predictor* row position, because the model
    loss at row position ``i`` predicts ``targets[i]``. For content targets,
    depth/line-start are computed from the target token's within-thread local
    step, not from the predictor token itself.
    """
    N = meta.N

    # Trajectory: thread -> sorted [(local_step, row_pos), ...].
    traj = defaultdict(list)
    for i in range(N):
        if meta.thread_idx[i] < 0:
            continue
        traj[meta.thread_idx[i]].append((meta.local_step[i], i))
    for gt in traj:
        traj[gt].sort()

    next_in_thread: List[int] = [-1] * N
    for entries in traj.values():
        for k in range(len(entries) - 1):
            _, pos = entries[k]
            _, npos = entries[k + 1]
            next_in_thread[pos] = npos

    # Thread-group partition for line_role.
    n_warmup = len(doc.warmup)
    n_middle = len(doc.middle)
    has_tail = doc.tail is not None
    tail_thread_idx = n_warmup + n_middle if has_tail else None

    def line_role_for_thread(gt: int) -> int:
        if gt < 0:
            return -1
        if gt < n_warmup:
            return LINE_ROLE_WARMUP
        if gt < n_warmup + n_middle:
            return LINE_ROLE_MIDDLE_SIBLING
        if has_tail and gt == tail_thread_idx:
            return LINE_ROLE_TAIL
        return -1

    def first_content_ls(gt: int) -> int:
        # Tail thread: ls=0 sot, ls=1 sot_tail, ls=2 first content.
        if has_tail and gt == tail_thread_idx:
            return 2
        # All other threads: ls=0 anchor, ls=1 first content.
        return 1

    # Wave-width precompute for middle ramp. Each middle thread t's
    # contribution sits at wave step w = t + ls.
    middle_lengths = [p.length for p in doc.middle]
    k_middle = len(middle_lengths)

    def width_at_wave(w: int) -> int:
        count = 0
        for t in range(k_middle):
            if t > w:
                break
            ls = w - t
            if 0 <= ls <= middle_lengths[t]:
                count += 1
        return count

    anchor_ids = {sot_id, eot_id, bos_id}
    if sot_tail_id is not None:
        anchor_ids.add(sot_tail_id)

    line_role: List[int] = [-1] * N
    depth_in_line: List[int] = [-1] * N
    line_start_mask: List[int] = [0] * N
    is_anchor_target: List[int] = [0] * N
    wave_width: List[int] = [0] * N

    for i in range(N):
        gt = meta.thread_idx[i]
        if gt < 0:
            continue
        role = line_role_for_thread(gt)
        line_role[i] = role
        if role == LINE_ROLE_MIDDLE_SIBLING:
            within_t = gt - n_warmup
            wave_width[i] = width_at_wave(within_t + meta.local_step[i])
        elif role in (LINE_ROLE_WARMUP, LINE_ROLE_TAIL):
            wave_width[i] = 1

        target = targets[i]
        if target < 0:
            continue
        if target in anchor_ids:
            is_anchor_target[i] = 1

        npos = next_in_thread[i]
        if npos == -1:
            # End-of-line override target (eot/bos). No within-thread
            # successor; depth/line-start don't apply.
            continue
        target_ls = meta.local_step[npos]
        depth_in_line[i] = _depth_bucket(target_ls)
        if is_anchor_target[i] == 0 and target_ls == first_content_ls(meta.thread_idx[npos]):
            line_start_mask[i] = 1

    return {
        "line_role": line_role,
        "depth_in_line": depth_in_line,
        "line_start_mask": line_start_mask,
        "is_anchor_target": is_anchor_target,
        "wave_width": wave_width,
    }


def compute_position_meta_v4(meta: RowMeta, targets: List[int],
                             content_token_ids: set[int]) -> dict:
    """Target-position metadata for v4 position-loss diagnostics.

    Returns two length-N arrays aligned to predictor row positions:

      - ``target_abs_doc_pos``: natural source-order content-token index of
        the target, counted 0-based by thread order and then local-step order.
      - ``target_reordered_pos``: row position at which that target token sits
        after the v4 reorder.

    Non-content targets and ignored/control targets receive -1. This keeps the
    loss bucketed by the token being predicted, not by the predictor slot.
    """
    N = meta.N

    # Map each input-side content token row position to its source-order index.
    content_positions = []
    for i, tok in enumerate(meta.tokens):
        if meta.thread_idx[i] < 0:
            continue
        if tok not in content_token_ids:
            continue
        content_positions.append((meta.thread_idx[i], meta.local_step[i], i))
    content_positions.sort()
    abs_by_row_pos = {row_pos: abs_pos
                      for abs_pos, (_, _, row_pos) in enumerate(content_positions)}

    traj = defaultdict(list)
    for i in range(N):
        if meta.thread_idx[i] < 0:
            continue
        traj[meta.thread_idx[i]].append((meta.local_step[i], i))
    for gt in traj:
        traj[gt].sort()

    next_in_thread: List[int] = [-1] * N
    for entries in traj.values():
        for k in range(len(entries) - 1):
            _, pos = entries[k]
            _, npos = entries[k + 1]
            next_in_thread[pos] = npos

    target_abs_doc_pos: List[int] = [-1] * N
    target_reordered_pos: List[int] = [-1] * N
    for i in range(N):
        target = targets[i]
        if target not in content_token_ids:
            continue
        npos = next_in_thread[i]
        if npos == -1:
            continue
        if npos not in abs_by_row_pos:
            continue
        target_abs_doc_pos[i] = abs_by_row_pos[npos]
        target_reordered_pos[i] = npos

    return {
        "target_abs_doc_pos": target_abs_doc_pos,
        "target_reordered_pos": target_reordered_pos,
    }


def _is_cuda_device(device) -> bool:
    return str(device).startswith("cuda")


def tokenizing_distributed_data_loader_with_state_multithread_v4_slices(
    tokenizer, B, T, split,
    kmax=16,
    rope_stride=256,
    rotary_seq_len=None,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    buffer_size=1000,
    warmup_threshold=16,
):
    """Eval-only v4 dataloader that yields slice metadata.

    This mirrors the production v4 multithread dataloader's row construction
    and best-fit packing, then additionally emits the diagnostic slice fields.
    It exists outside ``nanochat.data`` so training-critical data code does not
    carry paper/dump-only metadata paths.

    Yields ``(inputs, targets, rope_idx, slice_meta, state_dict)``.
    """
    assert split in ("train", "val")
    _ = kmax  # v4 middle ramp is intentionally uncapped; kept for call-site symmetry.

    sot_id = tokenizer.encode_special("<|sot|>")
    bos_id = tokenizer.get_bos_token_id()
    eot_id = tokenizer.encode_special("<|eot|>")
    sot_tail_id = tokenizer.encode_special("<|sot_tail|>")

    nl_class = assert_walker_invariants(tokenizer)
    nl_token_ids = {tid for tid, c in nl_class.items()
                    if c in (NL_CLASS_PURE, NL_CLASS_TRAIL)}
    assert nl_token_ids, "tokenizer has no newline-bearing tokens"
    bare_nl_ids = tokenizer.encode("\n")
    assert (len(bare_nl_ids) == 1
            and nl_class[bare_nl_ids[0]] == NL_CLASS_PURE), (
        f"tokenizer.encode('\\n') = {bare_nl_ids}; must be a single PURE_NL token"
    )

    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    pq_idx, rg_idx, epoch = 0, 0, 1
    doc_buffer = []

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        prepped_texts = []
        for doc_text in doc_batch:
            if not doc_text or not doc_text.strip():
                continue
            doc_text = doc_text.lstrip("\n\r")
            if doc_text:
                prepped_texts.append(doc_text)
        if not prepped_texts:
            return

        doc_token_lists = tokenizer.encode(prepped_texts, num_threads=tokenizer_threads)
        for doc_tokens in doc_token_lists:
            blocks = split_tokens_into_blocks(doc_tokens, nl_class)
            if not blocks:
                continue
            doc = build_doc_v4(blocks, warmup_threshold=warmup_threshold)
            if doc is None:
                continue
            meta = compile_doc_v4(
                doc,
                rope_stride=rope_stride,
                sot_id=sot_id,
                sot_tail_id=sot_tail_id,
                bos_id=bos_id,
                nl_token_ids=nl_token_ids,
            )
            if meta.N > T:
                continue
            targets = compute_targets_v4(meta, doc, bos_id=bos_id, eot_id=eot_id)
            slice_meta_doc = compute_slice_meta_v4(
                meta, doc, targets,
                sot_id=sot_id,
                sot_tail_id=sot_tail_id,
                eot_id=eot_id,
                bos_id=bos_id,
            )
            doc_buffer.append({
                "tokens": meta.tokens,
                "targets": targets,
                "rope_idx": meta.rope_idx,
                "rope_max": max(meta.rope_idx),
                "N": meta.N,
                "slice_meta": slice_meta_doc,
            })

    use_cuda = _is_cuda_device(device)
    num_fields = 3
    cpu_buffer = torch.empty(num_fields * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(num_fields * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[0 * B * T:1 * B * T].view(B, T)
    cpu_targets = cpu_buffer[1 * B * T:2 * B * T].view(B, T)
    cpu_rope_idx = cpu_buffer[2 * B * T:3 * B * T].view(B, T)
    inputs = gpu_buffer[0 * B * T:1 * B * T].view(B, T)
    targets = gpu_buffer[1 * B * T:2 * B * T].view(B, T)
    rope_idx = gpu_buffer[2 * B * T:3 * B * T].view(B, T)

    cpu_slice = {
        name: torch.empty(B * T, dtype=torch.int8, pin_memory=use_cuda).view(B, T)
        for name in SLICE_FIELDS
    }
    gpu_slice = {
        name: torch.empty(B * T, dtype=torch.int8, device=device).view(B, T)
        for name in SLICE_FIELDS
    }

    while True:
        cpu_inputs.zero_()
        cpu_targets.fill_(-1)
        cpu_rope_idx.zero_()
        for name in SLICE_FIELDS:
            cpu_slice[name].fill_(-1)

        for row_idx in range(B):
            pos = 0
            rope_cursor = 0
            while pos < T:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = T - pos
                best_idx = -1
                best_len = 0
                for i, item in enumerate(doc_buffer):
                    n = item["N"]
                    if n <= remaining and n > best_len:
                        best_idx = i
                        best_len = n

                if best_idx < 0:
                    break

                item = doc_buffer.pop(best_idx)
                N = item["N"]
                cpu_inputs[row_idx, pos:pos + N] = torch.tensor(item["tokens"], dtype=torch.long)
                cpu_targets[row_idx, pos:pos + N] = torch.tensor(item["targets"], dtype=torch.long)
                cpu_rope_idx[row_idx, pos:pos + N] = (
                    torch.tensor(item["rope_idx"], dtype=torch.long) + rope_cursor
                )
                for name in SLICE_FIELDS:
                    cpu_slice[name][row_idx, pos:pos + N] = torch.tensor(
                        item["slice_meta"][name], dtype=torch.int8,
                    )

                doc_max_abs_rope = rope_cursor + item["rope_max"]
                rope_cursor = ((doc_max_abs_rope // rope_stride) + 1) * rope_stride
                pos += N

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        if rotary_seq_len is not None:
            max_rope = int(cpu_rope_idx.max().item())
            if max_rope >= rotary_seq_len:
                warnings.warn(
                    f"rope_idx max {max_rope} >= rotary_seq_len {rotary_seq_len}; "
                    f"skipping batch (would have asserted in RoPE gather).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        for name in SLICE_FIELDS:
            gpu_slice[name].copy_(cpu_slice[name], non_blocking=use_cuda)

        yield inputs, targets, rope_idx, gpu_slice, state_dict


def tokenizing_distributed_data_loader_multithread_v4_slices(*args, **kwargs):
    """State-free helper for sliced v4 eval.

    Yields ``(inputs, targets, rope_idx, slice_meta_dict)``.
    """
    for inputs, targets, rope_idx, slice_meta, _state in (
        tokenizing_distributed_data_loader_with_state_multithread_v4_slices(*args, **kwargs)
    ):
        yield inputs, targets, rope_idx, slice_meta
