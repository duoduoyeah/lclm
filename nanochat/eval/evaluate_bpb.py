"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, return_dict=False):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).

    With `return_dict=True`, returns a dict with both BPB and per-token cross-entropies:
        {
            "bpb": float,                       # bits per byte (specials excluded)
            "ce_per_non_special_token": float,  # mean nats per non-special target
                                                # (specials with token_bytes==0 excluded;
                                                #  comparable to train/loss with c_L
                                                #  override positions removed — useful
                                                #  for block-MT where <eot>/<bot_K>/<bos>
                                                #  override targets inflate train/loss).
            "ce_per_token": float,              # mean nats per ALL valid targets
                                                # (incl. specials; mirrors train/loss).
            "n_non_special_targets": int,
            "n_valid_targets": int,
            "total_bytes": int,
        }
    """
    # record the losses
    device = model.get_device()
    total_nats_non_special = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_nats_all = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    total_count_non_special = torch.tensor(0, dtype=torch.int64, device=device)
    total_count_all = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        # Standard loaders yield (x, y); MT validation yields extra tensors
        # such as rope_idx. Forward any extras positionally.
        batch = next(batch_iter)
        x, y, *extras = batch
        loss2d = model(x, y, *extras, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            non_special = num_bytes2d > 0
            total_nats_non_special += (loss2d * non_special).sum()
            total_nats_all += (loss2d * valid).sum()
            total_bytes += num_bytes2d.sum()
            total_count_non_special += non_special.sum()
            total_count_all += valid.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            non_special = num_bytes2d > 0
            total_nats_non_special += (loss2d * non_special).sum()
            total_nats_all += loss2d.sum()
            total_bytes += num_bytes2d.sum()
            total_count_non_special += non_special.sum()
            total_count_all += y.numel()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        for t in (total_nats_non_special, total_nats_all, total_bytes,
                  total_count_non_special, total_count_all):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats_non_special = total_nats_non_special.item()
    total_nats_all = total_nats_all.item()
    total_bytes = total_bytes.item()
    total_count_non_special = total_count_non_special.item()
    total_count_all = total_count_all.item()
    bpb = total_nats_non_special / (math.log(2) * total_bytes) if total_bytes else float('inf')
    if not return_dict:
        return bpb
    ce_non_special = total_nats_non_special / total_count_non_special if total_count_non_special else float('inf')
    ce_all = total_nats_all / total_count_all if total_count_all else float('inf')
    return {
        "bpb": bpb,
        "ce_per_non_special_token": ce_non_special,
        "ce_per_token": ce_all,
        "n_non_special_targets": total_count_non_special,
        "n_valid_targets": total_count_all,
        "total_bytes": total_bytes,
    }
