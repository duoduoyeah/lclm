"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import random
import re

from jinja2 import Template
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Prompt rendering utilities

VALID_SHOT_LAYOUTS = ("normal", "one_line")


def _validate_shot_layout(shot_layout):
    if shot_layout not in VALID_SHOT_LAYOUTS:
        raise ValueError(
            f"shot_layout must be one of {VALID_SHOT_LAYOUTS}, got {shot_layout!r}"
        )


def _one_line_text(text):
    """Collapse newline boundaries while preserving ordinary in-line spacing."""
    return re.sub(r"[ \t]*\r?\n[ \t]*", " ", str(text)).strip()


def _one_line_join(*parts):
    pieces = [_one_line_text(p) for p in parts]
    return " ".join(p for p in pieces if p)

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None,
                      shot_layout="normal"):
    """Render complete prompts for a multiple choice question"""
    _validate_shot_layout(shot_layout)
    if shot_layout == "one_line":
        fewshot_examples = fewshot_examples or []
        prefix_lines = [
            _one_line_join(
                example["query"],
                continuation_delimiter,
                example["choices"][example["gold"]],
            )
            for example in fewshot_examples
        ]
        prefix = "\n".join(prefix_lines)
        prompts = [
            _one_line_join(item["query"], continuation_delimiter, choice)
            for choice in item["choices"]
        ]
        if prefix:
            prompts = [prefix + "\n" + prompt for prompt in prompts]
        return prompts

    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None,
                          shot_layout="normal"):
    """Render complete prompts for a schema question"""
    _validate_shot_layout(shot_layout)
    if shot_layout == "one_line":
        fewshot_examples = fewshot_examples or []
        prefix_lines = [
            _one_line_join(
                example["context_options"][example["gold"]],
                continuation_delimiter,
                example["continuation"],
            )
            for example in fewshot_examples
        ]
        prefix = "\n".join(prefix_lines)
        prompts = [
            _one_line_join(context_option, continuation_delimiter, item["continuation"])
            for context_option in item["context_options"]
        ]
        if prefix:
            prompts = [prefix + "\n" + prompt for prompt in prompts]
        return prompts

    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None,
                      shot_layout="normal"):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    _validate_shot_layout(shot_layout)
    if shot_layout == "one_line":
        fewshot_examples = fewshot_examples or []
        prefix_lines = [
            _one_line_join(
                example["context"],
                continuation_delimiter,
                example["continuation"],
            )
            for example in fewshot_examples
        ]
        target_prefix = _one_line_join(item["context"], continuation_delimiter)
        target_with = _one_line_join(
            item["context"], continuation_delimiter, item["continuation"]
        )
        prefix = "\n".join(prefix_lines)
        if prefix:
            return [prefix + "\n" + target_prefix, prefix + "\n" + target_with]
        return [target_prefix, target_with]

    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids, rope_idx=None):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.

    `rope_idx` (optional BxT): when provided, passed through to `model.forward`
    for models that consume it (MultithreadLM). Vanilla SimpleGPT / HF models
    ignore the kwarg and use their own positional scheme.
    """
    batch_size, seq_len = input_ids.size()
    if rope_idx is not None:
        outputs = model(input_ids, rope_idx=rope_idx)
    else:
        outputs = model(input_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta,
                     mt_encoding="v4_tail", v4_parallel_cap=None,
                     shot_layout="normal"):
    """Evaluate a single example, return True if correct, False otherwise.

    For MultithreadLM, `mt_encoding` controls the path:
      - "v4_tail" — MT-faithful v4 layout (default; final source block
                    scored as the row-last tail-SOT line).
      - "arange"  — DIAGNOSTIC: bypass the MT-faithful encoder entirely
                    and treat the MT model like a vanilla AR LM with
                    contiguous arange(T) positions. This is the OLD code
                    path that ran before `core_mt.py` landed — it triggers
                    MT's rope-arange fallback (`multithread_lm.py:305-309`)
                    and is OOD past row position 256. Useful as a control:
                    compare v4_tail-vs-arange to isolate whether the
                    MT-faithful encoder is helping or hurting per task.

    (The legacy v3 "chain"/"bot_k"/"train_dist" encodings were removed in
    the v4-only cleanup.)

    `v4_parallel_cap` (int | None) is forwarded into `compile_doc_v4` under
    `mt_encoding="v4_tail"` so the eval row layout matches the cap the
    checkpoint was trained with.

    For non-MT models `mt_encoding` is ignored (vanilla AR path).
    """
    # MT dispatch — delayed import to avoid a cycle (core_mt imports
    # render_prompts_* / stack_sequences / find_common_length from here).
    from nanochat.models.multithread_lm import MultithreadLM
    if isinstance(model, MultithreadLM) and mt_encoding != "arange":
        from nanochat.eval.core_mt import evaluate_example_mt
        return evaluate_example_mt(idx, model, tokenizer, data, device,
                                   task_meta, encoding=mt_encoding,
                                   v4_parallel_cap=v4_parallel_cap,
                                   shot_layout=shot_layout)
    # Fall through: vanilla AR path. For MT models with mt_encoding="arange"
    # this gives them model(input_ids) with no rope_idx — MT's forward
    # supplies arange(T) as fallback. Same behavior as pre-core_mt.py code.

    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item)
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(
            item, continuation_delimiter, fewshot_examples,
            shot_layout=shot_layout)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(
            item, continuation_delimiter, fewshot_examples,
            shot_layout=shot_layout)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(
            item, continuation_delimiter, fewshot_examples,
            shot_layout=shot_layout)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2)
    # In these cases, we have to truncate sequences to max length and adjust the indices
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:]) # take the last max_tokens tokens
                new_start_idxs.append(s - num_to_crop) # shift the indices down
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t) # keep unchanged
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack up all the sequences into a batch
    pad_token_id = tokenizer.get_bos_token_id() # use BOS as pad token is ok
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward the model, get the autoregressive loss and argmax prediction at each token
    losses, predictions = forward_model(model, input_ids)

    # See if the losses/predictions come out correctly
    if task_type == 'language_modeling':
        # language modeling task is currently always batch size 1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] predict input_ids[i+1] autoregressively
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        # For MC/schema: find the option with lowest average loss
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta, mt_encoding="v4_tail",
                  v4_parallel_cap=None, shot_layout="normal"):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.

    For MultithreadLM MT-faithful encodings, items whose candidate text
    contains a newline are pre-filtered (decision §10.5 of
    `design/eval/core-mt-eval.md`; 7/89500 items in the 22-task CORE bundle,
    all LAMBADA formatting noise). The denominator is the post-filter
    count, so the score is the mean over evaluated items only. The
    diagnostic `mt_encoding="arange"` path intentionally keeps the old
    vanilla-AR behavior, including the original denominator.

    `v4_parallel_cap` (int | None) is forwarded into the v4_tail encoder
    so eval row layout matches the checkpoint's training cap.
    """
    # MT pre-filter: drop multi-line-candidate items so the K=1-chain
    # encoder never has to bisect a candidate across multiple lines.
    from nanochat.models.multithread_lm import MultithreadLM
    if isinstance(model, MultithreadLM) and mt_encoding != "arange":
        from nanochat.eval.core_mt import filter_multiline_items
        data, dropped = filter_multiline_items(data, task_meta['task_type'])
        if dropped > 0:
            # Only print from rank 0; harmless if it slips through under torchrun.
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"  [mt-filter] dropped {dropped} multi-line-candidate items "
                      f"(task_type={task_meta['task_type']})")

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    counted = torch.zeros(len(data), dtype=torch.float32, device=device)
    # stride the examples to each rank
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device,
                                      task_meta, mt_encoding=mt_encoding,
                                      v4_parallel_cap=v4_parallel_cap,
                                      shot_layout=shot_layout)
        if is_correct is None:
            continue  # item was skipped (e.g., MT rope cache overflow guard)
        correct[idx] = float(is_correct)
        counted[idx] = 1.0
    # sync results across all the processes if running distributed
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(counted, op=dist.ReduceOp.SUM)
    # compute the mean over items actually evaluated
    total_counted = counted.sum().item()
    if total_counted == 0:
        return 0.0
    mean_correct = correct.sum().item() / total_counted
    return mean_correct
