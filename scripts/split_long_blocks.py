"""
Block-MT data preprocessing — run-preserving long-block splitter.

Reads parquet shards from the active dataset (NANOCHAT_DATASET, e.g.
'climbmix'), and per design/blockmt/blockmt-newline-preserve-runs.md:

  1. Strips leading `\\n`/`\\r` runs from each doc (avoids degenerate
     all-newline parent paragraphs).
  2. Truncates any consecutive `\\n`/`\\r` run longer than
     `MAX_RUN_TOKENS=16` (data-wash; anything longer is corruption /
     trailing whitespace artifact, indistinguishable to humans from
     "16 blank lines"). Guarantees every newline run fits inside one
     per-block token budget without further splitting.
  3. Splits doc text on `[\\n\\r]+` runs into content/separator pairs.
  4. For each `(content, separator)` pair: if total tokens
     (content + separator) ≤ MAX_BLOCK_TOKENS, emit verbatim. Otherwise
     recursively splits the content at sentence-ish punctuation near
     midpoint, joining intermediate sub-blocks with synthetic `\\n` and
     attaching the original separator after the last sub-block.
  5. Does NOT append `\\n` at doc end — docs may end mid-content
     (Q7 relaxation; last block of a doc can have any c_L class, the
     `<eot>` / `<bot-K>` / `<bos>` override at c_L fires regardless).

Per-sub-block token-budget constraints satisfied by the splitter:
  - intermediate sub-block: tokens(sub) + 1 ≤ MAX_BLOCK_TOKENS
  - last sub-block:        tokens(sub) + len(sep_tokens) ≤ MAX_BLOCK_TOKENS

The default cap `MAX_BLOCK_TOKENS=254` corresponds to the layout
constraint `stride > max_block_length + 1` with `rope_stride=256` (one
slot for the paragraph anchor, one slot for the next stage's `<bot-K>`
opener in the decision-thread case).

Multi-worker `Pool` (spawn ctx) — one shard per worker, each loads its
own tokenizer copy. Output written atomically (`.tmp` then rename).

Example:
    python -m scripts.split_long_blocks \\
        --tokenizer=nanochat_64k_mt2 \\
        --workers=32 \\
        --max-tokens=254
"""

import argparse
import multiprocessing as mp
import os
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.data.dataset import list_parquet_files, get_active_spec
from nanochat.tokenizer import RustBPETokenizer


# Sentence-ending punctuation (ASCII + CJK fullwidth + ellipsis).
# Deliberately excludes `,` and `:` — comma-splitting produces nonsense
# clause-only sub-blocks for prose; colon is often non-terminal (URLs,
# time formats).
PUNCT_RE = re.compile(r"[.!?;。！？；…]")

# Per-block token cap. With rope_stride=256 the layout constraint
# `stride > max_block_length + 1` gives L ≤ 254 (content + trailing run,
# anchor not counted).
DEFAULT_MAX_TOKENS = 254

# Cap any single consecutive `\n`/`\r` run at 16 tokens. Real prose runs
# are 1-4; corruption runs are 100+. Truncating at 16 preserves all
# perceivable typography and guarantees `content + run ≤ MAX_BLOCK_TOKENS`
# can be enforced by the content splitter without needing to split the
# run itself (which the dataloader's Q1 walker would coalesce anyway).
MAX_RUN_TOKENS = 16

# Run-detection regex for splitting and truncation. Matches mixed
# `\n`/`\r` runs as one group; e.g. `\r\n\r\n` is one run of length 4.
RUN_RE = re.compile(r"[\n\r]+")

# Search ±25% of midpoint for a punctuation boundary.
WINDOW_FRAC = 0.25


def split_long_block(text, tokenizer, max_tokens=DEFAULT_MAX_TOKENS,
                     right_sep_tokens=1, punct_re=PUNCT_RE):
    """Recursively split a content text at sentence-ish punctuation near
    midpoint until every sub-block fits the per-block cap, accounting
    for the trailing-separator budget on the rightmost sub-block.

    Returns a flat list of sub-block texts. Caller joins intermediate
    sub-blocks with `\\n` and appends the original separator after the
    last one.

    Per-output-sub-block invariant:
      - intermediate (followed by synthetic `\\n`):
            tokens(sub) + 1            ≤ max_tokens
      - last (followed by original separator):
            tokens(sub) + right_sep_tokens ≤ max_tokens
    """
    if not text:
        return [text]
    # Fast path: char count + right_sep ≤ max_tokens implies the same
    # for token count (BPE never produces more tokens than chars).
    if len(text) + right_sep_tokens <= max_tokens:
        return [text]
    # Precise check
    n_tokens = len(tokenizer.encode(text))
    if n_tokens + right_sep_tokens <= max_tokens:
        return [text]

    char_mid = len(text) // 2
    window = max(1, int(len(text) * WINDOW_FRAC))
    lo, hi = max(0, char_mid - window), min(len(text), char_mid + window)

    best_pos = -1
    for m in punct_re.finditer(text, lo, hi):
        best_pos = m.end()  # last match in window

    if best_pos < 0:
        best_pos = char_mid
    if best_pos == 0 or best_pos == len(text):
        best_pos = char_mid

    left = text[:best_pos]
    right = text[best_pos:]
    # Left is intermediate → its right-sep budget is 1 (synthetic `\n`).
    # Right is rightmost in this scope → inherits caller's budget.
    return (
        split_long_block(left, tokenizer, max_tokens, right_sep_tokens=1,
                         punct_re=punct_re)
        + split_long_block(right, tokenizer, max_tokens,
                           right_sep_tokens=right_sep_tokens,
                           punct_re=punct_re)
    )


def preprocess_doc(doc_text, tokenizer, max_tokens=DEFAULT_MAX_TOKENS,
                   max_run_tokens=MAX_RUN_TOKENS):
    """Apply run-preserving block preprocessing to a doc text. See
    module docstring for the full pipeline.
    """
    if not doc_text:
        return doc_text

    # 1. Strip leading newline run — keeps the dataloader's first block
    #    (the parent paragraph in the multithread layout) from being a
    #    degenerate all-newline block.
    doc_text = doc_text.lstrip("\n\r")
    if not doc_text:
        return doc_text

    # 2. Cap pathologically long newline runs (data wash).
    if max_run_tokens > 0:
        doc_text = RUN_RE.sub(
            lambda m: m.group(0)[:max_run_tokens]
                       if len(m.group(0)) > max_run_tokens
                       else m.group(0),
            doc_text,
        )

    # 3. Split into content/separator alternation. After lstrip the
    #    first element is non-empty content; thereafter parts alternates
    #    content, separator, content, separator, ..., possibly ending
    #    with an empty content if doc_text ended with a separator.
    parts = re.split(r"([\n\r]+)", doc_text)

    out = []
    n = len(parts)
    for i in range(0, n, 2):
        content = parts[i]
        separator = parts[i + 1] if i + 1 < n else ""
        if not content:
            # Trailing empty content slot after the last separator —
            # just emit the separator (already pushed by the previous
            # iteration's right side; here separator is "" since i+1
            # is out of range, so this is a no-op except for safety).
            if separator:
                out.append(separator)
            continue
        # Fast-filter: short content guarantees small token count.
        # If `len(content) + len(separator) ≤ max_tokens` then BPE
        # cannot produce more tokens than chars, so we're safe.
        if len(content) + len(separator) <= max_tokens:
            out.append(content)
            out.append(separator)
            continue
        # Precise token check
        content_tokens = tokenizer.encode(content)
        sep_token_count = len(tokenizer.encode(separator)) if separator else 0
        if len(content_tokens) + sep_token_count <= max_tokens:
            out.append(content)
            out.append(separator)
            continue
        # Need to split this content piece.
        right_budget = max(1, sep_token_count)
        sub_blocks = split_long_block(
            content, tokenizer, max_tokens,
            right_sep_tokens=right_budget,
        )
        out.append("\n".join(sub_blocks))
        out.append(separator)

    return "".join(out)


# Worker-side state. Each Pool process loads its own tokenizer copy
# via the initializer below.
_TOK = None
_OUT_DIR = None
_MAX_TOKENS = DEFAULT_MAX_TOKENS
_MAX_RUN_TOKENS = MAX_RUN_TOKENS


def _init_worker(tokenizer_dir, out_dir, max_tokens, max_run_tokens):
    global _TOK, _OUT_DIR, _MAX_TOKENS, _MAX_RUN_TOKENS
    _TOK = RustBPETokenizer.from_directory(tokenizer_dir)
    _OUT_DIR = out_dir
    _MAX_TOKENS = max_tokens
    _MAX_RUN_TOKENS = max_run_tokens


def _process_shard(args):
    """Process one parquet shard. Returns (basename, n_docs, n_modified)."""
    in_path, out_basename = args
    pf = pq.ParquetFile(in_path)
    out_path = os.path.join(_OUT_DIR, out_basename + ".tmp")
    n_docs = 0
    n_modified = 0
    writer = None
    try:
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            docs = rg.column("text").to_pylist()
            new_docs = []
            for doc in docs:
                n_docs += 1
                new_doc = preprocess_doc(doc, _TOK, _MAX_TOKENS,
                                         max_run_tokens=_MAX_RUN_TOKENS)
                if new_doc != doc:
                    n_modified += 1
                new_docs.append(new_doc)
            table = pa.table({"text": pa.array(new_docs, type=pa.string())})
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    final_path = os.path.join(_OUT_DIR, out_basename)
    os.rename(out_path, final_path)
    return os.path.basename(in_path), n_docs, n_modified


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", default="nanochat_64k_mt2",
                        help="Tokenizer name under $NANOCHAT_BASE_DIR/tokenizers/. "
                             "Default `nanochat_64k_mt2` includes the new <|eot|>.")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir name under $NANOCHAT_BASE_DIR "
                             "(default: <input data_dir_name>_split)")
    parser.add_argument("--max-shards", type=int, default=-1,
                        help="Limit number of input shards per split (-1 = all)")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel shard workers (default 32)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Per-block token cap (default {DEFAULT_MAX_TOKENS} = stride-2)")
    parser.add_argument("--max-run-tokens", type=int, default=MAX_RUN_TOKENS,
                        help=f"Cap on any single \\n/\\r run (default {MAX_RUN_TOKENS}). "
                             f"Pathological runs are truncated to this many tokens at "
                             f"preprocess. 0 disables the cap (not recommended — breaks "
                             f"the per-block cap for corrupt docs).")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = parser.parse_args()

    base = get_base_dir()
    tokenizer_dir = os.path.join(base, "tokenizers", args.tokenizer)
    spec = get_active_spec()
    in_dir = os.path.join(base, spec.data_dir_name)

    out_dir_name = args.output_dir or (spec.data_dir_name + "_split")
    out_dir = os.path.join(base, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    tasks = []
    seen_basenames = set()
    for split in splits:
        shards = list_parquet_files(split, data_dir=in_dir)
        if args.max_shards > 0:
            shards = shards[:args.max_shards]
        for in_path in shards:
            basename = os.path.basename(in_path)
            # climbmix-style: train + val both point at same set of files
            # (val_is_last_train_shard). Deduplicate.
            if basename in seen_basenames:
                continue
            seen_basenames.add(basename)
            tasks.append((in_path, basename))

    # Resume: skip shards whose final output already exists. The atomic
    # `.tmp → rename` in _process_shard guarantees a present final == a
    # complete write, so existence is a sufficient done-marker.
    n_total = len(tasks)
    tasks = [(p, b) for (p, b) in tasks
             if not os.path.exists(os.path.join(out_dir, b))]
    n_skipped = n_total - len(tasks)

    print(f"Tokenizer:      {args.tokenizer} ({tokenizer_dir})", file=sys.stderr)
    print(f"Input dir:      {in_dir}", file=sys.stderr)
    print(f"Output dir:     {out_dir}", file=sys.stderr)
    print(f"Block cap:      {args.max_tokens}", file=sys.stderr)
    print(f"Run cap:        {args.max_run_tokens}", file=sys.stderr)
    print(f"Workers:        {args.workers}", file=sys.stderr)
    print(f"Shards:         {len(tasks)} to process "
          f"({n_skipped} already done, skipped)", file=sys.stderr)
    print(file=sys.stderr)

    total_docs = 0
    total_modified = 0
    if args.workers <= 1 or len(tasks) <= 1:
        _init_worker(tokenizer_dir, out_dir, args.max_tokens, args.max_run_tokens)
        for task in tasks:
            name, n_docs, n_modified = _process_shard(task)
            total_docs += n_docs
            total_modified += n_modified
            print(f"  {name}: {n_docs:,} docs, {n_modified:,} modified "
                  f"({100*n_modified/max(n_docs,1):.3f}%)", file=sys.stderr)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers,
                      initializer=_init_worker,
                      initargs=(tokenizer_dir, out_dir, args.max_tokens,
                                args.max_run_tokens)) as pool:
            for name, n_docs, n_modified in pool.imap_unordered(_process_shard, tasks):
                total_docs += n_docs
                total_modified += n_modified
                print(f"  {name}: {n_docs:,} docs, {n_modified:,} modified "
                      f"({100*n_modified/max(n_docs,1):.3f}%)", file=sys.stderr)

    print(file=sys.stderr)
    print(f"Total: {total_docs:,} docs, {total_modified:,} modified "
          f"({100*total_modified/max(total_docs,1):.3f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
