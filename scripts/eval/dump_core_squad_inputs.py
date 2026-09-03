"""
Dump the exact CORE SQuAD input rows sent to vanilla or MT-v4 eval.

This is a diagnostic for SQuAD prompt/layout issues. It intentionally calls the
same CORE prompt renderer and encoder functions used by evaluation:

- vanilla SimpleGPT: nanochat.eval.core.render_prompts_lm + batch_sequences_lm
- MT-v4: nanochat.eval.core.render_prompts_lm + batch_sequences_mt_lm

For MT-v4, the JSON also includes the v4 row metadata produced by the production
split/build/compile helpers, so the reordered row can be inspected by position,
rope index, thread, stage, and local step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.eval.core import batch_sequences_lm, render_prompts_lm
from nanochat.eval.core_mt import batch_sequences_mt_lm
from nanochat.eval.core_mt import _candidate_token_count, _get_nl_class, _get_specials
from nanochat.data.multithread_dataloader import split_tokens_into_blocks, build_doc_v4
from nanochat.data.multithread_layout import compile_doc_v4
from scripts.eval.core_eval_data import (
    load_core_task,
    take_core_subsample,
    fewshot_examples_for,
)


_SQUAD_FINAL_QA_RE = re.compile(r'\nQuestion: ([^\n]*)\nAnswer:\s*$')


def _rewrite_squad_qa_line(ctx: str) -> str:
    out, n = _SQUAD_FINAL_QA_RE.subn(r'\nQuestion: \1 Answer:', ctx)
    return out if n else ctx


def _load_checkpoint_meta(model_tag: str, step: int | None) -> dict[str, Any]:
    ckpt_dir = Path(get_base_dir()) / "base_checkpoints" / model_tag
    if step is None:
        metas = sorted(ckpt_dir.glob("meta_*.json"))
        if not metas:
            raise FileNotFoundError(f"no meta_*.json found under {ckpt_dir}")
        path = metas[-1]
    else:
        path = ckpt_dir / f"meta_{step:06d}.json"
    with path.open(encoding="utf-8") as f:
        meta = json.load(f)
    meta["_meta_path"] = str(path)
    return meta


def _safe_decode(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    try:
        return tokenizer.decode(ids)
    except Exception as exc:
        return f"<decode error: {exc}>"


def _piece(tokenizer, token_id: int, specials_by_id: dict[int, str]) -> str:
    if token_id in specials_by_id:
        return specials_by_id[token_id]
    return _safe_decode(tokenizer, [int(token_id)])


def _specials_by_id(tokenizer) -> dict[int, str]:
    out: dict[int, str] = {}
    for name in tokenizer.get_special_tokens():
        tid = tokenizer.encode_special(name)
        if tid is not None:
            out[int(tid)] = name
    return out


def _extract_question(context: str) -> str:
    marker = "\nQuestion:"
    if marker in context:
        return context.rsplit(marker, 1)[1].split("\nAnswer:", 1)[0].strip()
    return ""


def _row_records(
    tokenizer,
    tokens: list[int],
    *,
    start_idx: int,
    end_idx: int,
    rope_idx: list[int] | None = None,
    thread_idx: list[int] | None = None,
    stage_idx: list[int] | None = None,
    local_step: list[int] | None = None,
) -> list[dict[str, Any]]:
    specials = _specials_by_id(tokenizer)
    rows = []
    for pos, tok in enumerate(tokens):
        rec: dict[str, Any] = {
            "pos": pos,
            "token_id": int(tok),
            "piece": _piece(tokenizer, int(tok), specials),
            "is_answer_token": bool(start_idx <= pos < end_idx),
            "predicts_answer_token": bool(start_idx - 1 <= pos < end_idx - 1),
        }
        if rope_idx is not None:
            rec["rope_idx"] = int(rope_idx[pos])
        if thread_idx is not None:
            rec["thread_idx"] = int(thread_idx[pos])
        if stage_idx is not None:
            rec["stage_idx"] = int(stage_idx[pos])
        if local_step is not None:
            rec["local_step"] = int(local_step[pos])
        rows.append(rec)
    return rows


def _window(rows: list[dict[str, Any]], lo: int, hi: int) -> list[dict[str, Any]]:
    lo = max(0, lo)
    hi = min(len(rows), hi)
    return rows[lo:hi]




def _format_range(start: int | None, end: int | None, *, end_exclusive: bool = False) -> str:
    if start is None or end is None:
        return ""
    if end_exclusive:
        end -= 1
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _compact_spans(values: list[int]) -> str:
    if not values:
        return ""
    vals = sorted(set(int(v) for v in values))
    spans = []
    start = prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
            continue
        spans.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = v
    spans.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(spans)


def _row_order_text(rows: list[dict[str, Any]]) -> str:
    return "".join(str(r["piece"]) for r in rows)


def _segment_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    return {
        "pos_start": int(first["pos"]),
        "pos_end": int(last["pos"]) + 1,
        "token_count": len(rows),
        "token_ids": [int(r["token_id"]) for r in rows],
        "text": _row_order_text(rows),
        "has_answer_token": any(bool(r["is_answer_token"]) for r in rows),
        "has_answer_prediction": any(bool(r["predicts_answer_token"]) for r in rows),
        "rope_start": first.get("rope_idx"),
        "rope_end": (int(last["rope_idx"]) + 1) if "rope_idx" in last else None,
        "thread_idx": first.get("thread_idx"),
        "stage_idx": first.get("stage_idx"),
        "local_start": first.get("local_step"),
        "local_end": (int(last["local_step"]) + 1) if "local_step" in last else None,
    }


def _row_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the exact encoded row into readable contiguous row-order runs."""
    if not rows:
        return []

    def can_merge(prev: dict[str, Any], cur: dict[str, Any]) -> bool:
        if int(cur["pos"]) != int(prev["pos"]) + 1:
            return False
        for key in ("thread_idx", "stage_idx"):
            if prev.get(key) != cur.get(key):
                return False
        for key in ("local_step", "rope_idx"):
            if key in prev and key in cur and int(cur[key]) != int(prev[key]) + 1:
                return False
        return True

    segments: list[dict[str, Any]] = []
    current = [rows[0]]
    for row in rows[1:]:
        if can_merge(current[-1], row):
            current.append(row)
        else:
            segments.append(_segment_record(current))
            current = [row]
    segments.append(_segment_record(current))
    return segments


def _thread_logical_lines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group row tokens by thread/local step for comparison with row order."""
    if not rows or "thread_idx" not in rows[0]:
        return []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["thread_idx"]), []).append(row)

    out = []
    for thread, rs in sorted(grouped.items()):
        ordered = sorted(rs, key=lambda r: int(r.get("local_step", r["pos"])))
        out.append({
            "thread_idx": thread,
            "stage_idx": ordered[0].get("stage_idx"),
            "row_pos_spans": _compact_spans([int(r["pos"]) for r in ordered]),
            "rope_spans": _compact_spans([int(r["rope_idx"]) for r in ordered if "rope_idx" in r]),
            "local_spans": _compact_spans([int(r["local_step"]) for r in ordered if "local_step" in r]),
            "token_count": len(ordered),
            "text": _row_order_text(ordered),
        })
    return out

def _summarize_windows(rows: list[dict[str, Any]], start_idx: int, end_idx: int) -> dict[str, Any]:
    return {
        "first_40": _window(rows, 0, 40),
        "answer_window": _window(rows, start_idx - 12, end_idx + 12),
        "last_80": _window(rows, len(rows) - 80, len(rows)),
    }


def _v4_meta_for_prompt(tokenizer, prompts: list[str], candidate_token_count: int, rope_stride: int):
    full_tokens = tokenizer.encode(prompts)[1]
    nl_class = _get_nl_class(tokenizer)
    sp = _get_specials(tokenizer)
    blocks = split_tokens_into_blocks(full_tokens, nl_class)
    doc = build_doc_v4(blocks, warmup_threshold=16)
    meta = compile_doc_v4(
        doc,
        rope_stride=rope_stride,
        sot_id=sp["sot_id"],
        sot_tail_id=sp["sot_tail_id"],
        bos_id=sp["bos_id"],
    )
    return blocks, doc, meta


def dump_one(args) -> dict[str, Any]:
    all_data, task_meta = load_core_task("squad")
    task_meta["num_fewshot"] = args.squad_num_fewshot
    if args.squad_rewrite == "qa_line":
        for item in all_data:
            item["context"] = _rewrite_squad_qa_line(item["context"])
    elif args.squad_rewrite != "none":
        raise ValueError(f"unsupported squad_rewrite: {args.squad_rewrite}")
    data = take_core_subsample(all_data, args.num_items)
    if not (0 <= args.sample_index < len(data)):
        raise IndexError(f"sample_index {args.sample_index} outside 0..{len(data)-1}")

    model_meta = _load_checkpoint_meta(args.model_tag, args.step)
    tokenizer = get_tokenizer(name=model_meta.get("tokenizer_name"))
    item = data[args.sample_index]
    mt_path = args.model_kind == "mt"
    fewshot = fewshot_examples_for(args.sample_index, data, task_meta, mt_path=mt_path)
    prompts = render_prompts_lm(
        item, task_meta["continuation_delimiter"], fewshot,
        shot_layout=args.core_shot_layout)

    result: dict[str, Any] = {
        "task": "squad",
        "model_kind": args.model_kind,
        "model_tag": args.model_tag,
        "step": model_meta.get("step"),
        "meta_path": model_meta.get("_meta_path"),
        "tokenizer_name": model_meta.get("tokenizer_name"),
        "mt_encoding": args.mt_encoding if mt_path else None,
        "squad_num_fewshot": args.squad_num_fewshot,
        "squad_rewrite": args.squad_rewrite,
        "core_shot_layout": args.core_shot_layout,
        "shuffle_seed": 1337,
        "num_items_after_shuffle": len(data),
        "sample_index": int(item["_sample_index"]),
        "source_jsonl_line": int(item["_source_jsonl_line"]),
        "question": _extract_question(item.get("context", "")),
        "gold_answer": item.get("continuation", ""),
        "fewshot": [
            {
                "source_jsonl_line": int(ex.get("_source_jsonl_line", -1)),
                "question": _extract_question(ex.get("context", "")),
                "answer": ex.get("continuation", ""),
            }
            for ex in fewshot
        ],
        "prompt_without": prompts[0],
        "prompt_with": prompts[1],
    }

    if args.model_kind == "simple":
        rows_tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
        tokens = rows_tokens[0]
        start_idx = int(start_idxs[0])
        end_idx = int(end_idxs[0])
        row = _row_records(tokenizer, tokens, start_idx=start_idx, end_idx=end_idx)
        result.update({
            "encoder": "batch_sequences_lm",
            "row_length": len(tokens),
            "answer_start_idx": start_idx,
            "answer_end_idx": end_idx,
            "answer_token_ids": tokens[start_idx:end_idx],
            "answer_token_text": [_safe_decode(tokenizer, [t]) for t in tokens[start_idx:end_idx]],
            "row": row,
            "row_order_text": _row_order_text(row),
            "row_order_segments": _row_segments(row),
            "thread_logical_lines": _thread_logical_lines(row),
            "windows": _summarize_windows(row, start_idx, end_idx),
        })
    else:
        encoded = batch_sequences_mt_lm(
            tokenizer,
            prompts,
            rope_stride=args.rope_stride,
            encoding=args.mt_encoding,
        )
        if encoded is None:
            result["skipped"] = True
            result["skip_reason"] = "MT encoder rejected item"
            return result
        rows_tokens, rows_rope, start_idxs, end_idxs = encoded
        tokens = rows_tokens[0]
        ropes = rows_rope[0]
        start_idx = int(start_idxs[0])
        end_idx = int(end_idxs[0])
        full_tokens_per_prompt = tokenizer.encode(prompts)
        candidate_token_count = _candidate_token_count(full_tokens_per_prompt[1], len(full_tokens_per_prompt[0]))
        blocks = None
        meta = None
        thread_idx = stage_idx = local_step = None
        if args.mt_encoding == "v4_tail":
            blocks, doc, meta = _v4_meta_for_prompt(
                tokenizer, prompts, candidate_token_count, args.rope_stride)
            assert list(meta.tokens) == list(tokens), "diagnostic v4 meta row differs from batch_sequences_mt_lm"
            assert list(meta.rope_idx) == list(ropes), "diagnostic v4 rope row differs from batch_sequences_mt_lm"
            thread_idx = list(meta.thread_idx)
            stage_idx = list(meta.stage_idx)
            local_step = list(meta.local_step)
            result["v4_doc_summary"] = {
                "num_blocks": len(blocks),
                "block_lengths": [len(b) for b in blocks],
                "warmup_lengths": [p.length for p in doc.warmup],
                "middle_lengths": [p.length for p in doc.middle],
                "tail_length": doc.tail.length if doc.tail is not None else None,
                "tail_thread_idx": len(doc.warmup) + len(doc.middle) if doc.tail is not None else None,
            }
        row = _row_records(
            tokenizer,
            tokens,
            start_idx=start_idx,
            end_idx=end_idx,
            rope_idx=ropes,
            thread_idx=thread_idx,
            stage_idx=stage_idx,
            local_step=local_step,
        )
        result.update({
            "encoder": f"batch_sequences_mt_lm/{args.mt_encoding}",
            "row_length": len(tokens),
            "rope_max": max(ropes),
            "answer_start_idx": start_idx,
            "answer_end_idx": end_idx,
            "candidate_token_count": candidate_token_count,
            "answer_token_ids": tokens[start_idx:end_idx],
            "answer_token_text": [_safe_decode(tokenizer, [t]) for t in tokens[start_idx:end_idx]],
            "row": row,
            "row_order_text": _row_order_text(row),
            "row_order_segments": _row_segments(row),
            "thread_logical_lines": _thread_logical_lines(row),
            "windows": _summarize_windows(row, start_idx, end_idx),
        })
    return result




def _md_cell(value: Any, max_len: int | None = None) -> str:
    text = str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    text = text.replace("|", "\\|").replace("`", "\\`")
    if max_len is not None and len(text) > max_len:
        text = text[:max_len] + "..."
    return text

def write_markdown(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# CORE SQuAD Input Dump\n\n")
        for key in [
            "model_kind", "model_tag", "step", "tokenizer_name", "mt_encoding",
            "squad_num_fewshot", "squad_rewrite", "core_shot_layout", "sample_index", "source_jsonl_line", "question",
            "gold_answer", "encoder", "row_length", "answer_start_idx", "answer_end_idx",
            "rope_max",
        ]:
            if key in result:
                f.write(f"- {key}: `{result[key]}`\n")
        if "v4_doc_summary" in result:
            f.write(f"- v4_doc_summary: `{json.dumps(result['v4_doc_summary'])}`\n")
        f.write("\n## Few-Shot Examples\n\n")
        for i, ex in enumerate(result.get("fewshot", [])):
            f.write(f"{i}. line `{ex['source_jsonl_line']}` answer `{ex['answer']}` question `{ex['question']}`\n")
        f.write("\n## Prompt Without Continuation Tail\n\n")
        tail = result.get("prompt_without", "")[-3000:]
        f.write("```text\n" + tail + "\n```\n")

        f.write("\n## Exact Row-Order Input Text\n\n")
        f.write("This is `''.join(piece)` over the encoded row after the production reorder.\n\n")
        f.write("````text\n" + result.get("row_order_text", "") + "\n````\n")

        segments = result.get("row_order_segments", [])
        if segments:
            f.write("\n## Exact Row-Order Segments\n\n")
            f.write("Contiguous row positions with the same thread/stage and consecutive local/RoPE indices are collapsed. Text cells are truncated; JSON keeps the full text.\n\n")
            f.write("| pos | n | rope | thread | stage | local | ans | pred_ans | text |\n")
            f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
            for seg in segments:
                f.write(
                    f"| {_format_range(seg['pos_start'], seg['pos_end'], end_exclusive=True)} "
                    f"| {seg['token_count']} "
                    f"| {_format_range(seg.get('rope_start'), seg.get('rope_end'), end_exclusive=True)} "
                    f"| {seg.get('thread_idx', '')} "
                    f"| {seg.get('stage_idx', '')} "
                    f"| {_format_range(seg.get('local_start'), seg.get('local_end'), end_exclusive=True)} "
                    f"| {int(seg['has_answer_token'])} "
                    f"| {int(seg['has_answer_prediction'])} "
                    f"| `{_md_cell(seg['text'], 360)}` |\n"
                )

        logical = result.get("thread_logical_lines", [])
        if logical:
            f.write("\n## Logical Threads\n\n")
            f.write("Same tokens grouped by thread and sorted by local step. This is only a readability view; the row-order sections above are what the model receives.\n\n")
            f.write("| thread | stage | row_pos | rope | local | n | text |\n")
            f.write("|---:|---:|---|---|---|---:|---|\n")
            for line in logical:
                f.write(
                    f"| {line['thread_idx']} | {line.get('stage_idx', '')} "
                    f"| `{line['row_pos_spans']}` | `{line['rope_spans']}` "
                    f"| `{line['local_spans']}` | {line['token_count']} "
                    f"| `{_md_cell(line['text'], 360)}` |\n"
                )

        f.write("\n## Row Windows\n\n")
        for name, rows in result.get("windows", {}).items():
            f.write(f"### {name}\n\n")
            f.write("| pos | token | piece | ans | pred_ans | rope | thread | stage | local |\n")
            f.write("|---:|---:|---|---|---|---:|---:|---:|---:|\n")
            for r in rows:
                piece = str(r["piece"]).replace("\n", "\\n").replace("|", "\\|")
                f.write(
                    f"| {r['pos']} | {r['token_id']} | `{piece}` | "
                    f"{int(r['is_answer_token'])} | {int(r['predicts_answer_token'])} | "
                    f"{r.get('rope_idx', '')} | {r.get('thread_idx', '')} | "
                    f"{r.get('stage_idx', '')} | {r.get('local_step', '')} |\n"
                )
            f.write("\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-kind", choices=["simple", "mt"], required=True)
    p.add_argument("--model-tag", required=True)
    p.add_argument("--step", type=int, default=5952)
    p.add_argument("--squad-num-fewshot", type=int, choices=[0, 1, 2, 4, 8, 10], required=True)
    p.add_argument("--squad-rewrite", default="none", choices=["none", "qa_line"])
    p.add_argument("--core-shot-layout", default="normal", choices=["normal", "one_line"])
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--num-items", type=int, default=-1,
                   help="CORE shuffled items to keep before indexing; -1 means full SQuAD")
    p.add_argument("--mt-encoding", default="v4_tail",
                   choices=["v4_tail"])
    p.add_argument("--rope-stride", type=int, default=256)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = dump_one(args)
    base = Path(get_base_dir()) / "dump"
    base.mkdir(parents=True, exist_ok=True)
    enc = args.mt_encoding if args.model_kind == "mt" else "ar"
    rewrite = "" if args.squad_rewrite == "none" else f"_{args.squad_rewrite}"
    layout = "" if args.core_shot_layout == "normal" else f"_shot{args.core_shot_layout}"
    stem = (
        f"core_squad_input_dump_{args.model_kind}_{enc}_"
        f"{args.squad_num_fewshot}shot{rewrite}{layout}_idx{args.sample_index}_s{args.step:06d}"
    )
    json_path = Path(args.output_json) if args.output_json else base / f"{stem}.json"
    md_path = Path(args.output_md) if args.output_md else base / f"{stem}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_markdown(md_path, result)
    print(f"JSON written to: {json_path}")
    print(f"Markdown written to: {md_path}")
    print(f"row_length={result.get('row_length')} answer_span=[{result.get('answer_start_idx')}, {result.get('answer_end_idx')})")
    if "v4_doc_summary" in result:
        print("v4_doc_summary=" + json.dumps(result["v4_doc_summary"]))


if __name__ == "__main__":
    main()
