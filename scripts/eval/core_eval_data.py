"""Shared CORE eval-data helpers (tokenizer/layout-agnostic).

Loading a CORE task's items + meta, deterministic subsampling, and
few-shot example selection. Factored out of the (now-removed) v3
`dump_core_coqa_answer_compare.py` so the v4 dump tools can reuse them
without pulling in any v3 layout code.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import yaml

from nanochat.common import download_file_with_lock, get_base_dir
from nanochat.eval.core_mt import _item_has_multiline_candidate
from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle


def ensure_eval_bundle() -> str:
    base = get_base_dir()
    bundle = os.path.join(base, "eval_bundle")
    if not os.path.exists(bundle):
        download_file_with_lock(
            EVAL_BUNDLE_URL,
            "eval_bundle.zip",
            postprocess_fn=place_eval_bundle,
        )
    return bundle


def load_core_task(task_label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = ensure_eval_bundle()
    config_path = os.path.join(bundle, "core.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task = None
    for candidate in config["icl_tasks"]:
        if candidate["label"] == task_label:
            task = candidate
            break
    if task is None:
        labels = [t["label"] for t in config["icl_tasks"]]
        raise ValueError(f"CORE task {task_label!r} not found. Available: {labels}")

    task_meta = {
        "label": task["label"],
        "task_type": task["icl_task_type"],
        "dataset_uri": task["dataset_uri"],
        "num_fewshot": task["num_fewshot"][0],
        "continuation_delimiter": task.get("continuation_delimiter", " "),
    }

    data_path = os.path.join(bundle, "eval_data", task_meta["dataset_uri"])
    data: list[dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_source_jsonl_line"] = line_no
            data.append(item)
    return data, task_meta


def take_core_subsample(data: list[dict[str, Any]], num_items: int) -> list[dict[str, Any]]:
    shuffled = list(data)
    random.Random(1337).shuffle(shuffled)
    if num_items > 0:
        shuffled = shuffled[:num_items]
    for sample_idx, item in enumerate(shuffled):
        item["_sample_index"] = sample_idx
    return shuffled


def fewshot_examples_for(
    idx: int,
    data: list[dict[str, Any]],
    task_meta: dict[str, Any],
    *,
    mt_path: bool,
) -> list[dict[str, Any]]:
    num_fewshot = task_meta["num_fewshot"]
    if num_fewshot <= 0:
        return []

    rng = random.Random(1234 + idx)
    if mt_path:
        available = [
            i for i in range(len(data))
            if i != idx and not _item_has_multiline_candidate(data[i], task_meta["task_type"])
        ]
    else:
        available = [i for i in range(len(data)) if i != idx]
    if len(available) < num_fewshot:
        return []
    return [data[i] for i in rng.sample(available, num_fewshot)]
