"""Build 10-shot CORE prompt-format ablation table from copied CSV artifacts."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "paper" / "eval_bundle_d24" / "phase2_core"

RUNS = (
    (
        "6.2B",
        "10-shot",
        "Vanilla AR",
        "artifacts/core_csv/r8_simple_10shot_two_line_base_model_005952.csv",
        "artifacts/format_ablation/r8_simple_10shot_same_line_base_model_005952_shotone_line.csv",
    ),
    (
        "6.2B",
        "10-shot",
        "MT-LM",
        "artifacts/core_csv/r8_mt_v4_10shot_two_line_base_model_005952_v4_tail.csv",
        "artifacts/format_ablation/r8_mt_v4_10shot_same_line_base_model_005952_v4_tail_shotone_line.csv",
    ),
    (
        "15.5B",
        "10-shot",
        "Vanilla AR",
        "artifacts/core_csv/r20_simple_10shot_two_line_base_model_014800.csv",
        "artifacts/format_ablation/r20_simple_10shot_same_line_base_model_014800_shotone_line.csv",
    ),
    (
        "15.5B",
        "10-shot",
        "MT-LM",
        "artifacts/core_csv/r20_mt_v4_10shot_two_line_base_model_014800_v4_tail.csv",
        "artifacts/format_ablation/r20_mt_v4_10shot_same_line_base_model_014800_v4_tail_shotone_line.csv",
    ),
)


def _core_score(rel_path: str) -> float:
    with (CORE / rel_path).open(newline="") as f:
        for row in csv.reader(f):
            row = [cell.strip() for cell in row]
            if row and row[0] == "CORE":
                return float(row[-1])
    raise ValueError(f"CORE row not found in {rel_path}")


def main() -> None:
    out_path = CORE / "tables" / "core_format_ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "training_budget",
                "regime",
                "model",
                "two_line",
                "same_line",
                "delta",
                "two_line_source",
                "same_line_source",
                "status",
            ]
        )
        for budget, regime, model, two_rel, same_rel in RUNS:
            two = _core_score(two_rel)
            same = _core_score(same_rel)
            writer.writerow(
                [
                    budget,
                    regime,
                    model,
                    f"{two:.3f}",
                    f"{same:.3f}",
                    f"{same - two:+.3f}",
                    two_rel,
                    same_rel,
                    "generated_from_source_csv",
                ]
            )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
