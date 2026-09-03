"""Build the D24 line-start slice table from bundled eval JSONs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "paper" / "eval_bundle_d24" / "phase1_bpb_loss"

RUNS = (
    (
        "6.2B",
        "Vanilla AR",
        "artifacts/slices/d24_r8_simple_val_loss_eval_slices_step005952.json",
    ),
    (
        "6.2B",
        "MT-LM",
        "artifacts/slices/d24_r8_mt_v4_val_loss_eval_slices_step005952.json",
    ),
    (
        "15.5B",
        "Vanilla AR",
        "artifacts/slices/d24_r20_simple_val_loss_eval_slices_step014800.json",
    ),
    (
        "15.5B",
        "MT-LM",
        "artifacts/slices/d24_r20_mt_v4_val_loss_eval_slices_step014800.json",
    ),
)


def _fmt_bpb(x: float) -> str:
    return f"{x:.3f}"


def _fmt_ppl(ce: float) -> str:
    return f"{math.exp(ce):.2f}"


def _rows_for_run(training_budget: str, model: str, rel_path: str) -> list[list[str]]:
    path = BUNDLE / rel_path
    with path.open() as f:
        report = json.load(f)["report"]

    first_bpb = report["bucket"]["depth_in_line"]["bpb"][0]
    first_ce = report["bucket"]["depth_in_line"]["ce_per_token"][0]
    boundary = report["mask"]["line_start_mask"]["on"]
    rest = report["mask"]["line_start_mask"]["off"]
    aggregate = report["aggregate"]

    values = (
        ("first_line_token", first_bpb, first_ce),
        ("line_boundary_token", boundary["bpb"], boundary["ce_per_token"]),
        ("rest_of_line", rest["bpb"], rest["ce_per_token"]),
        (
            "aggregate_non_special",
            aggregate["bpb"],
            aggregate["ce_per_non_special_token"],
        ),
    )
    return [
        [
            training_budget,
            model,
            slice_name,
            _fmt_bpb(bpb),
            _fmt_ppl(ce),
            rel_path,
            "generated_from_source_json",
        ]
        for slice_name, bpb, ce in values
    ]


def main() -> None:
    out_path = BUNDLE / "tables" / "slice_breakdown.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "training_budget",
                "model",
                "slice",
                "bpb",
                "ppl",
                "source_artifact",
                "status",
            ]
        )
        for run in RUNS:
            writer.writerows(_rows_for_run(*run))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
