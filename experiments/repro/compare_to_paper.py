#!/usr/bin/env python3
"""Diff a fresh eval rerun against the numbers reported in the paper.

Compares against hardcoded reference values (copied from
paper/eval_bundle_d24/paper_tables/main_results.csv at the private research
repo's revision current as of 2026-09-02) to whatever run_bpb_eval.sh /
run_core_eval.sh just produced. Hardcoded, not read from paper/, because
paper/ is a private submodule of the research repo -- it isn't part of this
public release and anyone outside the lab running this script won't have it
(this script tried reading it once during our own internal testing, right
after paper/ was removed from this branch, and hit exactly that).

Usage:
    python experiments/repro/compare_to_paper.py --bpb-raw-dir $NANOCHAT_BASE_DIR/eval_runs/d24_release_repro_bpb/raw
    python experiments/repro/compare_to_paper.py --core-dir $NANOCHAT_BASE_DIR/eval_runs/d24_release_repro_core/core
    python experiments/repro/compare_to_paper.py --bpb-raw-dir ... --core-dir ...

Tolerance: defaults to 0.007 bpb / 0.01 CORE. The 0.007 val/bpb figure carries
over from a *different* comparison the user previously called acceptable
(d8 multithread causal variants, not this d24 LCLM-vs-AR release) -- treat it
as a starting point, not a pre-approved threshold for this release, and
confirm with the user rather than silently rubber-stamping a gap against it.
"""
import argparse
import csv
import glob
import json
import os
import sys

# model tag -> (tier, model, val_bpb, core_two_line_10shot)
# Source: paper/eval_bundle_d24/paper_tables/main_results.csv, "fresh_bpb_sourced"
# rows (bpb from the fresh 20M-token d24_bpb_v4_20m_20260604 rerun; CORE from
# phase2_core/tables/core_per_task.csv, shots=10, layout=two_line, task=CORE).
PAPER_ROWS = {
    "d24_r20_climbmix_blockmt_v4": ("r20", "mt_v4", 0.7282, 0.2217),
    "d24_r20_climbmix_simple": ("r20", "simple", 0.7131, 0.2579),
    "d24_r8_climbmix_blockmt_v4": ("r8", "mt_v4", 0.7525, 0.2040),
    "d24_r8_climbmix_simple": ("r8", "simple", 0.7370, 0.2323),
}

# model tag -> (tier, model) as used above
TAG_TO_ROW = {tag: (tier, model) for tag, (tier, model, _, _) in PAPER_ROWS.items()}


def load_paper_rows():
    rows = {}
    for tag, (tier, model, bpb, core) in PAPER_ROWS.items():
        rows[(tier, model)] = {"val_bpb": str(bpb), "core_two_line_10shot": str(core)}
    return rows


def find_fresh_bpb(bpb_raw_dir):
    """tag -> bpb, from raw/<tag>/step*/val_loss_eval.json"""
    out = {}
    for tag in TAG_TO_ROW:
        matches = glob.glob(os.path.join(bpb_raw_dir, tag, "step*", "val_loss_eval.json"))
        if not matches:
            continue
        with open(sorted(matches)[-1]) as f:
            d = json.load(f)
        out[tag] = d.get("bpb", d.get("metrics", {}).get("bpb"))
    return out


def find_fresh_core(core_dir, shots="default", layout="two_line"):
    """tag -> CORE centered score, from <tag>_<step>_<shots>shot_<layout>.csv"""
    out = {}
    for tag in TAG_TO_ROW:
        matches = glob.glob(os.path.join(core_dir, f"{tag}_*_{shots}shot_{layout}.csv"))
        if not matches:
            continue
        with open(sorted(matches)[-1]) as f:
            for parts in csv.reader(f):
                if not parts:
                    continue
                label = parts[0].strip()
                if label in ("CORE", "CORE_PARTIAL"):
                    out[tag] = float(parts[2].strip())
    return out


def verdict(gap, tol):
    if gap is None:
        return "?"
    return "OK" if abs(gap) <= tol else "GAP"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bpb-raw-dir", default=None, help="raw/ dir from run_bpb_eval.sh")
    ap.add_argument("--core-dir", default=None, help="core/ dir from run_core_eval.sh")
    ap.add_argument("--bpb-tol", type=float, default=0.007, help="acceptable |val_bpb gap| (default 0.007)")
    ap.add_argument("--core-tol", type=float, default=0.01, help="acceptable |CORE gap| (default 0.01)")
    args = ap.parse_args()

    if not args.bpb_raw_dir and not args.core_dir:
        ap.error("pass at least one of --bpb-raw-dir / --core-dir")

    paper = load_paper_rows()
    fresh_bpb = find_fresh_bpb(args.bpb_raw_dir) if args.bpb_raw_dir else {}
    fresh_core = find_fresh_core(args.core_dir) if args.core_dir else {}

    print(f"{'model':32s} {'paper bpb':>10s} {'fresh bpb':>10s} {'gap':>8s} {'ok?':>5s}   "
          f"{'paper CORE':>10s} {'fresh CORE':>10s} {'gap':>8s} {'ok?':>5s}")
    any_gap = False
    for tag, (tier, model) in TAG_TO_ROW.items():
        row = paper.get((tier, model))
        if row is None:
            print(f"{tag:32s}  (no paper row for tier={tier} model={model})")
            continue
        p_bpb = float(row["val_bpb"])
        p_core = float(row["core_two_line_10shot"])
        f_bpb = fresh_bpb.get(tag)
        f_core = fresh_core.get(tag)
        bpb_gap = (f_bpb - p_bpb) if f_bpb is not None else None
        core_gap = (f_core - p_core) if f_core is not None else None
        bpb_ok = verdict(bpb_gap, args.bpb_tol)
        core_ok = verdict(core_gap, args.core_tol)
        any_gap = any_gap or bpb_ok == "GAP" or core_ok == "GAP"
        print(
            f"{tag:32s} {p_bpb:10.4f} "
            f"{(f_bpb if f_bpb is not None else float('nan')):10.4f} "
            f"{(bpb_gap if bpb_gap is not None else float('nan')):8.4f} {bpb_ok:>5s}   "
            f"{p_core:10.4f} "
            f"{(f_core if f_core is not None else float('nan')):10.4f} "
            f"{(core_gap if core_gap is not None else float('nan')):8.4f} {core_ok:>5s}"
        )

    print()
    print(f"tolerances: bpb <= {args.bpb_tol}, CORE <= {args.core_tol} "
          f"(override with --bpb-tol/--core-tol; confirm these with the user before trusting them)")
    if any_gap:
        print("=> at least one GAP beyond tolerance -- surface to the user, don't silently pass.")
    sys.exit(1 if any_gap else 0)


if __name__ == "__main__":
    main()
