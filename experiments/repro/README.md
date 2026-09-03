# d24 release reproduction — portable scripts

Reruns the eval behind the paper's d24 LCLM (Block-MT) vs. Vanilla AR
head-to-head, for the four checkpoints shipped on Hugging Face:

| tag | arch | tier | tokens | HF repo |
| --- | --- | --- | ---: | --- |
| `d24_r20_climbmix_blockmt_v4` | LCLM (multithread_lm) | r20 | 15.5B | `duoduoyeah/nanochat-d24-blockmt-v4-r20` |
| `d24_r20_climbmix_simple` | Vanilla AR (simple_gpt) | r20 | 15.5B | `duoduoyeah/nanochat-d24-simple-r20` |
| `d24_r8_climbmix_blockmt_v4` | LCLM (multithread_lm) | r8 | 6.2B | `duoduoyeah/nanochat-d24-blockmt-v4-r8` |
| `d24_r8_climbmix_simple` | Vanilla AR (simple_gpt) | r8 | 6.2B | `duoduoyeah/nanochat-d24-simple-r8` |

No SBATCH/cluster assumptions in this folder. These scripts just need the
repo's Python env, a CUDA GPU, and `NANOCHAT_BASE_DIR` pointing at a
`base_checkpoints/` holding the four tags above.

## Getting the data, from nothing (for anyone outside the lab)

On the lab cluster `.env` already points `NANOCHAT_BASE_DIR` at a
pre-populated shared cache — skip straight to **Run**, below. Anyone else
starts with nothing on disk; one command gets everything from the internet:

```bash
bash experiments/repro/bootstrap.sh
```

This downloads the four checkpoints + tokenizers from Hugging Face (needs
`HF_TOKEN` while the repos are private; none needed once they're public) and
derives the `climbmix_split` validation shard used for BPB/CORE eval by
downloading the public `karpathy/climbmix-400b-shuffle` val shard and running
`scripts/split_long_blocks.py` on it locally — `climbmix_split` itself isn't
hosted anywhere, it's a local derivative, so this is a real (if small)
preprocessing step, not a plain download. Everything lands under
`./.cache/nanochat` inside this checkout (mirrors the usual
`~/.cache/nanochat` default, just repo-local so a clone is self-contained).
The CORE benchmark's own eval bundle (task data/prompts) already
self-downloads on first use from a public URL baked into `scripts/base_eval.py`
— nothing to do for that.

See `bootstrap.sh`, `download_checkpoints.py`, and `download_dataset.sh` for
the pieces individually (e.g. to re-run just one).

## TLDR: what gets rerun

1. **val BPB** (`run_bpb_eval.sh`) — held-out bpb on a fixed 20M-token budget,
   per model. This is the number in `main_results.csv`'s `val_bpb` column.
2. **CORE** (`run_core_eval.sh`) — the 22-task CORE downstream benchmark
   (arc_easy/arc_challenge, hellaswag, jeopardy, bigbench_qa_wikidata, …),
   10-shot two-line by default (`main_results.csv`'s `core_two_line_10shot`
   column); set `SHOTS=0` for the 0-shot appendix row, or
   `CORE_SHOT_LAYOUT=one_line` for the same-line ablation row.

Not included (optional/secondary, not needed to check "does this match the
paper"): the r8 MT parallel-cap sweep (cap1/2/4/8), per-K/wave-width and
position-loss diagnostics, token-yield measurements, CORE stability re-checks,
and the r4-simple-only row. Ask if you want those added too.

## Run

`NANOCHAT_BASE_DIR` must be set (lab: from `.env`; elsewhere: exported by
`bootstrap.sh`, or set it yourself to wherever you pointed the downloads).

```bash
NPROC=1 bash experiments/repro/run_bpb_eval.sh
NPROC=1 bash experiments/repro/run_core_eval.sh
SHOTS=0 NPROC=1 bash experiments/repro/run_core_eval.sh   # 0-shot appendix
python experiments/repro/compare_to_paper.py \
    --bpb-raw-dir "$NANOCHAT_BASE_DIR/eval_runs/d24_release_repro_bpb/raw" \
    --core-dir "$NANOCHAT_BASE_DIR/eval_runs/d24_release_repro_core/core"
```

`NPROC=1` will work but is slow for CORE; bump it if you have more than one
GPU free.

## Known gotcha (verified against `scripts/base_eval.py`)

The CORE CSV writer names its output `base_model_<step>.csv` — it does not
include the model tag. `d24_r20_climbmix_blockmt_v4` and
`d24_r20_climbmix_simple` share step 14800, so running them back-to-back
without renaming the CSV in between silently overwrites one with the other.
The original paper bundle hit this exact bug (see the "stale/wrong source"
note in `paper/eval_bundle_d24/phase2_core/provenance/source_paths.csv`).
`run_core_eval.sh` renames each output immediately after the run completes,
so run the four models sequentially (the script already does), not in
parallel against the same `NANOCHAT_BASE_DIR`.

## Reading the comparison

`compare_to_paper.py` diffs fresh numbers against reference values hardcoded
in the script itself (copied from `paper/eval_bundle_d24/paper_tables/main_results.csv`
in the private research repo -- that submodule isn't part of this public
release, so the numbers can't be read live) and flags anything outside
`--bpb-tol` (default 0.007) / `--core-tol` (default 0.01). The 0.007 default is carried over from a
*different* comparison the user previously accepted (d8 multithread-causal
variants) — treat it as a starting point for this d24 LCLM-vs-AR release,
not a pre-cleared threshold; confirm before treating a borderline gap as
"fine."
