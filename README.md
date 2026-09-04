# LCLM (Block-MT)

d24-scale causal LM that reorders next-token prediction across parallel line-threads per training block, evaluated head-to-head against a matched vanilla AR baseline on ClimbMix.

## Models

| model | tier |
| --- | --- |
| [LCLM · Block-MT](https://huggingface.co/duoduoyeah/nanochat-d24-blockmt-v4-r20) | r20 |
| [Vanilla AR](https://huggingface.co/duoduoyeah/nanochat-d24-simple-r20) | r20 |
| [LCLM · Block-MT](https://huggingface.co/duoduoyeah/nanochat-d24-blockmt-v4-r8) | r8 |
| [Vanilla AR](https://huggingface.co/duoduoyeah/nanochat-d24-simple-r8) | r8 |

## Experiments

| exp | cmd |
| --- | --- |
| bpb | `experiments/repro/run_bpb_eval.sh` |
| core | `experiments/repro/run_core_eval.sh` |
