"""
Sniff a nanochat-style HF checkpoint without a GPU.

Default mode (no weights download):
  - huggingface_hub.list_repo_files(repo) -> print files.
  - hf_hub_download README.md and the single meta_*.json.
  - Pretty-print model_config and user_config.
  - 3-way diff model_config vs LegacyGPTConfig defaults (Karpathy d34 arch).

--with-weights:
  - Also hf_hub_download model_*.pt and torch.load on CPU (~8.6 GB for d34).
  - Sanity-check total param count against the documented d34 number.
  - Set-diff state_dict.keys() against
      LegacyGPT(LegacyGPTConfig(**model_config)).state_dict().keys()
  - Print an OK/FAIL line that doubles as the Step 6 gate in
    design/paper/open-source-nanochat.md.

Examples:
  python -m scripts.inspect_hf_checkpoint --repo karpathy/nanochat-d34
  python -m scripts.inspect_hf_checkpoint --repo karpathy/nanochat-d34 --with-weights
"""

import argparse
import dataclasses
import fnmatch
import json
import sys
from dataclasses import dataclass


# Expected total param count for karpathy/nanochat-d34 from the design doc:
# wte(65536*2176) + lm_head(65536*2176) + 34*(2176^2 * 4 + 2176*8704 * 2).
_D34_EXPECTED_PARAMS = 2_217_082_880


@dataclass
class _LegacyGPTConfigStub:
    """Step-1 placeholder for LegacyGPTConfig.

    Replace once nanochat/models/gpt_legacy.py lands (Step 2):
        from nanochat.models.gpt_legacy import LegacyGPTConfig
    """
    sequence_len: int = 2048
    vocab_size: int = 65536
    n_layer: int = 34
    n_head: int = 17
    n_kv_head: int = 17
    n_embd: int = 2176


def _glob_one(files, pattern):
    matches = [f for f in files if fnmatch.fnmatch(f, pattern)]
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one file matching {pattern!r}, got {matches!r}"
        )
    return matches[0]


def _diff_config(meta_config, defaults):
    meta_keys = set(meta_config)
    default_keys = set(defaults)
    only_in_meta = sorted(meta_keys - default_keys)
    only_in_defaults = sorted(default_keys - meta_keys)
    mismatches = [
        (k, defaults[k], meta_config[k])
        for k in sorted(meta_keys & default_keys)
        if meta_config[k] != defaults[k]
    ]
    return only_in_meta, only_in_defaults, mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True,
                        help="HF repo id, e.g. karpathy/nanochat-d34")
    parser.add_argument("--revision", default=None,
                        help="HF revision (default: main)")
    parser.add_argument("--with-weights", action="store_true",
                        help="Also download model_*.pt and verify state_dict "
                             "keys against LegacyGPT (CPU, ~8.6 GB for d34).")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise SystemExit("huggingface_hub not installed; `pip install huggingface_hub`")

    rev_str = f"@{args.revision}" if args.revision else ""
    print(f"=== Inspecting {args.repo}{rev_str} ===")

    files = list_repo_files(args.repo, revision=args.revision)
    print("\nRepo files:")
    for f in sorted(files):
        print(f"  {f}")

    meta_name = _glob_one(files, "meta_*.json")
    print(f"\nMeta file: {meta_name}")
    local_meta = hf_hub_download(args.repo, meta_name, revision=args.revision)
    with open(local_meta) as f:
        meta = json.load(f)

    model_config = meta.get("model_config", {})
    user_config = meta.get("user_config", {})
    print("\nmodel_config:")
    print(json.dumps(model_config, indent=2, sort_keys=True))
    print("\nuser_config:")
    print(json.dumps(user_config, indent=2, sort_keys=True))

    if "README.md" in files:
        local_readme = hf_hub_download(args.repo, "README.md", revision=args.revision)
        with open(local_readme) as f:
            readme = f.read()
        print("\nREADME.md:")
        print(readme.rstrip())

    defaults = dataclasses.asdict(_LegacyGPTConfigStub())
    only_meta, only_def, mismatches = _diff_config(model_config, defaults)
    print("\n=== Config diff vs LegacyGPTConfig defaults ===")
    if not (only_meta or only_def or mismatches):
        print("  EXACT MATCH (keys & values agree with LegacyGPTConfig defaults)")
    else:
        if only_meta:
            print(f"  keys only in meta: {only_meta}")
        if only_def:
            print(f"  keys only in LegacyGPTConfig: {only_def}")
        if mismatches:
            print("  value mismatches (key: default -> meta):")
            for k, d, m in mismatches:
                print(f"    {k}: {d!r} -> {m!r}")

    if not args.with_weights:
        return

    # --- weights mode --------------------------------------------------
    try:
        import torch
    except ImportError:
        raise SystemExit("torch not installed; cannot --with-weights")

    # Fail fast before the 8.6 GB download if Step 2's module isn't present.
    try:
        from nanochat.models.gpt_legacy import LegacyGPT, LegacyGPTConfig
    except ImportError as e:
        raise SystemExit(
            f"Cannot import LegacyGPT (Step 2 not complete?): {e}\n"
            f"--with-weights state_dict diff requires nanochat/models/gpt_legacy.py."
        )

    weights_name = _glob_one(files, "model_*.pt")
    print(f"\nDownloading weights: {weights_name}")
    local_weights = hf_hub_download(args.repo, weights_name, revision=args.revision)
    state_dict = torch.load(local_weights, map_location="cpu", weights_only=True)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    total_params = sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))
    print(f"\nTotal parameter count: {total_params:,}")
    print(f"Expected for d34:      {_D34_EXPECTED_PARAMS:,}")
    if total_params != _D34_EXPECTED_PARAMS:
        print(f"  NOTE: param count differs by {total_params - _D34_EXPECTED_PARAMS:+,}")

    cfg = LegacyGPTConfig(**model_config)
    with torch.device("meta"):
        model = LegacyGPT(cfg)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = sorted(model_keys - ckpt_keys)
    extra = sorted(ckpt_keys - model_keys)

    print("\n=== state_dict key diff (LegacyGPT model vs checkpoint) ===")
    print(f"  model keys:      {len(model_keys)}")
    print(f"  checkpoint keys: {len(ckpt_keys)}")
    if not missing and not extra:
        print("  OK: zero missing, zero extra -- Step 6 gate passes.")
    else:
        print("  FAIL: state_dict mismatch -- fix gpt_legacy.py before Step 7.")
        if missing:
            print(f"  missing from checkpoint (n={len(missing)}):")
            for k in missing:
                print(f"    {k}")
        if extra:
            print(f"  extra in checkpoint (n={len(extra)}):")
            for k in extra:
                print(f"    {k}")
        sys.exit(1)


if __name__ == "__main__":
    main()
