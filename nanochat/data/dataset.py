"""
Pretraining dataset registry + parquet utilities.

Two datasets are wired up today:
  - climbmix (default): karpathy/climbmix-400b-shuffle, ~6.5k shards,
    train = all but last shard, val = last shard.
  - simple_story: duoduoyeah/simple-story-shuffle, 10 train + 10 val shards
    (separate `shard_*` and `validation_*` files), used for tiny-model
    debug / scaling experiments.

Select via the `NANOCHAT_DATASET={climbmix,simple_story}` env var.
"""

import os
import argparse
import time
from dataclasses import dataclass
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq

from nanochat.common import get_base_dir


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    base_url: str               # full file url = f"{base_url}/{filename}"
    data_dir_name: str          # local cache dir name under base_dir
    train_prefix: str           # train file basename prefix, e.g. "shard_"
    val_prefix: str             # val file basename prefix
    train_max_shard: int        # last train shard index (inclusive)
    val_max_shard: int          # last val shard index (inclusive)
    val_is_last_train_shard: bool  # climbmix-style: val == last file under train_prefix


DATASETS = {
    "climbmix": DatasetSpec(
        name="climbmix",
        base_url="https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
        data_dir_name="base_data_climbmix",
        train_prefix="shard_",
        val_prefix="shard_",
        train_max_shard=6541,
        val_max_shard=6542,
        val_is_last_train_shard=True,
    ),
    "simple_story": DatasetSpec(
        name="simple_story",
        base_url="https://huggingface.co/datasets/duoduoyeah/simple-story-shuffle/resolve/main/data",
        data_dir_name="simple_story_data",
        train_prefix="shard_",
        val_prefix="validation_",
        train_max_shard=9,
        val_max_shard=9,
        val_is_last_train_shard=False,
    ),
    "climbmix_split": DatasetSpec(
        name="climbmix_split",
        base_url="",  # local-only: produced by scripts/split_long_blocks.py from climbmix shards
        data_dir_name="base_data_climbmix_split",
        train_prefix="shard_",
        val_prefix="shard_",
        train_max_shard=6541,  # mirrors climbmix shard topology (only used by download CLI)
        val_max_shard=6542,
        val_is_last_train_shard=True,
    ),
}


def get_active_spec() -> DatasetSpec:
    name = os.environ.get("NANOCHAT_DATASET", "climbmix")
    if name not in DATASETS:
        raise ValueError(f"Unknown NANOCHAT_DATASET={name!r}. Options: {list(DATASETS)}")
    return DATASETS[name]


def get_data_dir(spec: DatasetSpec | None = None) -> str:
    spec = spec or get_active_spec()
    return os.path.join(get_base_dir(), spec.data_dir_name)


def index_to_filename(index: int, split: str, spec: DatasetSpec | None = None) -> str:
    spec = spec or get_active_spec()
    prefix = spec.train_prefix if split == "train" else spec.val_prefix
    return f"{prefix}{index:05d}.parquet"


def list_parquet_files(split, data_dir=None, warn_on_legacy=False):
    """Return full paths to parquet files for the given split, using the active dataset spec."""
    assert split in ("train", "val"), f"split must be 'train' or 'val', got {split}"
    spec = get_active_spec()
    data_dir = data_dir if data_dir is not None else get_data_dir(spec)

    # Climbmix-only legacy fallback (kept from the FinewebEdu->ClimbMix migration).
    if spec.name == "climbmix" and not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            print()
            print("    python -m nanochat.data.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            print()
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            print("=" * 80)
            print()
        data_dir = os.path.join(get_base_dir(), "base_data")

    if spec.val_is_last_train_shard:
        # train and val live under the same prefix; val is the final file.
        prefix = spec.train_prefix
        files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith(prefix) and f.endswith(".parquet") and not f.endswith(".tmp")
        )
        paths = [os.path.join(data_dir, f) for f in files]
        return paths[:-1] if split == "train" else paths[-1:]

    prefix = spec.train_prefix if split == "train" else spec.val_prefix
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith(prefix) and f.endswith(".parquet") and not f.endswith(".tmp")
    )
    return [os.path.join(data_dir, f) for f in files]


def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val".
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    for filepath in list_parquet_files(split=split):
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            yield rg.column("text").to_pylist()


# -----------------------------------------------------------------------------
# Download CLI

def _download_one(args):
    """Download a single (index, split, spec) tuple with retries + backoff."""
    index, split, spec = args
    data_dir = get_data_dir(spec)
    filename = index_to_filename(index, split, spec)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{spec.base_url}/{filename}"
    print(f"Downloading {filename}...")
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    return False


def _build_download_list(spec, split_choice, num_files):
    """Return list of (index, split, spec) tuples to download for the chosen split(s)."""
    out = []
    if spec.val_is_last_train_shard:
        # climbmix: train indices 0..train_max_shard, val is the single shard at val_max_shard
        if split_choice in ("train", "both"):
            n = spec.train_max_shard + 1 if num_files == -1 else min(num_files, spec.train_max_shard + 1)
            out.extend((i, "train", spec) for i in range(n))
        if split_choice in ("val", "both"):
            out.append((spec.val_max_shard, "train", spec))  # val file uses train prefix
    else:
        if split_choice in ("train", "both"):
            n = spec.train_max_shard + 1 if num_files == -1 else min(num_files, spec.train_max_shard + 1)
            out.extend((i, "train", spec) for i in range(n))
        if split_choice in ("val", "both"):
            n = spec.val_max_shard + 1 if num_files == -1 else min(num_files, spec.val_max_shard + 1)
            out.extend((i, "val", spec) for i in range(n))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1,
                        help="Number of shards per split to download (-1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4,
                        help="Number of parallel download workers (default: 4)")
    parser.add_argument("--split", type=str, default="both", choices=["train", "val", "both"],
                        help="Which split(s) to download (default: both)")
    args = parser.parse_args()

    spec = get_active_spec()
    data_dir = get_data_dir(spec)
    os.makedirs(data_dir, exist_ok=True)

    print(f"Dataset: {spec.name}")
    print(f"Base URL: {spec.base_url}")
    print(f"Target directory: {data_dir}")
    print()

    download_args = _build_download_list(spec, args.split, args.num_files)
    print(f"Downloading {len(download_args)} files using {args.num_workers} workers...")
    print()

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(_download_one, download_args)

    successful = sum(1 for ok in results if ok)
    print(f"Done! Downloaded: {successful}/{len(download_args)} files to {data_dir}")
