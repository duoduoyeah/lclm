"""Plot D24 position-indexed validation loss diagnostics.

Reads the archived position-loss JSON artifacts from the D24 paper bundle and
writes a compact summary table, plot-point table, and the panel-C PNG used in
the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path


CURVES = (
    ("simple_absolute", "Vanilla AR, source order", "#0072B2", "-"),
    ("mt_absolute", "MT-LM, source order", "#E69F00", "--"),
    ("mt_reordered", "MT-LM, row order", "#D55E00", "-."),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_cache_dir() -> Path:
    root = _repo_root()
    env_path = root / ".env"
    cache_dir: Path | None = None
    if env_path.exists():
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key == "NANOCHAT_BASE_DIR":
                    value = value.strip().strip("'\"")
                    cache_dir = Path(os.path.expanduser(os.path.expandvars(value)))
                    break
    if cache_dir is None:
        cache_dir = root / ".cache"
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    return cache_dir.resolve()


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _weighted_window_bpb(curve: dict, pos: int, half_window: int) -> float | None:
    total_nats = 0.0
    total_bytes = 0
    start = max(0, pos - half_window)
    stop = min(len(curve["bpb"]), pos + half_window + 1)
    for i in range(start, stop):
        bpb = curve["bpb"][i]
        bytes_i = curve["total_bytes"][i]
        if bpb is None or bytes_i <= 0:
            continue
        total_nats += bpb * math.log(2.0) * bytes_i
        total_bytes += bytes_i
    if total_bytes == 0:
        return None
    return total_nats / (math.log(2.0) * total_bytes)


def _weighted_window_ce(curve: dict, pos: int, half_window: int) -> float | None:
    total_nats = 0.0
    total_count = 0
    start = max(0, pos - half_window)
    stop = min(len(curve["ce_per_token"]), pos + half_window + 1)
    for i in range(start, stop):
        ce = curve["ce_per_token"][i]
        count = curve["n_tokens"][i]
        if ce is None or count <= 0:
            continue
        total_nats += ce * count
        total_count += count
    if total_count == 0:
        return None
    return total_nats / total_count


def _weighted_window_ppl(curve: dict, pos: int, half_window: int) -> float | None:
    ce = _weighted_window_ce(curve, pos, half_window)
    if ce is None:
        return None
    return math.exp(ce)


def _write_summary(paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "artifact",
                "simple_step",
                "mt_step",
                "curve",
                "ce_per_token",
                "bpb",
                "n_tokens",
                "total_bytes",
            ]
        )
        for path in paths:
            data = _load(path)
            for key, *_rest in CURVES:
                agg = data[key]["aggregate"]
                writer.writerow(
                    [
                        path.name,
                        data["simple_step"],
                        data["mt_step"],
                        key,
                        f"{agg['ce_per_token']:.9f}",
                        f"{agg['bpb']:.9f}",
                        agg["n_tokens"],
                        agg["total_bytes"],
                    ]
                )


def _write_plot_points(
    data: dict,
    out_path: Path,
    min_position: int,
    max_position: int,
    half_window: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "position",
                "curve",
                "raw_ce_per_token",
                "raw_bpb",
                "raw_ppl",
                "smoothed_ce_per_token",
                "smoothed_bpb",
                "smoothed_ppl",
                "n_tokens",
                "total_bytes",
            ]
        )
        for key, *_rest in CURVES:
            curve = data[key]
            n = min(max_position + 1, len(curve["positions"]))
            for pos in range(min_position, n):
                bpb = curve["bpb"][pos]
                ce = curve["ce_per_token"][pos]
                smooth_ce = _weighted_window_ce(curve, pos, half_window)
                smooth_bpb = _weighted_window_bpb(curve, pos, half_window)
                smooth_ppl = _weighted_window_ppl(curve, pos, half_window)
                writer.writerow(
                    [
                        pos,
                        key,
                        "" if ce is None else f"{ce:.9f}",
                        "" if bpb is None else f"{bpb:.9f}",
                        "" if ce is None else f"{math.exp(ce):.9f}",
                        "" if smooth_ce is None else f"{smooth_ce:.9f}",
                        "" if smooth_bpb is None else f"{smooth_bpb:.9f}",
                        "" if smooth_ppl is None else f"{smooth_ppl:.9f}",
                        curve["n_tokens"][pos],
                        curve["total_bytes"][pos],
                    ]
                )


def _plot(
    data: dict,
    out_path: Path,
    min_position: int,
    max_position: int,
    half_window: int,
) -> None:
    base_cache_dir = _repo_cache_dir()
    mpl_cache_dir = base_cache_dir / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    (base_cache_dir / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NANOCHAT_BASE_DIR", str(base_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(base_cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 3.35))
    for key, label, color, linestyle in CURVES:
        curve = data[key]
        n = min(max_position + 1, len(curve["positions"]))
        xs = []
        ys = []
        for pos in range(min_position, n):
            y = _weighted_window_ppl(curve, pos, half_window)
            if y is None:
                continue
            xs.append(pos)
            ys.append(y)
        ax.plot(xs, ys, label=label, color=color, linestyle=linestyle, linewidth=1.9)

    ax.set_xlim(min_position, max_position)
    ax.set_ylim(3.5, 22.0)
    ax.set_yticks([5, 10, 15, 20])
    ax.set_xlabel("Token position")
    ax.set_ylabel("Validation PPL")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        default="paper/eval_bundle_d24",
        help="D24 eval bundle root.",
    )
    parser.add_argument(
        "--plot-artifact",
        default="phase1_bpb_loss/artifacts/position_loss/position_loss_d24_r20_simple_vs_mt_v4_s014800_n20971520.json",
        help="Bundle-relative JSON artifact to plot.",
    )
    parser.add_argument("--min-position", type=int, default=16)
    parser.add_argument("--max-position", type=int, default=800)
    parser.add_argument("--window", type=int, default=9, help="Centered smoothing window in buckets.")
    parser.add_argument(
        "--paper-figure",
        default="paper/figures/d24_r20_position_loss.png",
        help="Path for the paper-facing PNG copy.",
    )
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    artifact_dir = bundle_dir / "phase1_bpb_loss" / "artifacts" / "position_loss"
    artifacts = sorted(artifact_dir.glob("position_loss_d24_r*_simple_vs_mt_v4_*.json"))
    if not artifacts:
        raise SystemExit(f"No position-loss artifacts found in {artifact_dir}")

    summary_path = bundle_dir / "phase1_bpb_loss" / "tables" / "position_loss_summary.csv"
    _write_summary(artifacts, summary_path)

    plot_artifact = bundle_dir / args.plot_artifact
    data = _load(plot_artifact)
    half_window = max(0, args.window // 2)
    points_path = bundle_dir / "phase1_bpb_loss" / "tables" / "position_loss_plot_points.csv"
    _write_plot_points(data, points_path, args.min_position, args.max_position, half_window)

    bundle_png = bundle_dir / "phase1_bpb_loss" / "figures" / "d24_r20_position_loss.png"
    _plot(data, bundle_png, args.min_position, args.max_position, half_window)

    paper_png = Path(args.paper_figure)
    paper_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle_png, paper_png)

    print(f"wrote {summary_path}")
    print(f"wrote {points_path}")
    print(f"wrote {bundle_png}")
    print(f"wrote {paper_png}")


if __name__ == "__main__":
    main()
