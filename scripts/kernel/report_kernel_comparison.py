"""
report_kernel_comparison.py
---------------------------
Compare benchmark results across kernel sizes (k0, k3, k5, k7) at horizon=1.

Loads per_session_results_k{N}.csv files from the benchmark directory and
produces three figures:

  fig1_rmse_strip_by_kernel  — per-session RMSE distribution, x-axis = kernel
                               size, faceted by model and region
  fig2_paired_diff_by_kernel — RMSE(kX) − RMSE(k0) per session, one panel
                               per model, one colour per kernel size
  fig3_median_vs_kernel      — median RMSE vs kernel size, one line per model,
                               two panels for vessel vs non-vessel

Usage
-----
    python scripts/report_kernel_comparison.py
    python scripts/report_kernel_comparison.py --results-dir derivatives/modeling/benchmark
    python scripts/report_kernel_comparison.py --out-dir derivatives/modeling/report_kernel_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fuspredict.evaluation.visualization import (
    plot_median_vs_kernel,
    plot_paired_diff_by_kernel,
    plot_strip_by_kernel,
)
from fuspredict.project import find_repo_root, load_project_config

HORIZON = 1
KERNEL_SIZES = [0, 3, 5, 7]
MODELS_OF_INTEREST = ["zero", "rolling_mean", "patch_lag_pca_ar", "frozen_pca_ar", "frozen_pca_rolling_mean", "convlstm", "convlstm_pca_latent"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_kernels(results_dir: Path) -> pd.DataFrame:
    """Load per_session_results_k{N}.csv for each kernel size and stack them."""
    frames = []
    for k in KERNEL_SIZES:
        path = results_dir / f"per_session_results_k{k}.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping k={k}")
            continue
        df = pd.read_csv(path)
        df["kernel"] = k
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No per_session_results_k*.csv found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare benchmark results across kernel sizes."
    )
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--config", default="config.yml")
    return parser.parse_args()


def main() -> None:
    repo_root = find_repo_root()
    args = parse_args()
    config = load_project_config(repo_root, config_name=args.config)

    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else repo_root / config["paths"]["modeling"] / "benchmark"
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo_root / config["paths"]["modeling"] / "report_kernel_comparison"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading kernel results from {results_dir} ...")
    df = load_all_kernels(results_dir)

    available_kernels = sorted(df["kernel"].unique().tolist())
    all_models = sorted(df["model"].unique().tolist())
    models = [m for m in MODELS_OF_INTEREST if m in all_models]
    print(f"  Kernels:  {available_kernels}")
    print(f"  Models:   {models}")
    print(f"  Sessions: {df['session_id'].nunique()}")

    written: list[Path] = []

    print("Generating fig1: RMSE strip by kernel...")
    path = out_dir / "fig1_rmse_strip_by_kernel"
    plot_strip_by_kernel(df, HORIZON, models, available_kernels, path)
    written.append(path.with_suffix(".png"))

    print("Generating fig2: paired diff vs k0...")
    path = out_dir / "fig2_paired_diff_by_kernel"
    plot_paired_diff_by_kernel(df, HORIZON, models, available_kernels, path)
    written.append(path.with_suffix(".png"))

    print("Generating fig3: median RMSE vs kernel size...")
    path = out_dir / "fig3_median_vs_kernel"
    plot_median_vs_kernel(df, HORIZON, models, available_kernels, path)
    written.append(path.with_suffix(".png"))
    written.append((out_dir / "fig3b_median_vs_kernel_by_region").with_suffix(".png"))

    print(f"\nDone. Wrote {len(written)} file(s) to {out_dir}:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
