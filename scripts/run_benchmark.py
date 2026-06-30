"""
run_benchmark.py
-----------------
Single entrypoint script for the fUS frame-prediction model comparison.

This is the only place in the pipeline that touches argparse, config
loading, and filesystem paths — every other module (``fuspredict.models``,
``fuspredict.evaluation``, ``fuspredict.data``) is a pure library that knows
nothing about the repo layout or CLI.

Usage
-----
Run the full benchmark with default settings::

    python scripts/run_benchmark.py

Run a quick smoke test on a couple of sessions with two models::

    python scripts/run_benchmark.py --models zero,rolling_mean --n-sessions 2

Save raw prediction arrays alongside the results table::

    python scripts/run_benchmark.py --save-predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from fuspredict.data.loading import load_sessions
from fuspredict.data.session import Session
from fuspredict.evaluation.benchmark import aggregate_results, run_benchmark
from fuspredict.models.base import Predictor
from fuspredict.models.convlstm import ConvLSTMPredictor
from fuspredict.models.pca_ar import FullFramePCAAR, PatchLagPCAAR
from fuspredict.models.pixel_ar import PixelAR
from fuspredict.models.rolling_mean import RollingMeanPredictor
from fuspredict.models.zero import ZeroPredictor
from fuspredict.project import find_repo_root, load_project_config

ALL_MODEL_NAMES = [
    "zero",
    "rolling_mean",
    "pixel_ar",
    "full_frame_pca_ar",
    "patch_lag_pca_ar",
    "convlstm",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the benchmark run.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: ``models``, ``n_sessions``, ``save_predictions``,
        ``overwrite``.
    """
    parser = argparse.ArgumentParser(
        description="Run the fus_predict model comparison benchmark.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(ALL_MODEL_NAMES),
        help=(
            "Comma-separated subset of model names to run. "
            f"Choices: {', '.join(ALL_MODEL_NAMES)}. Default: all."
        ),
    )
    parser.add_argument(
        "--n-sessions",
        type=int,
        default=None,
        help="Optional cap on the number of sessions to load, for quick testing.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save raw ground-truth/prediction arrays to the predictions directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing results CSV and start the benchmark fresh.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Predictor construction
# ---------------------------------------------------------------------------

def build_predictor_factories(modeling_cfg: dict) -> dict[str, Callable[[], Predictor]]:
    """
    Build a name -> zero-arg constructor mapping for all available models.

    Parameters
    ----------
    modeling_cfg : dict
        The ``modeling`` section of the project config, containing
        ``n_lags`` and per-model hyperparameter blocks.

    Returns
    -------
    dict of str to callable
        Maps model name to a zero-argument callable returning a freshly
        constructed :class:`~fuspredict.models.base.Predictor`.
    """
    n_lags = modeling_cfg["n_lags"]
    pixel_ar_cfg = modeling_cfg["pixel_ar"]
    full_frame_cfg = modeling_cfg["full_frame_pca_ar"]
    patch_lag_cfg = modeling_cfg["patch_lag_pca_ar"]
    convlstm_cfg = modeling_cfg["convlstm"]
    pca_basis_cfg = modeling_cfg["pca_basis"]

    return {
        "zero": lambda: ZeroPredictor(),
        "rolling_mean": lambda: RollingMeanPredictor(window=n_lags),
        "pixel_ar": lambda: PixelAR(
            p=n_lags,
            ridge_lambda=pixel_ar_cfg["ridge_lambda"],
        ),
        "full_frame_pca_ar": lambda: FullFramePCAAR(
            n_components=full_frame_cfg["n_components"],
            ar_lag=n_lags,
            ridge_lambda=full_frame_cfg["ridge_lambda"],
            seed=pca_basis_cfg["seed"],
        ),
        "patch_lag_pca_ar": lambda: PatchLagPCAAR(
            patch_size=patch_lag_cfg["patch_size"],
            n_components=patch_lag_cfg["n_components"],
            ar_lag=n_lags,
            ridge_lambda=patch_lag_cfg["ridge_lambda"],
            seed=pca_basis_cfg["seed"],
        ),
        "convlstm": lambda: ConvLSTMPredictor(
            hidden_channels=convlstm_cfg["hidden_channels"],
            kernel_size=convlstm_cfg["kernel_size"],
            lag=n_lags,
            lr=convlstm_cfg["learning_rate"],
            batch_size=convlstm_cfg["batch_size"],
            n_epochs=convlstm_cfg["n_epochs"],
            grad_clip_norm=convlstm_cfg["grad_clip_norm"],
            seed=pca_basis_cfg["seed"],
        ),
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary_table(aggregate_df: pd.DataFrame) -> None:
    """
    Print a clean console summary of aggregated benchmark results.

    Parameters
    ----------
    aggregate_df : pd.DataFrame
        Output of :func:`fuspredict.evaluation.benchmark.aggregate_results`,
        with columns ``model``, ``horizon``, ``rmse_mean``, ``rmse_std``,
        ``n_sessions``.
    """
    if aggregate_df.empty:
        print("No results to summarize.")
        return

    header = f"{'model':<20}{'horizon':>9}{'RMSE':>20}{'n_sessions':>12}"
    print("\n" + header)
    print("-" * len(header))
    for _, row in aggregate_df.sort_values(["model", "horizon"]).iterrows():
        rmse_str = f"{row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f}"
        print(
            f"{row['model']:<20}{int(row['horizon']):>9}{rmse_str:>20}"
            f"{int(row['n_sessions']):>12}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full benchmark pipeline end-to-end."""
    args = parse_args()
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(requested_models) - set(ALL_MODEL_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown model name(s): {sorted(unknown)}. "
            f"Choices: {ALL_MODEL_NAMES}"
        )

    repo_root = find_repo_root()
    config = load_project_config(repo_root)
    modeling_cfg = config["modeling"]
    ar_cfg = config["ar_analysis"]

    standardized_dir = repo_root / config["paths"]["preprocessing"] / "secundo" / "baseline_only_standardized"
    mask_dir = repo_root / config["paths"]["preprocessing"] / "secundo" / "tissue_masks"
    benchmark_dir = repo_root / config["paths"]["modeling"] / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    results_path = benchmark_dir / "per_session_results.csv"
    aggregate_path = benchmark_dir / "aggregate_summary.csv"
    predictions_dir = benchmark_dir / "predictions" if args.save_predictions else None

    if args.overwrite and results_path.exists():
        print(f"--overwrite passed: removing existing {results_path}")
        results_path.unlink()

    print(f"Loading sessions from {standardized_dir} ...")
    exclude_ids = ar_cfg["within_session_exclude"]
    sessions: list[Session] = load_sessions(
        standardized_dir=standardized_dir,
        mask_dir=mask_dir,
        exclude_ids=exclude_ids,
    )
    if args.n_sessions is not None:
        sessions = sessions[: args.n_sessions]
    print(f"Loaded {len(sessions)} session(s): {[s.id for s in sessions]}")

    if not sessions:
        raise RuntimeError("No sessions loaded; nothing to benchmark.")

    factories = build_predictor_factories(modeling_cfg)
    predictors: list[Predictor] = [factories[name]() for name in requested_models]
    print(f"Running models: {[p.name for p in predictors]}")

    per_session_df = run_benchmark(
        predictors=predictors,
        sessions=sessions,
        lag=modeling_cfg["n_lags"],
        horizons=modeling_cfg["horizons"],
        train_frac=modeling_cfg["train_frac"],
        results_path=results_path,
        predictions_dir=predictions_dir,
    )
    print(f"Wrote per-session results to {results_path}")

    aggregate_df = aggregate_results(per_session_df)
    aggregate_df.to_csv(aggregate_path, index=False)
    print(f"Wrote aggregate summary to {aggregate_path}")

    print_summary_table(aggregate_df)


if __name__ == "__main__":
    main()
