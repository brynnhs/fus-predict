"""
run_task_benchmark.py
----------------------
Train each model on the full baseline recording, evaluate on the matched
task-period recording.

Same model registry and config as run_benchmark.py, but instead of
splitting one session into a train/test portion of the same period, each
model is fit on the entirety of `baseline_only_standardized/` and scored
against `task_only_standardized/`. Context windows for the first few task
frames are built by bridging in the tail of the baseline recording (see
evaluate_predictor_on_task), so every task frame gets a full lag-length
causal context.

Usage
-----
Run the full benchmark with default settings::

    python scripts/run_task_benchmark.py

Run a quick smoke test on a couple of sessions with two models::

    python scripts/run_task_benchmark.py --models zero,rolling_mean --n-sessions 2

Save raw prediction arrays alongside the results table::

    python scripts/run_task_benchmark.py --save-predictions
"""

from __future__ import annotations

import argparse

import pandas as pd

from fuspredict.data.loading import load_sessions
from fuspredict.data.session import Session
from fuspredict.evaluation.benchmark import aggregate_results, print_summary_table, run_task_benchmark
from fuspredict.models.base import Predictor
from fuspredict.models.registry import ALL_MODEL_NAMES, build_predictor_factories
from fuspredict.project import find_repo_root, get_excluded_sessions, load_project_config


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the baseline-train/task-test benchmark.

    Returns
    
    -------
    argparse.Namespace
        Parsed arguments: ``models``, ``n_sessions``, ``save_predictions``,
        ``overwrite``, ``config``, ``kernel_sizes``.
    """
    parser = argparse.ArgumentParser(
        description="Train on all baseline frames, evaluate on task frames.",
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
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Config filename inside config/ (default: config.yml).",
    )
    parser.add_argument(
        "--kernel-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated spatial smoothing kernel sizes to benchmark "
            "(e.g. '3,5,7'). Default: all sizes listed in config plus 'none' "
            "(unsmoothed). Use 'none' alone to run only the unsmoothed baseline."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help=(
            "Suffix appended to the output directory name "
            "('task_benchmark' -> 'task_benchmark_h5'). Use this when running "
            "multiple horizons back to back, since results are otherwise keyed "
            "only by kernel size and would overwrite each other. Default: none."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the baseline-train/task-test benchmark pipeline end-to-end."""
    args = parse_args()
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(requested_models) - set(ALL_MODEL_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown model name(s): {sorted(unknown)}. "
            f"Choices: {ALL_MODEL_NAMES}"
        )

    repo_root = find_repo_root()
    config = load_project_config(repo_root, config_name=args.config)
    modeling_cfg = config["modeling"]
    ar_cfg = config["ar_analysis"]
    std_cfg = config["preprocessing"]["standardization"]

    all_subjects = config["subjects"]["all"]
    if len(all_subjects) != 1:
        raise ValueError(
            f"run_task_benchmark.py supports exactly one subject; config lists {all_subjects}. "
            "Run once per config or extend this script to loop over subjects."
        )
    subject = all_subjects[0]
    preprocessing_root = repo_root / config["paths"]["preprocessing"] / subject
    mask_dir = preprocessing_root / "tissue_masks"
    benchmark_dir = repo_root / config["paths"]["modeling"] / f"task_benchmark{args.output_suffix}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    results_path = benchmark_dir / "per_session_results.csv"
    aggregate_path = benchmark_dir / "aggregate_summary.csv"

    if args.overwrite and results_path.exists():
        print(f"--overwrite passed: removing existing {results_path}")
        results_path.unlink()

    # Determine which kernel sizes to run.
    # "none" means the unsmoothed baseline_only_standardized/ dir.
    config_kernels: list[int] = list(std_cfg.get("smooth_kernel_sizes", []))
    if args.kernel_sizes is None:
        kernel_tokens: list[str] = ["none"] + [str(k) for k in config_kernels]
    else:
        kernel_tokens = [t.strip() for t in args.kernel_sizes.split(",") if t.strip()]

    exclude_ids = get_excluded_sessions(config, subject, ar_cfg.get("within_session_exclude"))

    all_per_session: list[pd.DataFrame] = []

    for token in kernel_tokens:
        if token in ("none", "0"):
            std_dir = preprocessing_root / "baseline_only_standardized"
            task_std_dir = preprocessing_root / "task_only_standardized"
            glob_pattern = None  # default: baseline_*_unfiltered_standardized.nc
            task_glob_pattern = "task_*_unfiltered_standardized_zscore.nc"
            kernel_label = 0
        else:
            ks = int(token)
            std_dir = preprocessing_root / f"baseline_only_standardized_k{ks}"
            task_std_dir = preprocessing_root / f"task_only_standardized_k{ks}"
            glob_pattern = f"baseline_*_unfiltered_standardized_zscore_smooth{ks}x{ks}.nc"
            task_glob_pattern = f"task_*_unfiltered_standardized_zscore_smooth{ks}x{ks}.nc"
            kernel_label = ks

        if not task_std_dir.is_dir():
            print(f"\n=== Kernel size: {token} — no task dir at {task_std_dir}, skipping ===")
            continue

        print(f"\n=== Kernel size: {token} — loading baseline from {std_dir}, task from {task_std_dir} ===")
        sessions: list[Session] = load_sessions(
            standardized_dir=std_dir,
            mask_dir=mask_dir,
            exclude_ids=exclude_ids,
            glob_pattern=glob_pattern,
        )
        if args.n_sessions is not None:
            sessions = sessions[: args.n_sessions]
        print(f"Loaded {len(sessions)} baseline session(s): {[s.id for s in sessions]}")

        if not sessions:
            print(f"  No baseline sessions found for kernel={token}, skipping.")
            continue

        task_sessions_list = load_sessions(
            standardized_dir=task_std_dir,
            mask_dir=mask_dir,
            exclude_ids=exclude_ids,
            glob_pattern=task_glob_pattern,
        )
        task_sessions = {s.id: s for s in task_sessions_list}
        print(f"Loaded {len(task_sessions)} task session(s): {sorted(task_sessions)}")

        factories = build_predictor_factories(modeling_cfg)
        predictors: list[Predictor] = [factories[name]() for name in requested_models]
        print(f"Running models: {[p.name for p in predictors]}")

        predictions_dir = benchmark_dir / f"predictions_k{kernel_label}" if args.save_predictions else None

        kernel_results_path = benchmark_dir / f"per_session_results_k{kernel_label}.csv"
        if args.overwrite and kernel_results_path.exists():
            kernel_results_path.unlink()

        df = run_task_benchmark(
            predictors=predictors,
            sessions=sessions,
            task_sessions=task_sessions,
            lag=modeling_cfg["n_lags"],
            horizons=modeling_cfg["horizons"],
            results_path=kernel_results_path,
            predictions_dir=predictions_dir,
        )
        df["kernel_size"] = kernel_label
        all_per_session.append(df)
        print(f"  Per-session results: {kernel_results_path}")

    if not all_per_session:
        raise RuntimeError("No results produced across any kernel size.")

    per_session_df = pd.concat(all_per_session, ignore_index=True)
    per_session_df.to_csv(results_path, index=False)
    print(f"\nWrote combined per-session results to {results_path}")

    aggregate_df = aggregate_results(per_session_df)
    aggregate_df.to_csv(aggregate_path, index=False)
    print(f"Wrote aggregate summary to {aggregate_path}")

    print_summary_table(aggregate_df)


if __name__ == "__main__":
    main()
