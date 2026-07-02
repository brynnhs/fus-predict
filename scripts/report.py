"""
report.py
---------
Generate comparison figures and statistics from a completed benchmark run.

This is a pure orchestration script — no model fitting, no session loading,
no preprocessing. It reads the CSV outputs of ``run_benchmark.py`` (and,
for the spatial figures, the optional saved prediction ``.npz`` arrays from
``run_benchmark.py --save-predictions``) and calls into
``fuspredict.evaluation.stats`` and ``fuspredict.evaluation.visualization``
to produce figures and a Wilcoxon stats table.

Usage
-----
Run the full report after a benchmark run::

    python scripts/run_benchmark.py --save-predictions
    python scripts/report.py

Skip the prediction-dependent spatial figures (no --save-predictions needed)::

    python scripts/report.py --skip-spatial

Point at a non-default results directory or output location::

    python scripts/report.py --results-dir derivatives/modeling/benchmark --out-dir derivatives/modeling/report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fuspredict.evaluation.stats import compute_wilcoxon
from fuspredict.evaluation.visualization import (
    load_predictions,
    plot_paired_diff,
    plot_rmse_strip,
    plot_rmse_vs_horizon,
    plot_rmse_vs_time,
    plot_skill_vs_horizon,
    plot_spatial_comparison,
    plot_spatial_rmse_diff,
    plot_spatial_strip,
    plot_wilcoxon_table,
    make_triplet_video,
)
from fuspredict.project import find_repo_root, load_project_config

PRIMARY_HORIZON = 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(default_session_id: str) -> argparse.Namespace:
    """
    Parse command-line arguments for the report run.

    Parameters
    ----------
    default_session_id : str
        Fallback ``--session-id`` default, read from
        ``config["ar_analysis"]["primary_session_id"]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: ``results_dir``, ``out_dir``, ``session_id``,
        ``skip_spatial``, ``config``.
    """
    parser = argparse.ArgumentParser(
        description="Generate comparison figures and statistics from a benchmark run.",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Config filename inside config/ (default: config.yml).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing per_session_results.csv and aggregate_summary.csv "
        "(default: derivatives/modeling/benchmark).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write figures and stats (default: derivatives/modeling/report).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=default_session_id,
        help="Session to use for the spatial figures (default: ar_analysis.primary_session_id).",
    )
    parser.add_argument(
        "--skip-spatial",
        action="store_true",
        help="Skip the prediction-dependent spatial figures "
        "(requires --save-predictions to have been used during run_benchmark.py).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scalar comparison figures
# ---------------------------------------------------------------------------

def generate_scalar_figures(
    per_session_df: pd.DataFrame,
    models: list[str],
    horizons: list[int],
    out_dir: Path,
    written: list[Path],
) -> None:
    """
    Generate the scalar (non-spatial) comparison figures.

    Parameters
    ----------
    per_session_df : pd.DataFrame
        Long-form per-session results.
    models : list of str
        All model names present in the data, in display order.
    horizons : list of int
        All horizons present in the data, in order.
    out_dir : Path
        Output directory for figures.
    written : list of Path
        Mutated in place to record every file written.
    """
    non_zero_models = [m for m in models if m != "zero"]

    print("Generating scalar comparison figures...")

    path = out_dir / "fig1_rmse_strip"
    plot_rmse_strip(per_session_df, PRIMARY_HORIZON, models, path)
    written.append(path.with_suffix(".pdf"))

    path = out_dir / "fig2_paired_diff"
    plot_paired_diff(per_session_df, PRIMARY_HORIZON, non_zero_models, path)
    written.append(path.with_suffix(".pdf"))

    path = out_dir / "fig3_rmse_vs_horizon"
    plot_rmse_vs_horizon(per_session_df, models, horizons, path)
    written.append(path.with_suffix(".pdf"))

    path = out_dir / "fig4_skill_vs_horizon"
    plot_skill_vs_horizon(per_session_df, non_zero_models, horizons, path)
    written.append(path.with_suffix(".pdf"))

    path = out_dir / "fig5_spatial_strip"
    plot_spatial_strip(per_session_df, PRIMARY_HORIZON, models, path)
    written.append(path.with_suffix(".pdf"))



# ---------------------------------------------------------------------------
# Wilcoxon statistics
# ---------------------------------------------------------------------------

def generate_wilcoxon(
    per_session_df: pd.DataFrame,
    models: list[str],
    horizons: list[int],
    out_dir: Path,
    written: list[Path],
) -> None:
    """
    Compute Wilcoxon statistics and render one table figure per horizon.

    Parameters
    ----------
    per_session_df : pd.DataFrame
        Long-form per-session results.
    models : list of str
        All model names present in the data.
    horizons : list of int
        All horizons present in the data.
    out_dir : Path
        Output directory for the stats CSV and table figures.
    written : list of Path
        Mutated in place to record every file written.
    """
    print("Computing Wilcoxon statistics...")
    stats_df = compute_wilcoxon(per_session_df, horizons, models)

    stats_path = out_dir / "wilcoxon_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    written.append(stats_path)

    if stats_df.empty:
        print("  No Wilcoxon results to render (empty stats table).")
        return

    for h in horizons:
        path = out_dir / f"fig6_wilcoxon_table_h{h}"
        plot_wilcoxon_table(stats_df, h, path)
        written.append(path.with_suffix(".pdf"))


# ---------------------------------------------------------------------------
# Spatial figures (single session, requires saved predictions)
# ---------------------------------------------------------------------------

def generate_spatial_figures(
    predictions_dir: Path,
    session_id: str,
    models: list[str],
    out_dir: Path,
    written: list[Path],
) -> None:
    """
    Generate the prediction-dependent spatial figures for one session.

    Loads saved ``(gt, pred)`` arrays at ``horizon=1`` for every model and,
    if any are found, renders the spatial comparison grid, the spatial
    RMSE-diff grid, and the rolling RMSE-vs-time plot. Missing or unreadable
    prediction files are skipped with a warning rather than raising.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing ``{model}_{session_id}_h{horizon}.npz`` files,
        as written by ``run_benchmark.py --save-predictions``.
    session_id : str
        Session to load predictions for.
    models : list of str
        All model names present in the benchmark results.
    out_dir : Path
        Output directory for figures.
    written : list of Path
        Mutated in place to record every file written.
    """
    if not predictions_dir.exists():
        print(
            f"  WARNING: predictions directory not found ({predictions_dir}); "
            "skipping spatial figures. Re-run run_benchmark.py with --save-predictions."
        )
        return

    print(f"Loading saved predictions for session {session_id} (h={PRIMARY_HORIZON})...")
    predictions: dict[str, tuple] = {}
    for model in models:
        npz_path = predictions_dir / f"{model}_{session_id}_h{PRIMARY_HORIZON}.npz"
        if not npz_path.exists():
            print(f"  WARNING: missing predictions file, skipping model: {npz_path}")
            continue
        try:
            predictions[model] = load_predictions(
                predictions_dir, model, session_id, PRIMARY_HORIZON
            )
        except Exception as exc:
            print(f"  WARNING: failed to load predictions for {model}: {exc}")

    if not predictions:
        print("  No predictions available for this session; skipping spatial figures.")
        return

    print(f"  Loaded predictions for: {list(predictions.keys())}")

    path = out_dir / "fig7_spatial_comparison"
    plot_spatial_comparison(predictions, session_id, path)
    written.append(path.with_suffix(".pdf"))

    path = out_dir / "fig8_spatial_rmse_diff"
    plot_spatial_rmse_diff(predictions, path)
    written.append(path.with_suffix(".pdf"))

    path = out_dir / "fig9_rmse_vs_time"
    plot_rmse_vs_time(predictions, path)
    written.append(path.with_suffix(".pdf"))
    
    for model, (gt, pred) in predictions.items():
        path = out_dir / f"video_{session_id}_{model}_triplet.mp4"
        make_triplet_video(gt, pred, path, session_id=session_id)
        written.append(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full reporting pipeline end-to-end."""
    repo_root = find_repo_root()
    # Pre-parse just --config so we can load the right config before full parse
    import sys as _sys
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default="config.yml")
    _pre_args, _ = _pre.parse_known_args(_sys.argv[1:])
    config = load_project_config(repo_root, config_name=_pre_args.config)
    ar_cfg = config.get("ar_analysis", {})
    default_session_id = str(ar_cfg.get("primary_session_id", ""))

    args = parse_args(default_session_id)

    results_dir = (
        Path(args.results_dir)
        if args.results_dir is not None
        else repo_root / config["paths"]["modeling"] / "benchmark"
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else repo_root / config["paths"]["modeling"] / "report"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "per_session_results.csv"
    aggregate_path = results_dir / "aggregate_summary.csv"

    for p in (results_path, aggregate_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\nRun scripts/run_benchmark.py first."
            )

    print(f"Loading results from {results_dir} ...")
    per_session_df = pd.read_csv(results_path)

    if per_session_df.empty:
        raise RuntimeError(f"{results_path} is empty; nothing to report.")

    models = sorted(per_session_df["model"].unique().tolist())
    horizons = sorted(int(h) for h in per_session_df["horizon"].unique().tolist())
    print(f"  Models:   {models}")
    print(f"  Horizons: {horizons}")
    print(f"  Sessions: {per_session_df['session_id'].nunique()}")

    written: list[Path] = []

    generate_scalar_figures(per_session_df, models, horizons, out_dir, written)
    generate_wilcoxon(per_session_df, models, horizons, out_dir, written)

    if not args.skip_spatial:
        if not args.session_id:
            print(
                "WARNING: no --session-id given and ar_analysis.primary_session_id "
                "is unset in config; skipping spatial figures."
            )
        else:
            print(f"\nSpatial figures (session: {args.session_id}) ...")
            predictions_dir = results_dir / "predictions"
            generate_spatial_figures(
                predictions_dir, args.session_id, models, out_dir, written
            )

    print(f"\n{'-' * 60}")
    print(f"Done. Wrote {len(written)} file(s) to {out_dir}:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
