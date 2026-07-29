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

    python scripts/benchmarking/run_benchmark.py --save-predictions
    python scripts/report.py

Skip the prediction-dependent spatial figures (no --save-predictions needed)::

    python scripts/report.py --skip-spatial

Point at a non-default results directory or output location::

    python scripts/report.py --results-dir derivatives/modeling/benchmark --out-dir derivatives/modeling/report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fuspredict.evaluation.stats import (
    _aligned,
    bootstrap_median_diff_ci,
    compute_wilcoxon,
    wilcoxon_test,
    rmse as _rmse_fn,
)
from fuspredict.data.loading import load_predictions
from fuspredict.evaluation.visualization import (
    plot_frame_recon_pred_compare,
    plot_paired_diff,
    plot_predictions_vs_time,
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

# Latent-space-reconstruction comparison: PCA/ICA-AR models vs the ConvLSTM
# latent-reconstruction models, scored on rmse_oracle_* (oracle-decoded RMSE).
LATENT_COMPARISON_AR_MODELS = ["frozen_pca_ar", "frozen_ica_ar"]
LATENT_COMPARISON_RECON_MODELS = ["convlstm_pca_latent", "convlstm_ica_latent"]

# Models with no PCA/ICA latent space of their own, scored against the
# PCA/ICA basis reconstruction of the target frames (each frozen AR model's
# oracle_gt) instead of the raw target frames. Synthetic model names added
# to per_session_df by compute_vs_oracle_rmse.
# {reference_model: {oracle_model: synthetic_model_name}}
VS_ORACLE_MODELS = {
    "convlstm": {
        "frozen_pca_ar": "convlstm_vs_pca_oracle",
        "frozen_ica_ar": "convlstm_vs_ica_oracle",
    },
    "rolling_mean": {
        "frozen_pca_ar": "rolling_mean_vs_pca_oracle",
        "frozen_ica_ar": "rolling_mean_vs_ica_oracle",
    },
}


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
        ``skip_spatial``.
    """
    parser = argparse.ArgumentParser(
        description="Generate comparison figures and statistics from a benchmark run.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["temporal_split", "full_baseline"],
        default="temporal_split",
        help="Which benchmark run to report on: 'temporal_split' (default) reads "
        "derivatives/modeling/benchmark/, the output of run_benchmark.py "
        "(train/test split within one session). 'full_baseline' reads "
        "derivatives/modeling/task_benchmark/, the output of run_task_benchmark.py "
        "(fit on the full baseline recording, score on the matched task recording). "
        "Only sets the results-dir/out-dir defaults; ignored if either is passed explicitly.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing per_session_results.csv and aggregate_summary.csv "
        "(default: derivatives/modeling/benchmark, or task_benchmark for "
        "--eval-mode full_baseline).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write figures and stats (default: derivatives/modeling/report, "
        "or task_report for --eval-mode full_baseline).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=default_session_id,
        help="Session to use for the spatial figures (default: ar_analysis.primary_session_id).",
    )
    parser.add_argument(
        "--kernel-size",
        type=str,
        default=None,
        help="Kernel size label to report on (e.g. 3, 5, none). Selects "
        "per_session_results_k{N}.csv and predictions_k{N}/ inside --results-dir, "
        "and writes output to report_k{N}/. If omitted, uses per_session_results.csv "
        "with no suffix (legacy behaviour).",
    )
    parser.add_argument(
        "--skip-spatial",
        action="store_true",
        help="Skip the prediction-dependent spatial figures "
        "(requires --save-predictions to have been used during run_benchmark.py).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to include (default: all models present "
        "in the results CSV).",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Config filename inside config/ (default: config.yml).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Non-latent models (ConvLSTM, rolling mean, ...) vs. PCA/ICA oracle
# reconstruction
# ---------------------------------------------------------------------------

def compute_vs_oracle_rmse(
    per_session_df: pd.DataFrame,
    predictions_dir: Path,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Score each reference model's predictions against the PCA/ICA basis
    reconstruction of the target frames, instead of the raw target frames.

    Reference models in :data:`VS_ORACLE_MODELS` (e.g. ``convlstm``,
    ``rolling_mean``) never touch a PCA/ICA latent space, so they have no
    ``reconstruct_oracle`` of their own. This instead reuses each frozen AR
    model's ``oracle_gt`` (the true target frame projected onto that
    model's frozen basis and inverted) as the ground truth, and computes
    RMSE between it and the reference model's own ``pred``. This isolates
    how much of the reference model's error is "real" prediction error vs.
    error that any model sharing that basis would incur just from the
    basis's reconstruction limits — without ever passing the reference
    model's predictions through a basis.

    Requires ``run_benchmark.py --save-predictions`` to have been run for
    every reference model and oracle model named in
    :data:`VS_ORACLE_MODELS`.

    Parameters
    ----------
    per_session_df : pd.DataFrame
        Long-form per-session results; used only to enumerate session IDs.
    predictions_dir : Path
        Directory containing ``{model}_{session_id}_h{horizon}.npz`` files.
    horizons : list of int
        Horizons to compute this comparison for.

    Returns
    -------
    pd.DataFrame
        New rows with columns ``session_id``, ``model``, ``horizon``,
        ``rmse_oracle_full`` (the same column name used by the frozen
        AR/ConvLSTM-latent models' oracle-decoded RMSE, so they can all be
        plotted together), in the same shape as ``per_session_df`` (other
        columns are absent/NaN). Rows are only produced where both the
        reference and oracle predictions files exist.
    """
    if not predictions_dir.exists():
        return pd.DataFrame()

    session_ids = sorted(per_session_df["session_id"].unique().tolist())
    rows: list[dict] = []

    for reference_model, oracle_map in VS_ORACLE_MODELS.items():
        for oracle_model, synth_name in oracle_map.items():
            for horizon in horizons:
                for session_id in session_ids:
                    ref_path = predictions_dir / f"{reference_model}_{session_id}_h{horizon}.npz"
                    oracle_path = predictions_dir / f"{oracle_model}_{session_id}_h{horizon}.npz"
                    if not ref_path.exists() or not oracle_path.exists():
                        continue
                    try:
                        _, ref_pred = load_predictions(
                            predictions_dir, reference_model, session_id, horizon
                        )
                        oracle_gt, _ = load_predictions(
                            predictions_dir, oracle_model, session_id, horizon,
                            gt_key="oracle_gt",
                        )
                    except Exception as exc:
                        print(f"  WARNING: failed to load predictions for {synth_name} "
                              f"({session_id}, h={horizon}): {exc}")
                        continue

                    n = min(ref_pred.shape[0], oracle_gt.shape[0])
                    if n == 0:
                        continue
                    rmse_oracle_full = float(_rmse_fn(ref_pred[:n], oracle_gt[:n]))
                    rows.append({
                        "session_id": session_id,
                        "model": synth_name,
                        "horizon": horizon,
                        "rmse_oracle_full": rmse_oracle_full,
                    })

    return pd.DataFrame(rows)


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
    vs_oracle_names = {
        synth_name
        for oracle_map in VS_ORACLE_MODELS.values()
        for synth_name in oracle_map.values()
    }
    scalar_models = [m for m in models if m not in vs_oracle_names]
    non_zero_models = [m for m in scalar_models if m != "zero"]

    print("Generating scalar comparison figures...")

    path = out_dir / "fig1_rmse_strip"
    plot_rmse_strip(per_session_df, PRIMARY_HORIZON, scalar_models, path)
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

    latent_models = [
        m
        for m in LATENT_COMPARISON_AR_MODELS + LATENT_COMPARISON_RECON_MODELS
        if m in models
    ]
    if any(m in models for m in LATENT_COMPARISON_AR_MODELS) and any(
        m in models for m in LATENT_COMPARISON_RECON_MODELS
    ):
        print("Generating latent-reconstruction comparison figures...")

        path = out_dir / "fig5b_spatial_strip_latent_comparison"
        plot_spatial_strip(
            per_session_df, PRIMARY_HORIZON, latent_models, path,
            rmse_cols=("rmse_oracle_vessel", "rmse_oracle_nonvessel"),
        )
        written.append(path.with_suffix(".pdf"))

    # Per-basis RMSE-vs-oracle comparison: the frozen AR model, the frozen
    # in-basis rolling mean, the ConvLSTM trained on that basis's latents,
    # and every other reference model (ConvLSTM, rolling mean, ...) — all
    # scored against that basis's oracle reconstruction and sharing
    # rmse_oracle_full — combined into a single strip per basis.
    basis_figures = [
        (
            "pca", "frozen_pca_ar", "frozen_pca_rolling_mean",
            "convlstm_pca_latent", "fig1b_rmse_strip_pca_comparison",
        ),
        (
            "ica", "frozen_ica_ar", "frozen_ica_rolling_mean",
            "convlstm_ica_latent", "fig1c_rmse_strip_ica_comparison",
        ),
    ]
    for basis, ar_model, basis_rolling_mean_model, recon_model, fig_name in basis_figures:
        vs_oracle_names = [
            oracle_map[ar_model]
            for oracle_map in VS_ORACLE_MODELS.values()
            if ar_model in oracle_map
        ]
        basis_models = [
            m for m in [ar_model, basis_rolling_mean_model, recon_model] + vs_oracle_names
            if m in models
        ]
        if len(basis_models) < 2:
            continue
        print(f"Generating {basis.upper()} oracle comparison figure...")

        path = out_dir / fig_name
        plot_rmse_strip(
            per_session_df, PRIMARY_HORIZON, basis_models, path,
            rmse_col="rmse_oracle_full",
        )
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


def generate_vs_oracle_wilcoxon(
    per_session_df: pd.DataFrame,
    horizon: int,
    out_dir: Path,
    written: list[Path],
) -> None:
    """
    Paired Wilcoxon test, per basis, between every pair of reference models
    in :data:`VS_ORACLE_MODELS` scored on ``rmse_oracle_full``.

    Sessions are paired by ``session_id`` (each reference model's oracle
    RMSE for that session), consistent with the rest of the pipeline's
    within-session paired-test convention (see
    :func:`fuspredict.evaluation.stats.compute_wilcoxon`). With two
    reference models (``convlstm``, ``rolling_mean``) this produces one
    comparison per basis; with more than two, every pair is tested.

    Parameters
    ----------
    per_session_df : pd.DataFrame
        Long-form per-session results, including the synthetic
        ``*_vs_pca_oracle`` / ``*_vs_ica_oracle`` rows added by
        :func:`compute_vs_oracle_rmse`.
    horizon : int
        Horizon to test (the synthetic rows are currently only computed at
        ``PRIMARY_HORIZON``).
    out_dir : Path
        Output directory for the stats CSV.
    written : list of Path
        Mutated in place to record every file written.
    """
    import itertools

    reference_models = list(VS_ORACLE_MODELS.keys())
    rows: list[dict] = []

    for oracle_model in {"frozen_pca_ar": "pca", "frozen_ica_ar": "ica"}:
        basis = {"frozen_pca_ar": "pca", "frozen_ica_ar": "ica"}[oracle_model]
        synth_names = {
            ref: VS_ORACLE_MODELS[ref].get(oracle_model)
            for ref in reference_models
        }
        synth_names = {k: v for k, v in synth_names.items() if v is not None}
        if len(synth_names) < 2:
            continue

        wide = _aligned(
            per_session_df, list(synth_names.values()), horizon,
            rmse_col="rmse_oracle_full",
        )
        if wide.empty:
            continue

        for ref_a, ref_b in itertools.combinations(synth_names, 2):
            name_a, name_b = synth_names[ref_a], synth_names[ref_b]
            if name_a not in wide.columns or name_b not in wide.columns:
                continue
            a = wide[name_a].values.astype(float)
            b = wide[name_b].values.astype(float)
            W, p = wilcoxon_test(a, b)
            med, ci_lo, ci_hi = bootstrap_median_diff_ci(a, b)
            rows.append({
                "basis": basis,
                "horizon": horizon,
                "model_A": name_a,
                "model_B": name_b,
                "n_sessions": len(a),
                "W": W,
                "p_value": p,
                "median_diff": med,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
            })

    if not rows:
        print("  No reference-vs-oracle Wilcoxon results (need >=2 reference models "
              "per basis with matching sessions).")
        return

    stats_df = pd.DataFrame(rows)
    stats_path = out_dir / "wilcoxon_stats_vs_oracle.csv"
    stats_df.to_csv(stats_path, index=False)
    written.append(stats_path)
    print(f"  Wrote {stats_path}")
    for _, row in stats_df.iterrows():
        print(
            f"    [{row['basis']}] {row['model_A']} vs {row['model_B']}: "
            f"n={row['n_sessions']}, W={row['W']:.1f}, p={row['p_value']:.3g}, "
            f"median_diff={row['median_diff']:.4f}"
        )


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

    sig_vals = []
    for gt, pred in predictions.values():
        n = min(gt.shape[0], pred.shape[0])
        sig_vals += [
            np.percentile(np.abs(gt[:n].mean(axis=0)), 98),
            np.percentile(np.abs(pred[:n].mean(axis=0)), 98),
        ]
    sig_lim = float(np.percentile(sig_vals, 98))

    path = out_dir / "fig7_spatial_comparison"
    plot_spatial_comparison(predictions, session_id, path, sig_lim=sig_lim)
    written.append(path.with_suffix(".pdf"))

    path = out_dir / "fig8_spatial_rmse_diff"
    plot_spatial_rmse_diff(predictions, path)
    written.append(path.with_suffix(".pdf"))

    path = out_dir / "fig9_rmse_vs_time"
    plot_rmse_vs_time(predictions, path)
    written.append(path.with_suffix(".pdf"))

    path = out_dir / "fig9b_predictions_vs_time"
    plot_predictions_vs_time(predictions, path)
    written.append(path.with_suffix(".pdf"))

    for model, (gt, pred) in predictions.items():
        path = out_dir / f"video_{session_id}_{model}_triplet.mp4"
        make_triplet_video(gt, pred, path, session_id=session_id)
        written.append(path)

    generate_latent_spatial_figures(predictions_dir, session_id, models, out_dir, written)


def generate_latent_spatial_figures(
    predictions_dir: Path,
    session_id: str,
    models: list[str],
    out_dir: Path,
    written: list[Path],
) -> None:
    """
    Generate the spatial RMSE-diff figure restricted to PCA/ICA-AR models vs
    the ConvLSTM latent-reconstruction models, using each model's basis
    reconstruction of the target frames (``oracle_gt``) as ground truth
    instead of the raw target frames, so all models are compared in
    reconstructed (not raw pixel) space.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing ``{model}_{session_id}_h{horizon}.npz`` files.
    session_id : str
        Session to load predictions for.
    models : list of str
        All model names present in the benchmark results.
    out_dir : Path
        Output directory for figures.
    written : list of Path
        Mutated in place to record every file written.
    """
    latent_models = [
        m
        for m in LATENT_COMPARISON_AR_MODELS + LATENT_COMPARISON_RECON_MODELS
        if m in models
    ]
    if not (
        any(m in models for m in LATENT_COMPARISON_AR_MODELS)
        and any(m in models for m in LATENT_COMPARISON_RECON_MODELS)
    ):
        return

    predictions: dict[str, tuple] = {}
    for model in latent_models:
        npz_path = predictions_dir / f"{model}_{session_id}_h{PRIMARY_HORIZON}.npz"
        if not npz_path.exists():
            print(f"  WARNING: missing predictions file, skipping model: {npz_path}")
            continue
        try:
            predictions[model] = load_predictions(
                predictions_dir, model, session_id, PRIMARY_HORIZON, gt_key="oracle_gt"
            )
        except Exception as exc:
            print(f"  WARNING: failed to load oracle predictions for {model}: {exc}")

    if not predictions:
        print("  No oracle predictions available; skipping latent comparison spatial figures.")
        return

    print(f"  Loaded oracle-space predictions for: {list(predictions.keys())}")

    path = out_dir / "fig8b_spatial_rmse_diff_latent"
    plot_spatial_rmse_diff(predictions, path)
    written.append(path.with_suffix(".pdf"))

    for model in LATENT_COMPARISON_RECON_MODELS:
        if model not in predictions:
            continue
        recon, pred = predictions[model]
        try:
            raw_gt, _ = load_predictions(predictions_dir, model, session_id, PRIMARY_HORIZON)
        except Exception as exc:
            print(f"  WARNING: failed to load raw GT for {model}: {exc}")
            continue
        n = min(raw_gt.shape[0], recon.shape[0], pred.shape[0])
        frame_i = n // 2
        path = out_dir / f"fig7b_frame_compare_{model}"
        plot_frame_recon_pred_compare(raw_gt[:n], recon[:n], pred[:n], frame_i, model, path)
        written.append(path.with_suffix(".pdf"))


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

    kernel_label = args.kernel_size  # e.g. "3", "5", "none", or None
    kernel_suffix = f"_k{kernel_label}" if kernel_label is not None else ""

    default_results_dirname = (
        "task_benchmark" if args.eval_mode == "full_baseline" else "benchmark"
    )
    default_report_dirname = (
        "task_report" if args.eval_mode == "full_baseline" else "report"
    )

    results_dir = (
        Path(args.results_dir)
        if args.results_dir is not None
        else repo_root / config["paths"]["modeling"] / default_results_dirname
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else repo_root / config["paths"]["modeling"] / f"{default_report_dirname}{kernel_suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"per_session_results{kernel_suffix}.csv"
    aggregate_path = results_dir / "aggregate_summary.csv"

    for p in (results_path, aggregate_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\nRun scripts/benchmarking/run_benchmark.py first."
            )

    print(f"Loading results from {results_dir} ...")
    per_session_df = pd.read_csv(results_path)

    if per_session_df.empty:
        raise RuntimeError(f"{results_path} is empty; nothing to report.")

    if args.models is not None:
        requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
        present = set(per_session_df["model"].unique().tolist())
        unknown = set(requested_models) - present
        if unknown:
            raise ValueError(
                f"Unknown model name(s): {sorted(unknown)}. "
                f"Present in {results_path.name}: {sorted(present)}"
            )

    horizons = sorted(int(h) for h in per_session_df["horizon"].unique().tolist())

    present_models = set(per_session_df["model"].unique().tolist())
    if any(m in present_models for m in VS_ORACLE_MODELS):
        predictions_dir = results_dir / f"predictions{kernel_suffix}"
        print("Computing reference-vs-oracle RMSE from saved predictions...")
        oracle_rows = compute_vs_oracle_rmse(
            per_session_df, predictions_dir, horizons
        )
        if not oracle_rows.empty:
            per_session_df = pd.concat([per_session_df, oracle_rows], ignore_index=True)
            print(f"  Added models: {sorted(oracle_rows['model'].unique().tolist())}")
        else:
            print("  No reference-vs-oracle rows computed (missing saved predictions?).")

    vs_oracle_names = {
        synth_name
        for oracle_map in VS_ORACLE_MODELS.values()
        for synth_name in oracle_map.values()
    }

    if args.models is not None:
        keep_models = set(requested_models) | vs_oracle_names
        per_session_df = per_session_df[per_session_df["model"].isin(keep_models)]

    models = sorted(per_session_df["model"].unique().tolist())
    print(f"  Models:   {models}")
    print(f"  Horizons: {horizons}")
    print(f"  Sessions: {per_session_df['session_id'].nunique()}")

    written: list[Path] = []

    scalar_models = [m for m in models if m not in vs_oracle_names]

    generate_scalar_figures(per_session_df, models, horizons, out_dir, written)
    generate_wilcoxon(per_session_df, scalar_models, horizons, out_dir, written)
    generate_vs_oracle_wilcoxon(per_session_df, PRIMARY_HORIZON, out_dir, written)

    if not args.skip_spatial:
        if not args.session_id:
            print(
                "WARNING: no --session-id given and ar_analysis.primary_session_id "
                "is unset in config; skipping spatial figures."
            )
        else:
            print(f"\nSpatial figures (session: {args.session_id}) ...")
            predictions_dir = results_dir / f"predictions{kernel_suffix}"
            synthetic_vs_oracle_models = {
                synth_name
                for oracle_map in VS_ORACLE_MODELS.values()
                for synth_name in oracle_map.values()
            }
            spatial_models = [m for m in models if m not in synthetic_vs_oracle_models]
            generate_spatial_figures(
                predictions_dir, args.session_id, spatial_models, out_dir, written
            )

    print(f"\n{'-' * 60}")
    print(f"Done. Wrote {len(written)} file(s) to {out_dir}:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
