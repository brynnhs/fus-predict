"""
rmse_plots.py
-------------
Per-session RMSE strip/paired-diff plots and horizon-sweep curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from fuspredict.evaluation.stats import _aligned
from fuspredict.evaluation.visualization._common import (
    COLORS,
    FIG_W_DOUBLE,
    LABELS,
    _despine,
    _iqr_ylim,
    savefig,
)


# ---------------------------------------------------------------------------
# Per-session strip plots
# ---------------------------------------------------------------------------

def plot_rmse_strip(
    df: pd.DataFrame,
    horizon: int,
    models: list[str],
    out_path: Path,
    rmse_col: str = "rmse_full",
) -> None:
    """Per-session RMSE jitter/strip plot at a given horizon, all models.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form results with columns ``session_id``, ``model``,
        ``horizon``, and ``rmse_col``.
    horizon : int
        Horizon to plot.
    models : list of str
        Models to include, in display order.
    out_path : Path
        Output PDF path (suffix is normalized).
    rmse_col : str
        Column to plot (default ``"rmse_full"``). Pass ``"rmse_oracle_full"``
        to compare latent-reconstruction models on their oracle-decoded RMSE.
    """
    wide = _aligned(df, models, horizon, rmse_col=rmse_col)
    if wide.empty:
        return

    rng = np.random.default_rng(0)
    all_vals_list = [wide[m].values.astype(float) for m in models if m in wide.columns]
    ylo, yhi = _iqr_ylim(all_vals_list)
    n_clipped = sum(int(np.sum((v < ylo) | (v > yhi))) for v in all_vals_list)

    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.5), constrained_layout=True)

    if "zero" in wide.columns:
        zero_mean = float(np.mean(wide["zero"].values.astype(float)))
        ax.axhline(
            zero_mean, color=COLORS["zero"], lw=1.5, ls="--",
            label=f"Zero mean ({zero_mean:.3f})", zorder=1,
        )

    for xi, m in enumerate(models):
        if m not in wide.columns:
            continue
        vals = wide[m].values.astype(float)
        color = COLORS.get(m, "#333333")
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            xi + jitter, vals, color=color, s=18, alpha=0.75,
            linewidths=0.3, edgecolors="white", zorder=3,
        )
        mn = float(np.mean(vals))
        ax.plot(
            [xi - 0.25, xi + 0.25], [mn, mn],
            color=color, lw=2.0, zorder=4, solid_capstyle="round",
        )

    ax.set_ylim(ylo, yhi)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABELS.get(m, m) for m in models], rotation=20, ha="right")
    ax.set_ylabel("RMSE (z-score)")
    if n_clipped:
        ax.annotate(
            f'{n_clipped} point{"s" if n_clipped > 1 else ""} outside axis',
            xy=(0.99, 0.01), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=7, color="grey",
        )
    ax.legend()
    ax.grid(axis="y", alpha=0.25, zorder=0)
    _despine(ax)

    savefig(fig, out_path)


def plot_paired_diff(
    df: pd.DataFrame, horizon: int, models: list[str], out_path: Path
) -> None:
    """Per-session RMSE minus zero-model RMSE, all non-zero models.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form results with columns ``session_id``, ``model``,
        ``horizon``, ``rmse_full``. Must include a ``"zero"`` model row.
    horizon : int
        Horizon to plot.
    models : list of str
        Non-zero models to include, in display order.
    out_path : Path
        Output PDF path.
    """
    wide = _aligned(df, ["zero"] + models, horizon)
    if wide.empty or "zero" not in wide.columns:
        return

    zero_vals = wide["zero"].values.astype(float)
    rng = np.random.default_rng(1)

    diffs_list = [
        wide[m].values.astype(float) - zero_vals for m in models if m in wide.columns
    ]
    if not diffs_list:
        return
    ylo, yhi = _iqr_ylim(diffs_list)
    n_clipped = sum(int(np.sum((d < ylo) | (d > yhi))) for d in diffs_list)

    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.5), constrained_layout=True)
    ax.axhline(0, color="black", lw=0.8, zorder=1)

    plotted = [m for m in models if m in wide.columns]
    for xi, m in enumerate(plotted):
        diff = wide[m].values.astype(float) - zero_vals
        color = COLORS.get(m, "#333333")
        jitter = rng.uniform(-0.18, 0.18, size=len(diff))
        dot_colors = ["#2ca02c" if d < 0 else "#d62728" for d in diff]
        ax.scatter(
            xi + jitter, diff, color=dot_colors, s=18, alpha=0.75,
            linewidths=0.3, edgecolors="white", zorder=3,
        )
        ax.plot(
            [xi - 0.25, xi + 0.25], [np.mean(diff), np.mean(diff)],
            color=color, lw=2.0, zorder=4, solid_capstyle="round",
        )

    ax.set_ylim(ylo, yhi)
    ax.set_xticks(range(len(plotted)))
    ax.set_xticklabels([LABELS.get(m, m) for m in plotted], rotation=20, ha="right")
    ax.set_ylabel("RMSE − zero RMSE (z-score)")
    if n_clipped:
        ax.annotate(
            f'{n_clipped} point{"s" if n_clipped > 1 else ""} outside axis',
            xy=(0.99, 0.01), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=7, color="grey",
        )
    ax.grid(axis="y", alpha=0.25, zorder=0)
    _despine(ax)

    savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Horizon sweeps
# ---------------------------------------------------------------------------

def plot_rmse_vs_horizon(
    df: pd.DataFrame, models: list[str], horizons: list[int], out_path: Path
) -> None:
    """Mean +/- std RMSE vs prediction horizon, all models.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form results with columns ``model``, ``horizon``, ``rmse_full``.
    models : list of str
        Models to plot, in legend order.
    horizons : list of int
        Horizons to plot on the x-axis, in order.
    out_path : Path
        Output PDF path.
    """
    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.5), constrained_layout=True)

    for m in models:
        means, stds = [], []
        for h in horizons:
            sub = df[(df["model"] == m) & (df["horizon"] == h)]["rmse_full"].dropna()
            means.append(float(sub.mean()) if len(sub) else float("nan"))
            stds.append(float(sub.std()) if len(sub) else float("nan"))

        means_arr = np.array(means)
        stds_arr = np.array(stds)
        c = COLORS.get(m, "#333333")
        ax.plot(
            horizons, means_arr, marker="o", ls="-" if m != "zero" else ":",
            color=c, lw=1.8, label=LABELS.get(m, m),
        )
        if m != "zero":
            ax.fill_between(
                horizons, means_arr - stds_arr, means_arr + stds_arr,
                color=c, alpha=0.07,
            )

    ax.set_xlabel("Prediction horizon (frames)")
    ax.set_ylabel("Mean RMSE (z-score) ± std")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _despine(ax)

    savefig(fig, out_path)


def plot_skill_vs_horizon(
    df: pd.DataFrame, models: list[str], horizons: list[int], out_path: Path
) -> None:
    """Mean +/- std skill (vs zero) vs prediction horizon, all non-zero models.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form results with columns ``model``, ``horizon``,
        ``skill_vs_zero``.
    models : list of str
        Non-zero models to plot, in legend order.
    horizons : list of int
        Horizons to plot on the x-axis, in order.
    out_path : Path
        Output PDF path.
    """
    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.5), constrained_layout=True)

    for m in models:
        means, stds = [], []
        for h in horizons:
            sub = df[(df["model"] == m) & (df["horizon"] == h)]["skill_vs_zero"].dropna()
            means.append(float(sub.mean()) if len(sub) else float("nan"))
            stds.append(float(sub.std()) if len(sub) else float("nan"))

        means_arr = np.array(means)
        stds_arr = np.array(stds)
        c = COLORS.get(m, "#333333")
        ax.plot(horizons, means_arr, marker="o", ls="-", color=c, lw=1.8, label=LABELS.get(m, m))
        ax.fill_between(
            horizons, means_arr - stds_arr, means_arr + stds_arr, color=c, alpha=0.07
        )

    ax.axhline(0, color="black", lw=0.8, ls=":", label="Zero reference")
    ax.set_xlabel("Prediction horizon (frames)")
    ax.set_ylabel("Skill vs zero ± std\n(1 − RMSE / RMSE$_0$)")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _despine(ax)

    savefig(fig, out_path)
