"""
kernel_plots.py
----------------
Spatial-smoothing-kernel comparison figures and the Wilcoxon stats table
renderer (rendering only; computation lives in
:mod:`fuspredict.evaluation.stats`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuspredict.evaluation.stats import significance_stars
from fuspredict.evaluation.visualization._common import (
    COLORS,
    FIG_W_DOUBLE,
    FIG_W_SINGLE,
    KERNEL_COLORS,
    LABELS,
    _despine,
    _iqr_ylim,
    savefig,
)


def plot_strip_by_kernel(
    df: pd.DataFrame,
    horizon: int,
    models: list[str],
    kernels: list[int],
    out_path: Path,
) -> None:
    """Per-session overall RMSE strip plot with kernel size on x-axis, faceted by model."""
    h_df = df[df["horizon"] == horizon].copy()

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(2.6 * n_models, 5.5),
        constrained_layout=True,
        sharey=True,
    )
    if n_models == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    all_vals_global = [
        h_df[h_df["kernel"] == k]["rmse_full"].dropna().values
        for k in kernels
    ]
    ylo, yhi = _iqr_ylim([v for v in all_vals_global if len(v)])

    for ax, model in zip(axes, models):
        m_df = h_df[h_df["model"] == model]
        all_vals = [m_df[m_df["kernel"] == k]["rmse_full"].dropna().values for k in kernels]

        for xi, (k, vals) in enumerate(zip(kernels, all_vals)):
            if len(vals) == 0:
                continue
            color = KERNEL_COLORS[k]
            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(
                xi + jitter, vals,
                color=color, s=28, alpha=0.75,
                linewidths=0.4, edgecolors="white", zorder=3,
            )
            mn = float(np.mean(vals))
            ax.plot(
                [xi - 0.24, xi + 0.24], [mn, mn],
                color=color, lw=2.6, zorder=4, solid_capstyle="round",
            )

        ax.set_xlim(-0.6, len(kernels) - 0.4)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(range(len(kernels)))
        ax.set_xticklabels([str(k) for k in kernels], fontsize=11)
        ax.set_xlabel("k", fontsize=12)
        ax.set_title(LABELS.get(model, model), fontsize=13, pad=10)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.tick_params(axis="y", labelsize=11)
        _despine(ax)

    axes[0].set_ylabel("RMSE (z-score)", fontsize=13)
    savefig(fig, out_path)


def plot_paired_diff_by_kernel(
    df: pd.DataFrame,
    horizon: int,
    models: list[str],
    kernels: list[int],
    out_path: Path,
) -> None:
    """Per-session RMSE(kX) − RMSE(k0), faceted by model."""
    h_df = df[df["horizon"] == horizon].copy()
    non_zero_kernels = [k for k in kernels if k != 0]

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(FIG_W_DOUBLE, 3.0),
        constrained_layout=True,
        sharey=True,
    )
    if n_models == 1:
        axes = [axes]

    rng = np.random.default_rng(7)

    all_diffs: list[np.ndarray] = []
    for model in models:
        m_df = h_df[h_df["model"] == model]
        baseline = m_df[m_df["kernel"] == 0].set_index("session_id")["rmse_full"]
        for k in non_zero_kernels:
            comp = m_df[m_df["kernel"] == k].set_index("session_id")["rmse_full"]
            common = baseline.index.intersection(comp.index)
            if len(common):
                all_diffs.append((comp.loc[common] - baseline.loc[common]).values)

    if not all_diffs:
        return
    ylo, yhi = _iqr_ylim(all_diffs)

    for ax, model in zip(axes, models):
        m_df = h_df[h_df["model"] == model]
        baseline = m_df[m_df["kernel"] == 0].set_index("session_id")["rmse_full"]

        ax.axhline(0, color="black", lw=0.8, zorder=1)

        for xi, k in enumerate(non_zero_kernels):
            comp = m_df[m_df["kernel"] == k].set_index("session_id")["rmse_full"]
            common = baseline.index.intersection(comp.index)
            if not len(common):
                continue
            diff = (comp.loc[common] - baseline.loc[common]).values
            color = KERNEL_COLORS[k]
            jitter = rng.uniform(-0.15, 0.15, size=len(diff))
            dot_colors = ["#2ca02c" if d < 0 else "#d62728" for d in diff]
            ax.scatter(
                xi + jitter, diff,
                color=dot_colors, s=14, alpha=0.75,
                linewidths=0.3, edgecolors="white", zorder=3,
            )
            ax.plot(
                [xi - 0.2, xi + 0.2], [float(np.mean(diff)), float(np.mean(diff))],
                color=color, lw=2.0, zorder=4, solid_capstyle="round",
            )

        ax.set_ylim(ylo, yhi)
        ax.set_xticks(range(len(non_zero_kernels)))
        ax.set_xticklabels([f"k={k}" for k in non_zero_kernels])
        ax.set_title(LABELS.get(model, model), fontsize=8)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        _despine(ax)

    axes[0].set_ylabel("RMSE(kX) − RMSE(k0)\n(z-score, green = improvement)")
    savefig(fig, out_path)


def _draw_median_lines_by_kernel(
    ax: plt.Axes, h_df: pd.DataFrame, models: list[str], kernels: list[int], col: str
) -> None:
    for model in models:
        medians = []
        for k in kernels:
            vals = h_df[(h_df["model"] == model) & (h_df["kernel"] == k)][col].dropna()
            medians.append(float(vals.median()) if len(vals) else float("nan"))
        ax.plot(kernels, medians, marker="o", lw=1.8, color=COLORS.get(model, "#333333"),
                label=LABELS.get(model, model))
    ax.set_xlabel("Kernel size")
    ax.set_xticks(kernels)
    ax.grid(alpha=0.2)
    _despine(ax)


def plot_median_vs_kernel(
    df: pd.DataFrame,
    horizon: int,
    models: list[str],
    kernels: list[int],
    out_path: Path,
) -> None:
    """Two figures: overall RMSE and vessel vs non-vessel, median across sessions."""
    h_df = df[df["horizon"] == horizon].copy()

    # Overall (single panel)
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 3.0), constrained_layout=True)
    _draw_median_lines_by_kernel(ax, h_df, models, kernels, "rmse_full")
    ax.set_ylabel("Median RMSE (z-score)")
    ax.legend(loc="upper right", fontsize=7)
    savefig(fig, out_path)

    # Vessel vs non-vessel (two panels)
    fig2, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, 3.0), constrained_layout=True, sharey=True)
    region_cols = {"rmse_vessel": "Vessel", "rmse_nonvessel": "Non-vessel"}
    for ax, (col, title) in zip(axes, region_cols.items()):
        _draw_median_lines_by_kernel(ax, h_df, models, kernels, col)
        ax.set_title(title)
    axes[0].set_ylabel("Median RMSE (z-score)")
    axes[1].legend(loc="upper right", fontsize=7)
    split_path = Path(str(out_path).replace("fig3_median_vs_kernel", "fig3b_median_vs_kernel_by_region"))
    savefig(fig2, split_path)


def plot_wilcoxon_table(stats_df: pd.DataFrame, horizon: int, out_path: Path) -> None:
    """Render a Wilcoxon stats table (full-frame region only) as a figure.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of :func:`fuspredict.evaluation.stats.compute_wilcoxon`.
    horizon : int
        Horizon to render.
    out_path : Path
        Output PDF path.
    """
    mask = (stats_df["horizon"] == horizon) & (stats_df["region"] == "full")
    sub = stats_df[mask].copy()
    if sub.empty:
        return

    display = sub.copy()
    display["p_value"] = display["p_value"].map(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )
    display["stars"] = sub["p_value"].map(significance_stars)
    display["median_diff"] = display["median_diff"].map(lambda x: f"{x:+.4f}")
    display["ci_95"] = sub.apply(
        lambda r: f'[{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}]', axis=1
    )
    display["W"] = display["W"].map(lambda x: f"{x:.0f}")
    display["model_A"] = display["model_A"].map(lambda m: LABELS.get(m, m))

    cols = ["model_A", "model_B", "n_sessions", "median_diff", "ci_95", "W", "p_value", "stars"]
    labels = ["Model A", "vs Model B", "N", "Median(A−B)", "95% CI", "W", "p-value", ""]
    present = [c for c in cols if c in display.columns]
    p_labels = [labels[cols.index(c)] for c in present]
    table_data = display[present].values.tolist()

    n_rows = len(table_data)
    n_cols = len(present)
    fig, ax = plt.subplots(
        figsize=(max(FIG_W_DOUBLE, n_cols * 1.4), 0.35 * (n_rows + 1) + 0.4),
        constrained_layout=True,
    )
    ax.axis("off")
    ax.set_title(f"Wilcoxon tests — full frame, h={horizon}", fontsize=8, pad=4)
    tbl = ax.table(cellText=table_data, colLabels=p_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)

    for j in range(n_cols):
        tbl[0, j].set_facecolor("white")
        tbl[0, j].set_text_props(fontweight="bold")
        tbl[0, j].set_edgecolor("#cccccc")

    for i, row in enumerate(sub.itertuples(), start=1):
        for j in range(n_cols):
            tbl[i, j].set_facecolor("#f2f2f2" if row.p_value < 0.05 else "white")
            tbl[i, j].set_edgecolor("#cccccc")

    savefig(fig, out_path)
