"""
visualization.py
-----------------
Pure plotting library for the model comparison pipeline.

Every function takes already-loaded DataFrames or arrays and an output
path. Nothing here fits models, resolves paths beyond ``load_predictions``
and the ``out_path`` arguments, or loads config. Callers are responsible
for producing ``per_session_results``-style DataFrames (see
:mod:`fuspredict.evaluation.benchmark`) and for loading prediction arrays
via :func:`load_predictions`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from fuspredict.evaluation.stats import _aligned, rmse, significance_stars
from fuspredict.plot_utils import savefig as _savefig_png_pdf

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
    }
)

FIG_DPI = 300
FIG_W_SINGLE = 3.5
FIG_W_DOUBLE = 7.0

COLORS = {
    "zero": "#aaaaaa",
    "persistence": "#8c564b",
    "rolling_mean": "#9467bd",
    "pixel_ar": "#1f77b4",
    "frozen_pca_ar": "#ff7f0e",
    "frozen_ica_ar": "#d62728",
    "frozen_pca_rolling_mean": "#8c564b",
    "frozen_ica_rolling_mean": "#7f7f7f",
    "patch_lag_pca_ar": "#e377c2",
    "convlstm": "#2ca02c",
    "convlstm_pca_latent": "#17becf",
    "convlstm_ica_latent": "#bcbd22",
    "convlstm_vs_pca_oracle": "#17becf",
    "convlstm_vs_ica_oracle": "#bcbd22",
    "rolling_mean_vs_pca_oracle": "#9467bd",
    "rolling_mean_vs_ica_oracle": "#c5b0d5",
}

LABELS = {
    "zero": "Zero",
    "persistence": "Persistence (n+1)",
    "rolling_mean": "Rolling Mean",
    "pixel_ar": "Pixel AR",
    "frozen_pca_ar": "Frozen PCA-AR",
    "frozen_ica_ar": "Frozen ICA-AR",
    "frozen_pca_rolling_mean": "Frozen PCA-Rolling Mean",
    "frozen_ica_rolling_mean": "Frozen ICA-Rolling Mean",
    "patch_lag_pca_ar": "Patch-lag PCA-AR",
    "convlstm": "ConvLSTM",
    "convlstm_pca_latent": "ConvLSTM PCA-latent",
    "convlstm_ica_latent": "ConvLSTM ICA-latent",
    "convlstm_vs_pca_oracle": "ConvLSTM vs Denoised GT (PCA)",
    "convlstm_vs_ica_oracle": "ConvLSTM vs Denoised GT (ICA)",
    "rolling_mean_vs_pca_oracle": "Rolling Mean vs Denoised GT (PCA)",
    "rolling_mean_vs_ica_oracle": "Rolling Mean vs Denoised GT (ICA)",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _despine(ax: plt.Axes) -> None:
    """Remove top and right spines from an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig: plt.Figure, out_path: Path) -> None:
    """Save a figure as PNG+PDF via plot_utils.savefig, then close it."""
    _savefig_png_pdf(fig, Path(out_path), dpi=FIG_DPI)
    plt.close(fig)


def _iqr_ylim(
    arrays: list[np.ndarray], iqr_scale: float = 1.5, margin: float = 0.05
) -> tuple[float, float]:
    """Compute y-axis limits spanning the full range of pooled values, with a margin."""
    all_vals = np.concatenate([a.ravel() for a in arrays if len(a)])
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    span = hi - lo
    return lo - margin * span, hi + margin * span


def _rolling(x: np.ndarray, win: int = 5) -> np.ndarray:
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()


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


# ---------------------------------------------------------------------------
# Spatial (vessel / non-vessel) strip plot
# ---------------------------------------------------------------------------

def plot_spatial_strip(
    df: pd.DataFrame,
    horizon: int,
    models: list[str],
    out_path: Path,
    rmse_cols: tuple[str, str] = ("rmse_vessel", "rmse_nonvessel"),
) -> None:
    """Per-session RMSE split into vessel / non-vessel pixels, all models.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form results with columns ``session_id``, ``model``,
        ``horizon``, and the two columns named in ``rmse_cols``.
    horizon : int
        Horizon to plot.
    models : list of str
        Models to include, in display order.
    out_path : Path
        Output PDF path.
    rmse_cols : tuple of str
        (vessel, non-vessel) column names (default
        ``("rmse_vessel", "rmse_nonvessel")``). Pass
        ``("rmse_oracle_vessel", "rmse_oracle_nonvessel")`` to compare
        latent-reconstruction models on their oracle-decoded RMSE.
    """
    rng = np.random.default_rng(2)

    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W_DOUBLE, 3.5), constrained_layout=True, sharey=False
    )

    for ax, rmse_col, region_label in zip(
        axes, rmse_cols, ["Vessel pixels", "Non-vessel pixels"]
    ):
        wide = _aligned(df, models, horizon, rmse_col=rmse_col)
        if wide.empty:
            ax.set_xlabel(f"{region_label}: no data")
            continue

        all_finite = np.concatenate(
            [wide[m].values.astype(float) for m in models if m in wide.columns]
        )
        all_finite = all_finite[np.isfinite(all_finite)]
        ylo, yhi = _iqr_ylim([all_finite])

        if "zero" in wide.columns:
            zero_mean = float(np.mean(wide["zero"].values.astype(float)))
            ax.axhline(
                zero_mean, color=COLORS["zero"], lw=1.5, ls="--",
                label=f"Zero mean ({zero_mean:.3f})", zorder=1,
            )

        n_clipped = 0
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
            ax.plot(
                [xi - 0.25, xi + 0.25], [np.mean(vals), np.mean(vals)],
                color=color, lw=2.0, zorder=4, solid_capstyle="round",
            )
            n_clipped += int(np.sum((vals < ylo) | (vals > yhi)))

        ax.set_ylim(ylo, yhi)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABELS.get(m, m) for m in models], rotation=20, ha="right")
        ax.set_ylabel("RMSE (z-score)")
        ax.set_xlabel(region_label)
        if n_clipped:
            ax.annotate(
                f"{n_clipped} pts outside axis",
                xy=(0.99, 0.01), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color="grey",
            )
        ax.legend()
        ax.grid(axis="y", alpha=0.25, zorder=0)
        _despine(ax)

    savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Kernel-size comparison figures
# ---------------------------------------------------------------------------

KERNEL_COLORS = {0: "#4e79a7", 3: "#f28e2b", 5: "#e15759", 7: "#76b7b2"}


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


# ---------------------------------------------------------------------------
# Wilcoxon statistics table (rendering only; computation lives in
# fuspredict.evaluation.stats)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prediction loading (the one I/O function in this module)
# ---------------------------------------------------------------------------

def load_predictions(
    predictions_dir: Path,
    model: str,
    session_id: str,
    horizon: int,
    gt_key: str = "gt",
) -> tuple[np.ndarray, np.ndarray]:
    """Load (gt, pred) arrays for one (model, session, horizon) triple.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing ``{model}_{session_id}_h{horizon}.npz`` files,
        each with arrays ``gt`` and ``pred`` (and, for latent-reconstruction
        models, ``oracle_gt``) of shape ``(N, H, W)``.
    model : str
        Model name, as used in the npz filename.
    session_id : str
        Session ID, as used in the npz filename.
    horizon : int
        Prediction horizon, as used in the npz filename.
    gt_key : str
        Which array to use as "ground truth" (default ``"gt"``, the raw
        target frames). Pass ``"oracle_gt"`` to instead use the model's
        PCA/ICA-oracle reconstruction of the target frames, for comparing
        latent-reconstruction models against each other in reconstructed
        space rather than raw pixel space.

    Returns
    -------
    (gt, pred) : tuple of np.ndarray
        Ground-truth and predicted frame stacks, each ``(N, H, W)`` float32.
    """
    path = Path(predictions_dir) / f"{model}_{session_id}_h{horizon}.npz"
    with np.load(path) as z:
        return z[gt_key].astype(np.float32), z["pred"].astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial figures (single session, multiple models)
# ---------------------------------------------------------------------------

def plot_spatial_comparison(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    session_id: str,
    out_path: Path,
    sig_lim: float | None = None,
    gt_col_title: str = "Ground truth (mean)",
) -> None:
    """GT mean | prediction mean | residual grid across models.

    Parameters
    ----------
    predictions : dict
        ``{model_name: (gt, pred)}``, each array shape ``(N, H, W)``, as
        returned by :func:`load_predictions`.
    session_id : str
        Session identifier. Not rendered in the figure; accepted for a
        consistent call signature with the other spatial-figure functions.
    out_path : Path
        Output PDF path.
    sig_lim : float, optional
        Fixed color-scale limit (z-score) for the GT/prediction columns,
        shared ``vmin=-sig_lim, vmax=sig_lim``. Pass this to keep the scale
        consistent across multiple related figures (e.g. fig7 and fig7b so
        the same model's prediction renders identically in both). If
        omitted, computed from this call's own models as before.
    gt_col_title : str
        Title for the first column. Override when the "gt" array passed in
        ``predictions`` isn't literal ground truth — e.g. pass something
        like ``"Basis reconstruction (mean)"`` when using
        ``load_predictions(..., gt_key="oracle_gt")``, since that array is
        the frozen-basis reconstruction of the target frames (test-period
        frames run through a basis fit only on train-period data, so it
        carries the train-period mean, not true test-period ground truth).
    """
    models = list(predictions.keys())
    n_models = len(models)
    if n_models == 0:
        return

    CMAP = "RdBu_r"
    sig_vals = []
    computed = []
    for m in models:
        gt, pred = predictions[m]
        gt = gt.astype(np.float64)
        pred = pred.astype(np.float64)
        n = min(gt.shape[0], pred.shape[0])
        gt, pred = gt[:n], pred[:n]
        gt_mean = gt.mean(axis=0)
        pred_mean = pred.mean(axis=0)
        residual = pred_mean - gt_mean
        computed.append((m, gt_mean, pred_mean, residual))
        sig_vals += [np.percentile(np.abs(gt_mean), 98), np.percentile(np.abs(pred_mean), 98)]

    if sig_lim is None:
        sig_lim = float(np.percentile(sig_vals, 98))

    fig, axes = plt.subplots(
        n_models, 3,
        figsize=(FIG_W_DOUBLE, n_models * FIG_W_DOUBLE / 3),
        constrained_layout=True,
    )
    if n_models == 1:
        axes = axes[np.newaxis, :]

    col_titles = [gt_col_title, "Prediction (mean)", "Residual: pred − GT"]
    for ri, (m, gt_mean, pred_mean, residual) in enumerate(computed):
        res_lim = float(np.percentile(np.abs(residual), 98))
        for ci, (data, lim) in enumerate(
            [(gt_mean, sig_lim), (pred_mean, sig_lim), (residual, res_lim)]
        ):
            axes[ri, ci].imshow(data, cmap=CMAP, vmin=-lim, vmax=lim)
            axes[ri, ci].axis("off")
        if ri == 0:
            for ci, title in enumerate(col_titles):
                axes[ri, ci].set_title(title, fontweight="bold")
        axes[ri, 0].text(
            -0.04, 0.5, LABELS.get(m, m),
            transform=axes[ri, 0].transAxes,
            ha="right", va="center", fontweight="bold", rotation=90,
        )

    sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=-sig_lim, vmax=sig_lim))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.01, label="z-score")

    savefig(fig, out_path)


def plot_frame_recon_pred_compare(
    gt: np.ndarray,
    recon: np.ndarray,
    pred: np.ndarray,
    frame_i: int,
    model: str,
    out_path: Path,
) -> None:
    """Single-frame panel: GT | basis reconstruction | prediction | two residuals.

    Five panels for one model at one frame:
    ground truth, basis reconstruction (the model's frozen-basis
    reconstruction of that frame, e.g. ``oracle_gt``), prediction,
    residual (pred − GT), and residual (pred − reconstruction). The last
    residual isolates the error contributed by the temporal-prediction step
    alone, since it's scored against the same basis-limited target the
    prediction was implicitly aiming for.

    Parameters
    ----------
    gt : np.ndarray, shape (N, H, W)
        Raw ground-truth frame stack.
    recon : np.ndarray, shape (N, H, W)
        Frozen-basis reconstruction of the same frames (e.g. loaded via
        ``load_predictions(..., gt_key="oracle_gt")``).
    pred : np.ndarray, shape (N, H, W)
        Model prediction stack.
    frame_i : int
        Frame index to display.
    model : str
        Model name, used in the title and for ``LABELS``/``COLORS`` lookup.
    out_path : Path
        Output PDF path.
    """
    gt_f = gt[frame_i].astype(np.float64)
    recon_f = recon[frame_i].astype(np.float64)
    pred_f = pred[frame_i].astype(np.float64)
    resid_gt = pred_f - gt_f
    resid_recon = pred_f - recon_f

    sig_lim = float(np.percentile(
        np.abs(np.concatenate([gt_f.ravel(), recon_f.ravel(), pred_f.ravel()])), 98
    ))
    resid_lim = float(np.percentile(
        np.abs(np.concatenate([resid_gt.ravel(), resid_recon.ravel()])), 98
    )) or 1.0

    fig, axes = plt.subplots(1, 5, figsize=(FIG_W_DOUBLE, FIG_W_DOUBLE / 5 + 0.4), constrained_layout=True)
    CMAP = "RdBu_r"

    panels = [
        ("Ground truth", gt_f, sig_lim),
        ("Basis reconstruction", recon_f, sig_lim),
        ("Prediction", pred_f, sig_lim),
        ("Residual: pred − GT", resid_gt, resid_lim),
        ("Residual: pred − recon", resid_recon, resid_lim),
    ]
    for ax, (title, data, lim) in zip(axes, panels):
        ax.imshow(data, cmap=CMAP, vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{LABELS.get(model, model)} — frame {frame_i}", fontsize=9, fontweight="bold")
    savefig(fig, out_path)


def plot_spatial_rmse_diff(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path
) -> None:
    """Pixel-wise RMSE(model) minus RMSE(zero), one panel per model.

    The zero-model RMSE map is derived from the ground truth of the first
    model in ``predictions`` (all models share the same GT for a session).

    Parameters
    ----------
    predictions : dict
        ``{model_name: (gt, pred)}``, each array shape ``(N, H, W)``, as
        returned by :func:`load_predictions`.
    out_path : Path
        Output PDF path.
    """
    models = list(predictions.keys())
    if not models:
        return

    ref_gt = predictions[models[0]][0].astype(np.float64)
    rmse_zero = rmse(ref_gt, np.zeros_like(ref_gt), axis=0)

    diffs, err_lims = [], []
    for m in models:
        gt, pred = predictions[m]
        gt = gt.astype(np.float64)
        pred = pred.astype(np.float64)
        n = min(gt.shape[0], pred.shape[0])
        diff = rmse(pred[:n], gt[:n], axis=0) - rmse_zero
        diffs.append(diff)
        err_lims.append(float(np.percentile(np.abs(diff), 98)))

    err_lim = float(np.percentile(err_lims, 98))
    CMAP = "bwr"
    n_models = len(models)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols

    cell = FIG_W_DOUBLE / ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * cell, nrows * cell), constrained_layout=True
    )
    axes_flat = np.array(axes).ravel()

    for idx, (m, diff) in enumerate(zip(models, diffs, strict=True)):
        ax = axes_flat[idx]
        ax.imshow(diff, cmap=CMAP, vmin=-err_lim, vmax=err_lim)
        ax.axis("off")
        ax.set_title(LABELS.get(m, m), fontweight="bold")

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=-err_lim, vmax=err_lim))
    sm.set_array([])
    fig.colorbar(
        sm, ax=axes_flat[:n_models], orientation="horizontal",
        shrink=0.6, pad=0.04, aspect=40,
        label="RMSE(model) − RMSE(zero) [z-score]     blue = model better",
    )

    savefig(fig, out_path)


def plot_rmse_vs_time(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path
) -> None:
    """Rolling RMSE over test-frame index, all models, one session.

    Parameters
    ----------
    predictions : dict
        ``{model_name: (gt, pred)}``, each array shape ``(N, H, W)``, as
        returned by :func:`load_predictions`.
    out_path : Path
        Output PDF path.
    """
    models = list(predictions.keys())
    if not models:
        return

    min_t = min(gt.shape[0] for gt, _ in predictions.values())
    t = np.arange(min_t)

    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.0), constrained_layout=True)

    ref_gt = predictions[models[0]][0].astype(np.float64)[:min_t]
    rmse_z = rmse(ref_gt, np.zeros_like(ref_gt), axis=(1, 2))
    ax.plot(t, _rolling(rmse_z), lw=1.2, color=COLORS["zero"], ls=":", label=LABELS["zero"])

    lw_map = {"convlstm": 2.0}
    for m in models:
        gt, pred = predictions[m]
        gt = gt.astype(np.float64)[:min_t]
        pred = pred.astype(np.float64)[:min_t]
        rmse_m = rmse(pred, gt, axis=(1, 2))
        ax.plot(
            t, _rolling(rmse_m),
            lw=lw_map.get(m, 1.6),
            color=COLORS.get(m, "#333333"),
            label=LABELS.get(m, m),
        )

    ax.set_xlabel("Target frame (test set)")
    ax.set_ylabel("RMSE (z-score)")
    ax.legend()
    ax.grid(alpha=0.2)
    _despine(ax)

    savefig(fig, out_path)


def plot_predictions_vs_time(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path
) -> None:
    """Frame-mean signal over time: ground truth vs. each model's prediction.

    Parameters
    ----------
    predictions : dict
        ``{model_name: (gt, pred)}``, each array shape ``(N, H, W)``, as
        returned by :func:`load_predictions`.
    out_path : Path
        Output PDF path.
    """
    models = list(predictions.keys())
    if not models:
        return

    min_t = min(gt.shape[0] for gt, _ in predictions.values())
    t = np.arange(min_t)

    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.0), constrained_layout=True)

    ref_gt = predictions[models[0]][0].astype(np.float64)[:min_t]
    gt_mean = ref_gt.mean(axis=(1, 2))
    ax.plot(t, gt_mean, lw=1.4, color="black", label="Ground truth")

    lw_map = {"convlstm": 2.0}
    for m in models:
        _, pred = predictions[m]
        pred = pred.astype(np.float64)[:min_t]
        pred_mean = pred.mean(axis=(1, 2))
        ax.plot(
            t, pred_mean,
            lw=lw_map.get(m, 1.6),
            color=COLORS.get(m, "#333333"),
            label=LABELS.get(m, m),
            alpha=0.85,
        )

    ax.set_xlabel("Target frame (test set)")
    ax.set_ylabel("Mean signal (z-score)")
    ax.legend()
    ax.grid(alpha=0.2)
    _despine(ax)

    savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Triplet video (GT | prediction | residual)
# ---------------------------------------------------------------------------

def make_triplet_video(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    session_id: str = "",
    frame_indices: np.ndarray | None = None,
    fps: float = 2.5,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    dpi: int = 120,
) -> None:
    """Render GT | prediction | residual as a side-by-side mp4 video.

    Parameters
    ----------
    gt : np.ndarray
        Ground-truth frames, shape ``(N, H, W)``.
    pred : np.ndarray
        Predicted frames, shape ``(N, H, W)``.
    out_path : Path
        Output ``.mp4`` path; parent directories are created automatically.
    session_id : str
        Label shown in the figure title.
    frame_indices : np.ndarray, optional
        Indices into the first axis of ``gt``/``pred`` to render.
        Defaults to all frames ``np.arange(N)``.
    fps : float
        Output video frame rate (default: 2.5).
    vmin, vmax : float, optional
        Colormap limits in z-score units. If omitted, symmetric 99th-percentile
        of |gt| over the selected frames is used.
    cmap : str
        Matplotlib colormap name (default: ``"RdBu_r"``).
    dpi : int
        Figure DPI for each rendered frame (default: 120).
    """
    import cv2

    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    N = gt.shape[0]

    if frame_indices is None:
        frame_indices = np.arange(N)

    indices = [int(i) for i in frame_indices if 0 <= int(i) < N]
    if not indices:
        return

    if vmin is None or vmax is None:
        abs_max = float(np.percentile(np.abs(gt[indices]), 99))
        if vmin is None:
            vmin = -abs_max
        if vmax is None:
            vmax = abs_max

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True, dpi=dpi)
    title = f"{session_id}  |  GT | Prediction | Residual" if session_id else "GT | Prediction | Residual"
    fig.suptitle(title, fontsize=10)

    ims = []
    dummy = np.zeros_like(gt[0])
    for ax, col_title in zip(axes, ["Ground truth", "Prediction", "Residual (GT − pred)"]):
        ax.set_title(col_title, fontsize=9)
        ax.axis("off")
        ims.append(ax.imshow(dummy, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                             interpolation="nearest"))
    cbar_ax = fig.add_axes([0.02, 0.08, 0.96, 0.03])
    fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal", label="z-score")
    frame_text = fig.text(0.5, 0.97, "", ha="center", va="top", fontsize=8, color="#555555")

    # probe frame dimensions
    ims[0].set_data(dummy)
    frame_text.set_text("probe")
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w_px, h_px = fig.canvas.get_width_height()
    h_px, w_px = buf.reshape(h_px, w_px, 4).shape[:2]

    out_path = Path(out_path).with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_px, h_px))

    for rank, t in enumerate(indices):
        residual = gt[t] - pred[t]
        ims[0].set_data(gt[t])
        ims[1].set_data(pred[t])
        ims[2].set_data(residual)
        frame_text.set_text(f"frame {t}  ({rank + 1}/{len(indices)})")

        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)
        writer.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))

    writer.release()
    plt.close(fig)
