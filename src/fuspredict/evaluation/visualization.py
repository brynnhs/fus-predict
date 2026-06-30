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
    "full_frame_pca_ar": "#ff7f0e",
    "patch_lag_pca_ar": "#e377c2",
    "convlstm": "#2ca02c",
}

LABELS = {
    "zero": "Zero",
    "persistence": "Persistence (n+1)",
    "rolling_mean": "Rolling Mean",
    "pixel_ar": "Pixel AR",
    "full_frame_pca_ar": "Full-frame PCA-AR",
    "patch_lag_pca_ar": "Patch-lag PCA-AR",
    "convlstm": "ConvLSTM",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _despine(ax: plt.Axes) -> None:
    """Remove top and right spines from an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig: plt.Figure, out_path: Path) -> None:
    """Save a figure as PDF and close it, creating parent directories as needed."""
    out_path = Path(out_path).with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def _iqr_ylim(
    arrays: list[np.ndarray], iqr_scale: float = 1.5, margin: float = 0.05
) -> tuple[float, float]:
    """Compute y-axis limits from the IQR of pooled values, with a margin."""
    all_vals = np.concatenate([a.ravel() for a in arrays if len(a)])
    q25, q75 = np.percentile(all_vals, [25, 75])
    iqr = q75 - q25
    lo, hi = q25 - iqr_scale * iqr, q75 + iqr_scale * iqr
    span = hi - lo
    return lo - margin * span, hi + margin * span


def _rolling(x: np.ndarray, win: int = 5) -> np.ndarray:
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()


# ---------------------------------------------------------------------------
# Per-session strip plots
# ---------------------------------------------------------------------------

def plot_rmse_strip(
    df: pd.DataFrame, horizon: int, models: list[str], out_path: Path
) -> None:
    """Per-session RMSE jitter/strip plot at a given horizon, all models.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form results with columns ``session_id``, ``model``,
        ``horizon``, ``rmse_full``.
    horizon : int
        Horizon to plot.
    models : list of str
        Models to include, in display order.
    out_path : Path
        Output PDF path (suffix is normalized).
    """
    wide = _aligned(df, models, horizon)
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
    df: pd.DataFrame, horizon: int, models: list[str], out_path: Path
) -> None:
    """Per-session RMSE split into vessel / non-vessel pixels, all models.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form results with columns ``session_id``, ``model``,
        ``horizon``, ``rmse_vessel``, ``rmse_nonvessel``.
    horizon : int
        Horizon to plot.
    models : list of str
        Models to include, in display order.
    out_path : Path
        Output PDF path.
    """
    rng = np.random.default_rng(2)

    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W_DOUBLE, 3.5), constrained_layout=True, sharey=False
    )

    for ax, rmse_col, region_label in zip(
        axes, ["rmse_vessel", "rmse_nonvessel"], ["Vessel pixels", "Non-vessel pixels"]
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
    predictions_dir: Path, model: str, session_id: str, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load (gt, pred) arrays for one (model, session, horizon) triple.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing ``{model}_{session_id}_h{horizon}.npz`` files,
        each with arrays ``gt`` and ``pred`` of shape ``(N, H, W)``.
    model : str
        Model name, as used in the npz filename.
    session_id : str
        Session ID, as used in the npz filename.
    horizon : int
        Prediction horizon, as used in the npz filename.

    Returns
    -------
    (gt, pred) : tuple of np.ndarray
        Ground-truth and predicted frame stacks, each ``(N, H, W)`` float32.
    """
    path = Path(predictions_dir) / f"{model}_{session_id}_h{horizon}.npz"
    with np.load(path) as z:
        return z["gt"].astype(np.float32), z["pred"].astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial figures (single session, multiple models)
# ---------------------------------------------------------------------------

def plot_spatial_comparison(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    session_id: str,
    out_path: Path,
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

    sig_lim = float(np.percentile(sig_vals, 98))

    fig, axes = plt.subplots(
        n_models, 3,
        figsize=(FIG_W_DOUBLE, n_models * FIG_W_DOUBLE / 3),
        constrained_layout=True,
    )
    if n_models == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Ground truth (mean)", "Prediction (mean)", "Residual: pred − GT"]
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
