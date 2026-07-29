"""
spatial_plots.py
-----------------
Spatial (vessel/non-vessel) RMSE plots and per-session spatial-map figures
(GT vs. prediction vs. residual, RMSE-vs-time, predictions-vs-time).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from fuspredict.evaluation.stats import _aligned, rmse
from fuspredict.evaluation.visualization._common import (
    COLORS,
    FIG_W_DOUBLE,
    LABELS,
    _despine,
    _iqr_ylim,
    _rolling,
    savefig,
)


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
        returned by :func:`fuspredict.data.loading.load_predictions`.
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
        returned by :func:`fuspredict.data.loading.load_predictions`.
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
        returned by :func:`fuspredict.data.loading.load_predictions`.
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
        returned by :func:`fuspredict.data.loading.load_predictions`.
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
