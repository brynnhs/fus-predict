"""
model_report_plots.py
---------------------
Visualization utilities for fUS prediction model results.

All functions work on numpy arrays and pandas DataFrames.
No file I/O or model fitting happens here.

Typical usage
-------------
    from fus_predict.model_report_plots import (
        plot_horizon_metric_grid,
        plot_prediction_examples,
    )

    # Plot RMSE vs horizon for multiple models
    plot_horizon_metric_grid(
        horizon_df,
        title_prefix="Linear baselines — session Se01",
        save_paths=["figures/horizon_metrics.png"],
    )

    # Plot GT vs prediction comparison grids
    plot_prediction_examples(
        dataset=test_ds,
        model_fns={
            "zero":       lambda ctx: predict_zero(ctx, K=10),
            "pixel_ar":   lambda ctx: predict_pixel_ar(ctx, params, K=10),
            "convlstm":   lambda ctx: convlstm_predict(ctx),
        },
        horizons=[1, 5, 10],
        output_dir="figures/predictions/",
    )
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CUSTOM_PALETTE = [
    "#D0E9F1",
    "#A3D0D4",
    "#48A7C8",
    "#041C3C",
    "#2A356B",
    "#565AA0",
]


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _robust_limits(
    x: np.ndarray,
    q_low: float = 1.0,
    q_high: float = 99.0,
) -> tuple[float, float]:
    arr  = np.asarray(x, dtype=np.float32)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, q_low))
    hi = float(np.percentile(vals, q_high))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        lo, hi = float(np.min(vals)), float(np.max(vals))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return 0.0, 1.0
    return lo, hi


def _robust_abs_limit(x: np.ndarray, q: float = 99.5, eps: float = 1e-8) -> float:
    arr   = np.abs(np.asarray(x, dtype=np.float32))
    vals  = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 1.0
    limit = float(np.percentile(vals, q))
    if not np.isfinite(limit) or limit <= eps:
        limit = float(np.max(vals))
    if not np.isfinite(limit) or limit <= eps:
        return 1.0
    return limit


def _finalize_plot(
    fig: plt.Figure,
    save_paths: list[str | Path] | None,
    show_inline: bool,
) -> None:
    fig.tight_layout()
    if save_paths:
        for path in save_paths:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
    if show_inline:
        plt.show()
    else:
        plt.close(fig)


def _pick_evenly_spaced_indices(n_items: int, max_items: int) -> list[int]:
    if n_items <= 0:
        return []
    count = max(1, min(int(max_items), int(n_items)))
    if count == 1:
        return [0]
    return np.unique(np.linspace(0, n_items - 1, num=count, dtype=np.int64)).astype(int).tolist()


def _format_model_display_name(model_name: str) -> str:
    name = str(model_name)
    if name == "persistence":
        return "Persistence"
    m = re.match(r"^pixel_ar_p(\d+)$", name)
    if m:
        return f"Pixel-AR (p={int(m.group(1))})"
    m = re.match(r"^(pca_var|pca_ar)_d(\d+)_p(\d+)$", name)
    if m:
        family, dim, lag = m.groups()
        return f"{'PCA-VAR' if family == 'pca_var' else 'PCA-AR'} (d={int(dim)}, p={int(lag)})"
    return name.replace("_", " ")


def _add_row_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.03, 0.95, str(label),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10, fontweight="bold", color="black",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
              "edgecolor": "none", "alpha": 0.82},
    )


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_horizon_metric_grid(
    horizon_df: pd.DataFrame,
    *,
    title_prefix: str,
    save_paths: list[str | Path] | None = None,
    show_inline: bool = False,
    exclude_models: list[str] | None = None,
    metrics: list[tuple[str, str]] | None = None,
) -> plt.Figure:
    """
    Plot one panel per metric showing metric vs forecast horizon for each model.

    Parameters
    ----------
    horizon_df : pd.DataFrame
        Must have columns: model, horizon, and at least one metric column
        (e.g. RMSE_mean). One row per (model, horizon) combination.
    title_prefix : str
        Figure suptitle.
    save_paths : list of path-like, optional
        Paths to save the figure.
    show_inline : bool
        If True, call plt.show(). Otherwise close the figure.
    exclude_models : list of str, optional
        Model names to exclude from the plot.
    metrics : list of (col_name, display_label), optional
        Which metric columns to plot. Defaults to MSE, RMSE, MAE, R2.
    """
    if horizon_df.empty:
        raise ValueError("horizon_df is empty.")
    missing = {"model", "horizon"} - set(horizon_df.columns)
    if missing:
        raise ValueError(f"horizon_df missing columns: {sorted(missing)}")

    metric_specs = metrics or [
        ("MSE_mean",  "MSE"),
        ("RMSE_mean", "RMSE"),
        ("MAE_mean",  "MAE"),
        ("R2_mean",   "R2"),
    ]
    available = [(col, lbl) for col, lbl in metric_specs if col in horizon_df.columns]
    if not available:
        raise ValueError("No requested metric columns found in horizon_df.")

    exclude = {str(x) for x in (exclude_models or [])}
    grouped = {
        str(m): grp.sort_values("horizon")
        for m, grp in horizon_df.groupby("model", sort=False)
        if str(m) not in exclude
    }
    if not grouped:
        raise ValueError("No models remain after filtering.")

    ncols = 2 if len(available) > 1 else 1
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.2 * nrows), squeeze=False)

    for ax, (col, lbl) in zip(axes.ravel(), available):
        for model_name, grp in grouped.items():
            ax.plot(grp["horizon"].values, grp[col].values, marker="o", linewidth=1.8, label=model_name)
        ax.set_title(f"{lbl} vs horizon")
        ax.set_xlabel("Horizon")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)

    for ax in axes.ravel()[len(available):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(4, len(labels)), fontsize=9, frameon=False)
        fig.suptitle(title_prefix, y=1.02)
    else:
        fig.suptitle(title_prefix)

    _finalize_plot(fig, save_paths=save_paths, show_inline=show_inline)
    return fig


def plot_prediction_examples(
    dataset,
    model_fns: dict[str, callable],
    horizons: list[int],
    output_dir: str | Path,
    *,
    example_indices: list[int] | None = None,
    max_examples: int = 3,
    combine_examples: bool = False,
    show_inline: bool = False,
) -> list[Path]:
    """
    Save prediction comparison grids and residual grids for selected examples.

    Each model function is called as `model_fns[name](context)` and must
    return a numpy array of shape (max_horizon, 1, H, W) — one prediction
    per horizon. The caller is responsible for any pre/post-processing.

    Parameters
    ----------
    dataset : FUSForecastWindowDataset
        Test dataset. Items must be (context, target) or (context, target, meta).
    model_fns : dict[str, callable]
        Mapping from model name to predict function: context → (H, 1, H, W).
    horizons : list of int
        Forecast horizons to visualize (1-indexed, e.g. [1, 5, 10]).
    output_dir : path-like
        Directory to save figures.
    example_indices : list of int, optional
        Which dataset indices to plot. Defaults to evenly spaced selection.
    max_examples : int
        Maximum number of examples when example_indices is None.
    combine_examples : bool
        If True, combine all examples into a single figure per model.
    show_inline : bool
        If True, call plt.show() instead of closing figures.

    Returns
    -------
    list of Path
        Paths to saved figure files.
    """
    if not model_fns:
        raise ValueError("model_fns must not be empty.")

    horizon_list = sorted({int(h) for h in horizons if int(h) > 0})
    if not horizon_list:
        raise ValueError("At least one positive horizon is required.")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if example_indices is None:
        example_indices = _pick_evenly_spaced_indices(len(dataset), max_examples)
    if not example_indices:
        raise ValueError("No example indices available.")

    model_names   = list(model_fns.keys())
    display_names = {name: _format_model_display_name(name) for name in model_names}
    max_h         = max(horizon_list)
    saved_paths: list[Path] = []

    # Build prediction payloads
    example_payloads = []
    for example_idx in example_indices:
        item    = dataset[int(example_idx)]
        context = _to_numpy(item[0])
        target  = _to_numpy(item[1])

        if target.shape[0] < max_h:
            raise ValueError(
                f"Target horizon {target.shape[0]} < requested max horizon {max_h}."
            )

        gt_frames = np.asarray(target[[h - 1 for h in horizon_list], 0], dtype=np.float32)

        preds_by_model: dict[str, np.ndarray] = {}
        for name, fn in model_fns.items():
            pred      = np.asarray(fn(context))
            pred_sel  = np.asarray(pred[[h - 1 for h in horizon_list], 0], dtype=np.float32)
            preds_by_model[name] = pred_sel

        example_payloads.append({
            "example_idx":   int(example_idx),
            "gt_frames":     gt_frames,
            "preds_by_model": preds_by_model,
        })

    # Shared display limits across all examples
    all_frames    = np.concatenate([
        np.concatenate([p["gt_frames"].reshape(-1)] +
                       [v.reshape(-1) for v in p["preds_by_model"].values()])
        for p in example_payloads
    ])
    all_residuals = np.concatenate([
        np.concatenate([(v - p["gt_frames"]).reshape(-1)
                        for v in p["preds_by_model"].values()])
        for p in example_payloads
    ])
    vmin, vmax       = _robust_limits(all_frames)
    residual_scale   = _robust_abs_limit(all_residuals)

    if combine_examples:
        # All examples side by side in one figure
        ncols_pred = len(example_payloads) * (1 + len(model_names))
        nrows      = len(horizon_list)
        fig_pred, axes_pred = plt.subplots(
            nrows, ncols_pred,
            figsize=(2.2 * ncols_pred, 2.5 * nrows),
            squeeze=False,
        )
        for ex_pos, payload in enumerate(example_payloads):
            ex_idx   = payload["example_idx"]
            gt       = payload["gt_frames"]
            preds    = payload["preds_by_model"]
            base_col = ex_pos * (1 + len(model_names))

            for row_i, h in enumerate(horizon_list):
                ax = axes_pred[row_i, base_col]
                ax.imshow(gt[row_i], cmap="gray", vmin=vmin, vmax=vmax)
                if row_i == 0:
                    ax.set_title(f"GT\nw={ex_idx}")
                if ex_pos == 0:
                    _add_row_label(ax, f"h={h}")
                ax.axis("off")

                for m_off, name in enumerate(model_names, start=1):
                    pred_frame = preds[name][row_i]
                    rmse       = float(np.sqrt(np.mean((pred_frame - gt[row_i]) ** 2)))
                    ax = axes_pred[row_i, base_col + m_off]
                    ax.imshow(pred_frame, cmap="gray", vmin=vmin, vmax=vmax)
                    if row_i == 0:
                        ax.set_title(f"{display_names[name]}\nw={ex_idx}")
                    ax.set_xlabel(f"RMSE={rmse:.3f}", fontsize=8)
                    ax.axis("off")

        range_tag = f"{example_payloads[0]['example_idx']:04d}_to_{example_payloads[-1]['example_idx']:04d}"
        fig_pred.suptitle(f"Predictions | windows={', '.join(str(p['example_idx']) for p in example_payloads)}")
        pred_path = out_root / f"predictions_window_{range_tag}.png"
        _finalize_plot(fig_pred, save_paths=[pred_path], show_inline=show_inline)
        saved_paths.append(pred_path)

        # Residual grid
        ncols_res = len(example_payloads) * len(model_names)
        fig_res, axes_res = plt.subplots(
            len(horizon_list), max(1, ncols_res),
            figsize=(2.2 * max(1, ncols_res), 2.5 * len(horizon_list)),
            squeeze=False,
        )
        for ex_pos, payload in enumerate(example_payloads):
            ex_idx = payload["example_idx"]
            gt     = payload["gt_frames"]
            preds  = payload["preds_by_model"]
            base   = ex_pos * len(model_names)

            for row_i, h in enumerate(horizon_list):
                for m_off, name in enumerate(model_names):
                    resid = preds[name][row_i] - gt[row_i]
                    rmse  = float(np.sqrt(np.mean(resid ** 2)))
                    ax    = axes_res[row_i, base + m_off]
                    ax.imshow(resid, cmap="bwr", vmin=-residual_scale, vmax=residual_scale)
                    if row_i == 0:
                        ax.set_title(f"{display_names[name]}\nw={ex_idx}")
                    if ex_pos == 0 and m_off == 0:
                        _add_row_label(ax, f"h={h}")
                    ax.set_xlabel(f"RMSE={rmse:.3f}", fontsize=8)
                    ax.axis("off")

        fig_res.suptitle(f"Residuals (pred - GT) | windows={', '.join(str(p['example_idx']) for p in example_payloads)}")
        resid_path = out_root / f"residuals_window_{range_tag}.png"
        _finalize_plot(fig_res, save_paths=[resid_path], show_inline=show_inline)
        saved_paths.append(resid_path)
        return saved_paths

    # One figure per example
    for payload in example_payloads:
        ex_idx = payload["example_idx"]
        gt     = payload["gt_frames"]
        preds  = payload["preds_by_model"]

        # Prediction comparison grid
        nrows = 1 + len(model_names)
        ncols = len(horizon_list)
        fig_pred, axes_pred = plt.subplots(
            nrows, ncols,
            figsize=(2.4 * ncols, 2.4 * nrows),
            squeeze=False,
        )
        for col_i, h in enumerate(horizon_list):
            ax = axes_pred[0, col_i]
            ax.imshow(gt[col_i], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"GT h={h}")
            ax.axis("off")

        for row_i, name in enumerate(model_names, start=1):
            for col_i, h in enumerate(horizon_list):
                pred_frame = preds[name][col_i]
                rmse       = float(np.sqrt(np.mean((pred_frame - gt[col_i]) ** 2)))
                ax = axes_pred[row_i, col_i]
                ax.imshow(pred_frame, cmap="gray", vmin=vmin, vmax=vmax)
                if col_i == 0:
                    ax.set_ylabel(display_names[name], rotation=0,
                                  ha="right", va="center", labelpad=30)
                ax.set_xlabel(f"RMSE={rmse:.3f}", fontsize=8)
                ax.axis("off")

        fig_pred.suptitle(f"Predictions | window={ex_idx}")
        pred_path = out_root / f"predictions_window_{ex_idx:04d}.png"
        _finalize_plot(fig_pred, save_paths=[pred_path], show_inline=show_inline)
        saved_paths.append(pred_path)

        # Residual grid
        fig_res, axes_res = plt.subplots(
            len(model_names), ncols,
            figsize=(2.4 * ncols, 2.4 * max(1, len(model_names))),
            squeeze=False,
        )
        for row_i, name in enumerate(model_names):
            for col_i, h in enumerate(horizon_list):
                resid = preds[name][col_i] - gt[col_i]
                rmse  = float(np.sqrt(np.mean(resid ** 2)))
                ax    = axes_res[row_i, col_i]
                ax.imshow(resid, cmap="bwr", vmin=-residual_scale, vmax=residual_scale)
                if row_i == 0:
                    ax.set_title(f"h={h}")
                if col_i == 0:
                    ax.set_ylabel(display_names[name], rotation=0,
                                  ha="right", va="center", labelpad=30)
                ax.set_xlabel(f"RMSE={rmse:.3f}", fontsize=8)
                ax.axis("off")

        fig_res.suptitle(f"Residuals (pred - GT) | window={ex_idx}")
        resid_path = out_root / f"residuals_window_{ex_idx:04d}.png"
        _finalize_plot(fig_res, save_paths=[resid_path], show_inline=show_inline)
        saved_paths.append(resid_path)

    return saved_paths