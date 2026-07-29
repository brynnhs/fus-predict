"""
_common.py
----------
Shared matplotlib style, color/label lookup tables, and small helpers used
across the visualization submodules.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

KERNEL_COLORS = {0: "#4e79a7", 3: "#f28e2b", 5: "#e15759", 7: "#76b7b2"}


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
