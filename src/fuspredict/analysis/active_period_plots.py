"""
active_period_plots.py
----------------------
Plotting functions for active-period analysis.

Every function takes arrays and an output path — no data loading,
no config, no model inference. Uses Agg backend (non-interactive).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family":    "serif",
        "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":      9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi":     300,
        "savefig.dpi":    300,
        "savefig.bbox":   "tight",
    }
)

_SIGMA_COLORS = ["#ffdddd", "#ffaaaa", "#ff7777"]


def fig_activation_delta(
    delta_map: np.ndarray,
    roi_mask: np.ndarray,
    session_id: str,
    out_path: str | Path,
) -> None:
    """Save a two-panel figure: delta map alone and with ROI overlay.

    Parameters
    ----------
    delta_map : np.ndarray, shape (H, W)
        log10 activation delta (post-onset mean - baseline mean).
    roi_mask : np.ndarray, shape (H, W), bool
    session_id : str
    out_path : str or Path
        Output file path (PNG or PDF).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmax = max(float(np.percentile(np.abs(delta_map), 98)), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im = axes[0].imshow(delta_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="Δlog10 power")
    axes[0].set_title("Activation delta\n(post-onset mean − baseline mean)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(delta_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    axes[1].contour(roi_mask.astype(float), levels=[0.5], colors=["yellow"], linewidths=[1.5])
    roi_patch = mpatches.Patch(edgecolor="yellow", facecolor="none", linewidth=1.5, label="ROI")
    axes[1].legend(handles=[roi_patch], loc="lower right", fontsize=8)
    axes[1].set_title(f"ROI overlay  ({roi_mask.sum()} px)", fontsize=9)
    axes[1].axis("off")

    fig.suptitle(f"Session {session_id} — activation delta", fontsize=10)
    fig.savefig(out_path)
    plt.close(fig)


def fig_roi_timeseries(
    signal_pct: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    fps: float,
    session_id: str,
    out_path: str | Path,
    window_s: float = 30.0,
) -> None:
    """Save a time-series figure of ROI-averaged % CBV around active-period onset.

    Parameters
    ----------
    signal_pct : np.ndarray, shape (T,)
        ROI-averaged % CBV for the full task period (frame 0 = onset).
    baseline_mean : float
        Baseline mean in % CBV (≈ 0).
    baseline_std : float
        Baseline std in % CBV.
    fps : float
    session_id : str
    out_path : str or Path
    window_s : float
        Seconds of the task period to display (from onset).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T      = len(signal_pct)
    n_show = min(T, int(round(window_s * fps)))
    sig    = signal_pct[:n_show]
    t_rel  = np.arange(n_show) / fps

    fig, ax = plt.subplots(figsize=(8, 3.5))

    band_cap   = baseline_mean + 3.5 * baseline_std
    band_edges = [
        baseline_mean,
        baseline_mean + baseline_std,
        baseline_mean + 2 * baseline_std,
        band_cap,
    ]
    for i, col in enumerate(_SIGMA_COLORS):
        ax.axhspan(band_edges[i], band_edges[i + 1], color=col, lw=0, zorder=1)

    ax.plot(t_rel, sig, color="#2ca02c", lw=2.0, zorder=5)
    ax.axhline(baseline_mean, color="#888888", lw=1.0, ls=":", zorder=2, label="baseline mean")
    ax.axhline(0, color="#cccccc", lw=0.7, zorder=1)
    ax.axvline(0, color="black", lw=1.2, ls="--", zorder=6, label="onset (t₀)")

    x_right = t_rel[-1]
    for n, edge in zip((1, 2, 3), band_edges[1:]):
        y_label = (band_edges[n - 1] + edge) / 2
        ax.text(x_right, y_label, f" +{n}σ", va="center", ha="left",
                fontsize=8, color="#cc3333", clip_on=False)

    y_bot = float(sig.min()) - 0.5 * baseline_std
    y_top = max(float(sig.max()), band_cap) + 0.5 * baseline_std
    ax.set_xlim(t_rel[0], t_rel[-1])
    ax.set_ylim(y_bot, y_top)
    ax.set_xlabel("Time from onset (s)")
    ax.set_ylabel("ROI-mean % CBV")
    ax.set_title(
        f"{session_id} — ROI signal from active-period onset\n"
        f"baseline: mean={baseline_mean:.2f}%, σ={baseline_std:.2f}%  |  ROI size read from mask",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    fig.savefig(out_path)
    plt.close(fig)
