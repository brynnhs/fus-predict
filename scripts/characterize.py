"""
scripts/characterize.py
------------------------
Signal characterisation — per-session and cross-session figures.

Loads standardized baseline sessions (as Session objects) and writes cached
numpy / CSV outputs and publication-ready figures to:

    derivatives/modeling/signal_characterization/secundo_29sessions/
                                                 <session_id>/

Figures produced per session:
  fig1_representative_frame   — single z-score frame + scale bar
  fig2_mean_trace_and_acf     — mean trace (top) + ACF stem (bottom)
  fig3_per_pixel_acf          — lag-1 histogram + spatial map
  fig3b_r2_ceiling            — R² ceiling histogram + spatial map
  fig4_variance_ratio         — Var(Δx)/Var(x) histogram
  fig5_patch_acf_sweep        — median ACF ± IQR per patch size
  fig6_rolling_mean_std       — rolling mean and rolling std (stationarity)
  fig7_within_patch_residual_corr — within-patch residual structure

Cached arrays / tables (per session sub-directory):
  corr_map.npy, mask.npy, varx_map.npy, vard_map.npy, ratio_map.npy
  gmean.npy, gstd.npy, lg_mean.npy, av_mean.npy
  lag1_summary.csv, variance_summary.csv, rolling_stats.csv
  patch_acf_<size>.npy  (one per patch size in acf_patch_sizes)

Usage:
  python scripts/characterize.py
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuspredict.autocorrelation import (
    robust_limits,
    safe_standardized_acf,
    safe_temporal_corr_map,
)
from fuspredict.data.loading import load_sessions
from fuspredict.data.session import Session
from fuspredict.models.pca_ar import PatchLagPCAAR
from fuspredict.plot_utils import savefig
from fuspredict.project import find_repo_root, load_project_config

matplotlib.use('Agg')


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Palatino Linotype', 'Palatino', 'Georgia', 'DejaVu Serif'],
    'font.style':         'normal',
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'axes.grid':          False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.major.size':   0,
    'ytick.major.size':   3,
    'xtick.minor.size':   0,
    'ytick.minor.size':   0,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'legend.frameon':     False,
})

_NAVY       = '#000000'
_STEEL_BLUE = '#4878A8'
_TERRACOTTA = '#A0522D'
_CI_RED     = '#C0392B'
_SINGLE_COL = 3.5   # inches
_DOUBLE_COL = 7.0   # inches

CONDITION = 'unfiltered'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_scale_bar(ax, pixel_size_mm: float = 0.1, bar_mm: float = 1.0,
                   color: str = 'white') -> None:
    bar_px = bar_mm / pixel_size_mm
    xl, yl = ax.get_xlim(), ax.get_ylim()
    x0 = xl[0] + 0.07 * (xl[1] - xl[0])
    y0 = (yl[0] + 0.07 * (yl[1] - yl[0])
          if yl[0] < yl[1]
          else yl[0] - 0.07 * (yl[0] - yl[1]))
    ax.plot([x0, x0 + bar_px], [y0, y0], lw=2, color=color, solid_capstyle='butt')
    ax.text(x0 + bar_px / 2, y0, f'{bar_mm:g} mm',
            color=color, ha='center', va='bottom', fontsize=7)


def _patch_mean_acf(frames: np.ndarray, patch_radius: int, max_lag: int) -> np.ndarray:
    """Return (P, max_lag) ACF matrix for each patch's mean time series."""
    T, H, W = frames.shape
    patches = PatchLagPCAAR.tile_patches(H, W, patch_radius)
    acfs = []
    for rs, cs in patches:
        ts = frames[:, rs, cs].mean(axis=(1, 2))
        ts = ts - ts.mean()
        c0 = np.dot(ts, ts)
        row = [
            float(np.dot(ts[lag:], ts[:-lag]) / c0) if c0 > 0 else 0.0
            for lag in range(1, max_lag + 1)
        ]
        acfs.append(row)
    return np.array(acfs)


# ---------------------------------------------------------------------------
# Per-session analysis mask
# ---------------------------------------------------------------------------

def _compute_analysis_mask(session: Session, min_var: float) -> np.ndarray:
    """
    Compute the analysis-specific pixel mask for a session.

    This is distinct from ``Session.vessel_mask``: it selects pixels with
    finite, non-degenerate signal (sufficient temporal variance) for use in
    signal-characterization statistics, not anatomical vessel/parenchyma
    classification.
    """
    fr  = session.frames
    var = np.nanvar(fr, axis=0)
    return np.all(np.isfinite(fr), axis=0) & np.isfinite(var) & (var > min_var)


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def _fig1_representative_frame(fr, mask, example_frame_idx, sout):
    """Fig 1: single representative z-score frame with scale bar."""
    cur     = np.where(mask, fr[example_frame_idx], np.nan)
    v0, v1  = robust_limits(cur, pctl=(2, 98))

    np.save(sout / 'example_frame.npy', cur)

    fig, ax = plt.subplots(figsize=(_SINGLE_COL, _SINGLE_COL), constrained_layout=True)
    im = ax.imshow(cur, cmap='gray', vmin=v0, vmax=v1, aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
    cb.set_label('Signal (z-score)')
    cb.ax.tick_params(labelsize=7)
    _add_scale_bar(ax)
    savefig(fig, sout / 'fig1_representative_frame')
    plt.close(fig)


def _fig2_mean_trace_and_acf(gmean, lg_mean, av_mean, frame_rate_hz, sout):
    """Fig 2: mean trace (top panel) + ACF stem (bottom panel)."""
    if not np.any(np.isfinite(gmean)):
        return

    time   = np.arange(gmean.size, dtype=float) / frame_rate_hz
    lags_s = np.asarray(lg_mean, float) / frame_rate_hz if lg_mean is not None else None
    ci     = 1.96 / np.sqrt(max(gmean.size, 1))

    fig, axes = plt.subplots(2, 1,
                             figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.55),
                             constrained_layout=True)
    axes[0].plot(time, gmean, lw=0.8, color=_NAVY)
    axes[0].axhline(0, color='0.55', ls='-', lw=0.8)
    gmin, gmax = float(np.nanmin(gmean)), float(np.nanmax(gmean))
    pad = max(0.3 * (gmax - gmin), 0.5)
    axes[0].set_ylim(gmin - pad, gmax + pad)
    axes[0].set_ylabel('Signal (z-score)')
    axes[0].set_xticklabels([])

    if lags_s is not None:
        ml, sl, bl = axes[1].stem(lags_s, av_mean,
                                   linefmt=_NAVY, markerfmt=' ', basefmt='none')
        plt.setp(sl, linewidth=1.1)
        axes[1].axhline(0,   color='0.5',   lw=0.6)
        axes[1].axhline(ci,  color=_CI_RED, ls='--', lw=0.8)
        axes[1].axhline(-ci, color=_CI_RED, ls='--', lw=0.8)
        axes[1].annotate('95% CI',
                         xy=(lags_s[-1], ci),
                         xytext=(1.02, ci),
                         textcoords=('axes fraction', 'data'),
                         ha='left', va='center', fontsize=7, color=_CI_RED)
        axes[1].set_ylim(-0.2, 1.0)
        axes[1].set_ylabel('Autocorrelation')
    else:
        axes[1].text(0.5, 0.5, 'ACF not available',
                     ha='center', va='center', transform=axes[1].transAxes)

    axes[1].set_xlabel('Time / Lag (s)')
    savefig(fig, sout / 'fig2_mean_trace_and_acf')
    plt.close(fig)


def _fig3_per_pixel_acf(corr, mask, lg_mean, av_mean, frame_rate_hz, sout):
    """Fig 3: per-pixel lag-1 ACF histogram (left) + spatial map (right)."""
    lagvals = corr[mask]
    lagvals = lagvals[np.isfinite(lagvals)]
    if lagvals.size == 0:
        return

    med        = float(np.median(lagvals))
    mean_lag1  = float(av_mean[1]) if (av_mean is not None and len(av_mean) > 1) else np.nan

    pd.DataFrame([{
        'count':           int(lagvals.size),
        'mean':            float(np.mean(lagvals)),
        'median':          med,
        'std':             float(np.std(lagvals)),
        'fraction_gt_0p5': float(np.mean(lagvals > .5)),
        'mean_trace_lag1': mean_lag1,
    }]).to_csv(sout / 'lag1_summary.csv', index=False)

    corr_vmin = max(-0.15, float(np.nanpercentile(lagvals, 2)))
    corr_vmax = min(0.5,   float(np.nanpercentile(lagvals, 99)))

    fig, axes = plt.subplots(1, 2,
                             figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.45),
                             constrained_layout=True)
    axes[0].hist(lagvals, bins=80, color=_STEEL_BLUE,
                 edgecolor='white', linewidth=0.6, density=True)
    axes[0].axvline(med, color=_NAVY,    ls='--', lw=1)
    axes[0].axvline(0.5, color=_CI_RED,  ls='--', lw=1)
    trans = axes[0].get_xaxis_transform()
    axes[0].text(med, 0.97, 'median',    transform=trans, ha='center', va='top', fontsize=7, color=_NAVY)
    axes[0].text(0.5, 0.97, 'threshold', transform=trans, ha='center', va='top', fontsize=7, color=_CI_RED)
    if np.isfinite(mean_lag1):
        axes[0].axvline(mean_lag1, color=_TERRACOTTA, ls=':', lw=1)
        axes[0].text(mean_lag1, 0.85, 'mean\ntrace', transform=trans,
                     ha='center', va='top', fontsize=7, color=_TERRACOTTA)
    axes[0].set_xlabel('Lag-1 autocorrelation')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim(-0.3, 0.8)

    im = axes[1].imshow(corr, cmap='seismic',
                        vmin=corr_vmin, vmax=corr_vmax, aspect='equal')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    cb = plt.colorbar(im, ax=axes[1], fraction=.046, pad=.04)
    cb.set_label('Lag-1 autocorrelation')
    cb.ax.tick_params(labelsize=7)
    _add_scale_bar(axes[1])

    savefig(fig, sout / 'fig3_per_pixel_acf')
    plt.close(fig)


def _fig3b_r2_ceiling(corr, mask, sout):
    """Fig 3b: R² ceiling map and histogram.

    R²(i,j) = r(i,j)² where r is the lag-1 Pearson autocorrelation.
    This is the theoretical upper bound on variance explained by a
    one-step-ahead linear predictor at each pixel.
    """
    r2 = np.where(np.isfinite(corr), corr ** 2, np.nan)
    np.save(sout / 'r2_ceiling_map.npy', r2.astype(np.float32))

    r2vals = r2[mask]
    r2vals = r2vals[np.isfinite(r2vals)]
    if r2vals.size == 0:
        return

    pd.DataFrame([{
        'median_r2':        float(np.median(r2vals)),
        'mean_r2':          float(np.mean(r2vals)),
        'std_r2':           float(np.std(r2vals)),
        'fraction_gt_0p25': float(np.mean(r2vals > 0.25)),
        'fraction_gt_0p50': float(np.mean(r2vals > 0.50)),
    }]).to_csv(sout / 'r2_ceiling_summary.csv', index=False)

    med_r2  = float(np.median(r2vals))
    r2_vmax = min(1.0, float(np.nanpercentile(r2vals, 99)) * 1.1)

    fig, axes = plt.subplots(1, 2,
                             figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.45),
                             constrained_layout=True)
    axes[0].hist(r2vals, bins=80, color=_STEEL_BLUE,
                 edgecolor='white', linewidth=0.6, density=True)
    axes[0].axvline(med_r2, color=_NAVY, ls='--', lw=1)
    trans = axes[0].get_xaxis_transform()
    axes[0].text(med_r2, 0.97, f'median {med_r2:.3f}', transform=trans,
                 ha='center', va='top', fontsize=7, color=_NAVY)
    axes[0].set_xlabel('R² ceiling  (lag-1 autocorrelation²)')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim(0, max(r2_vmax, 0.05))

    display = np.where(mask, r2, np.nan)
    im = axes[1].imshow(display, cmap='hot', vmin=0, vmax=r2_vmax, aspect='equal')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    cb = plt.colorbar(im, ax=axes[1], fraction=.046, pad=.04)
    cb.set_label('R² ceiling')
    cb.ax.tick_params(labelsize=7)
    _add_scale_bar(axes[1])

    savefig(fig, sout / 'fig3b_r2_ceiling')
    plt.close(fig)


def _fig4_variance_ratio(ratio, mask, sout):
    """Fig 4: histogram of Var(Δx) / Var(x)."""
    vals = ratio[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    med  = float(np.median(vals))
    xmax = min(float(np.percentile(vals, 99)) * 1.1, 5.0)

    pd.DataFrame([{
        'count':              int(vals.size),
        'median':             med,
        'mean':               float(np.mean(vals)),
        'fraction_above_1':   float(np.mean(vals > 1.0)),
    }]).to_csv(sout / 'variance_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(_SINGLE_COL * 1.3, _SINGLE_COL),
                           constrained_layout=True)
    ax.hist(vals[vals <= xmax], bins=80, color=_STEEL_BLUE,
            edgecolor='white', linewidth=0.6, density=True)
    ax.axvline(1.0, color=_NAVY,   ls='--', lw=1)
    ax.axvline(med, color=_CI_RED, ls='--', lw=1)
    ax.axvspan(1.0, xmax, color=_TERRACOTTA, alpha=0.08, zorder=0)
    trans = ax.get_xaxis_transform()
    ax.text(1.0, 0.97, 'ratio = 1',         transform=trans, ha='center', va='top', fontsize=7, color=_NAVY)
    ax.text(med, 0.97, f'median {med:.2f}', transform=trans, ha='center', va='top', fontsize=7, color=_CI_RED)
    noise_x = min(1.0 + (xmax - 1.0) * 0.5, xmax * 0.85)
    ax.text(noise_x, 0.88, 'noise-dominated', transform=trans,
            ha='center', va='top', fontsize=7, color='0.45')
    ax.set_xlabel('Variance ratio  Var(Δx) / Var(x)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, xmax)
    savefig(fig, sout / 'fig4_variance_ratio')
    plt.close(fig)


def _fig5_patch_acf_sweep(fr, mask, patch_sizes, frame_rate_hz, sout, max_lag: int = 15) -> None:
    """Fig 5: strip of panels — median ACF ± IQR for each patch size."""
    frames   = fr.copy().astype(np.float64)
    frames[:, ~mask] = 0.0

    n_cols   = len(patch_sizes)
    lag_axis = np.arange(1, max_lag + 1, dtype=float) / frame_rate_hz

    fig, axes = plt.subplots(1, n_cols,
                             figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.4),
                             sharey=True, constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for ax, ps in zip(axes, patch_sizes):
        pr      = max(1, ps // 2)
        acf_mat = _patch_mean_acf(frames, pr, max_lag)
        med     = np.median(acf_mat, axis=0)
        p25     = np.percentile(acf_mat, 25, axis=0)
        p75     = np.percentile(acf_mat, 75, axis=0)

        np.save(sout / f'patch_acf_{ps}.npy', acf_mat)

        ax.fill_between(lag_axis, p25, p75, alpha=0.25, color=_STEEL_BLUE)
        ax.plot(lag_axis, med, color=_STEEL_BLUE, lw=1.8)
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        ax.set_title(f'{ps}×{ps} px\nACF(1)={med[0]:.2f}', fontsize=8)
        ax.set_xlabel('Lag (s)')
        ax.set_xticks(lag_axis[::3])

    axes[0].set_ylabel('Autocorrelation')
    fig.suptitle('Patch-mean ACF vs patch size  (shading = IQR)', fontsize=9)
    savefig(fig, sout / 'fig5_patch_acf_sweep')
    plt.close(fig)


def _fig6_rolling_mean_std(gmean, frame_rate_hz, sout, window: int = 20) -> None:
    """Fig 6: rolling mean and rolling std of the analysis-mask mean."""
    if not np.any(np.isfinite(gmean)):
        return

    rolling = pd.DataFrame({'gmean': gmean})
    rolling[f'rolling_mean_w{window}'] = pd.Series(gmean).rolling(window).mean()
    rolling[f'rolling_std_w{window}']  = pd.Series(gmean).rolling(window).std()
    rolling.to_csv(sout / 'rolling_stats.csv', index=False)

    time = np.arange(gmean.size, dtype=float) / frame_rate_hz
    gmin, gmax = float(np.nanmin(gmean)), float(np.nanmax(gmean))
    pad  = max(0.3 * (gmax - gmin), 0.5)

    fig, axes = plt.subplots(2, 1,
                             figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.45),
                             sharex=True, constrained_layout=True)
    axes[0].plot(time, gmean, color='0.80', lw=0.6)
    axes[0].plot(time, rolling[f'rolling_mean_w{window}'],
                 color=_NAVY, lw=1.2, label=f'Rolling mean (w={window})')
    axes[0].axhline(0, color='0.55', lw=0.8)
    axes[0].set_ylim(gmin - pad, gmax + pad)
    axes[0].set_ylabel('Signal (z-score)')
    axes[0].legend(loc='upper right')

    axes[1].plot(time, rolling[f'rolling_std_w{window}'],
                 color=_TERRACOTTA, lw=1.2, label=f'Rolling std (w={window})')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Rolling std (z-score)')
    axes[1].legend(loc='upper right')

    savefig(fig, sout / 'fig6_rolling_mean_std')
    plt.close(fig)


def _fig7_within_patch_residual_corr(fr, mask, patch_size: int, sout) -> None:
    """Fig 7: within-patch residual structure diagnostics.

    Panel A — histogram of mean pairwise within-patch residual correlations.
    Panel B — histogram of within-patch residual std / patch-mean std.
    """
    frames = fr.astype(np.float64)
    frames[:, ~mask] = np.nan

    patch_radius = max(1, patch_size // 2)
    patches      = PatchLagPCAAR.tile_patches(fr.shape[1], fr.shape[2], patch_radius)
    T            = frames.shape[0]

    mean_pair_corrs:     list[float] = []
    noise_signal_ratios: list[float] = []

    for rs, cs in patches:
        block = frames[:, rs, cs]
        px    = block.reshape(T, -1)

        finite_frac = np.isfinite(px).mean(axis=0)
        valid = finite_frac >= 0.9
        if valid.sum() < 4:
            continue
        px = px[:, valid]

        patch_mean = np.nanmean(px, axis=1, keepdims=True)
        residuals  = px - patch_mean
        residuals  = np.where(np.isfinite(residuals), residuals, 0.0)

        res_std = residuals.std(axis=0)
        has_var = res_std > 1e-10
        if has_var.sum() < 4:
            continue
        res_z = residuals[:, has_var] / res_std[has_var]
        S_v2  = res_z.shape[1]

        R       = (res_z.T @ res_z) / T
        n_pairs = S_v2 * (S_v2 - 1)
        mean_r  = float((R.sum() - np.trace(R)) / n_pairs)
        mean_pair_corrs.append(mean_r + 1.0 / (S_v2 - 1))

        pm_std = float(patch_mean.squeeze().std())
        if pm_std > 0:
            noise_signal_ratios.append(float(residuals.std()) / pm_std)

    if not mean_pair_corrs:
        print('  Fig 7: no valid patches, skipping.')
        return

    corr_arr  = np.array(mean_pair_corrs,     dtype=np.float32)
    ratio_arr = np.array(noise_signal_ratios,  dtype=np.float32)

    np.save(sout / 'within_patch_mean_pairwise_corr.npy', corr_arr)
    np.save(sout / 'within_patch_noise_signal_ratio.npy', ratio_arr)

    mean_corr    = float(np.mean(corr_arr))
    median_ratio = float(np.median(ratio_arr)) if len(ratio_arr) else np.nan

    fig, axes = plt.subplots(1, 2,
                             figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.45),
                             constrained_layout=True)

    half = float(np.abs(corr_arr).max()) * 1.3
    axes[0].hist(corr_arr, bins=30, range=(-half, half), color=_STEEL_BLUE,
                 edgecolor='white', linewidth=0.5, density=True)
    axes[0].axvline(mean_corr, color=_NAVY, ls='--', lw=1.2)
    trans = axes[0].get_xaxis_transform()
    axes[0].text(mean_corr, 0.97, f'mean = {mean_corr:.2e}',
                 transform=trans, ha='left', va='top', fontsize=7, color=_NAVY)
    axes[0].set_xlabel('Mean pairwise Pearson r (within-patch residuals)')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim(-half, half)

    if len(ratio_arr):
        xmax = float(np.percentile(ratio_arr, 99)) * 1.1
        axes[1].hist(ratio_arr[ratio_arr <= xmax], bins=60, color=_STEEL_BLUE,
                     edgecolor='white', linewidth=0.5, density=True)
        axes[1].axvspan(1.0, xmax, color=_TERRACOTTA, alpha=0.10, zorder=0)
        axes[1].axvline(1.0,         color=_NAVY,   ls='--', lw=1.0)
        axes[1].axvline(median_ratio, color=_CI_RED, ls='--', lw=1.2)
        trans2 = axes[1].get_xaxis_transform()
        axes[1].text(median_ratio, 0.97, f'median = {median_ratio:.1f}×',
                     transform=trans2, ha='left', va='top', fontsize=7, color=_CI_RED)
        axes[1].text(1.0, 0.97, 'ratio = 1',
                     transform=trans2, ha='right', va='top', fontsize=7, color=_NAVY)
        axes[1].set_xlabel('Within-patch residual std / patch-mean std')
        axes[1].set_ylabel('Density')
        axes[1].set_xlim(0, xmax)
    else:
        axes[1].text(0.5, 0.5, 'No data', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=9)

    savefig(fig, sout / 'fig7_within_patch_structure')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-session dispatcher
# ---------------------------------------------------------------------------

def run_session(session: Session, mask: np.ndarray, sout: Path, cfg: dict) -> None:
    """
    Compute and save per-session statistics and figures.

    Parameters
    ----------
    session : Session
        Loaded session (frames, fps, id).
    mask : np.ndarray
        Analysis mask for this session, shape ``(H, W)``, from
        :func:`_compute_analysis_mask`.
    sout : Path
        Output directory for this session's cached arrays/tables/figures.
    cfg : dict
        ``persistence_analysis`` config (merged with AR-analysis defaults).
    """
    sout.mkdir(parents=True, exist_ok=True)

    fr  = session.frames
    fps = session.fps

    if mask.sum() == 0 or fr.shape[0] < 2:
        print(f'  {session.id} — skipped (empty mask or too few frames)')
        return

    MIN_VAR        = cfg['min_var']
    MAX_LAG        = cfg['max_acf_lag']
    EX_FRAME       = int(np.clip(cfg['example_target_frame'], 1, fr.shape[0] - 1))
    PATCH_SZ       = cfg.get('acf_patch_sizes', [5, 10, 15, 20, 25, 30])
    MODEL_PATCH_SZ = int(cfg.get('patch_lag_pca_ar_patch_size', 15))
    ROLL_WIN       = cfg.get('pearson_rolling_window', 20)

    gmean = np.nanmean(fr[:, mask], axis=1)
    gstd  = np.nanstd(fr[:, mask],  axis=1)

    varx  = np.nanvar(fr, axis=0).astype(np.float32)
    vard  = np.nanvar(np.diff(fr, axis=0), axis=0).astype(np.float32)
    ratio = np.divide(vard, varx,
                      out=np.full_like(vard, np.nan),
                      where=varx > MIN_VAR)
    corr, _ = safe_temporal_corr_map(fr[:-1], fr[1:])
    corr    = np.asarray(corr, np.float32)
    for a in [varx, vard, ratio, corr]:
        a[~mask] = np.nan

    lg_mean, av_mean = safe_standardized_acf(gmean, MAX_LAG)

    np.save(sout / 'corr_map.npy',  corr)
    np.save(sout / 'mask.npy',      mask.astype(np.uint8))
    np.save(sout / 'varx_map.npy',  varx)
    np.save(sout / 'vard_map.npy',  vard)
    np.save(sout / 'ratio_map.npy', ratio)
    np.save(sout / 'gmean.npy',     gmean)
    np.save(sout / 'gstd.npy',      gstd)
    if lg_mean is not None:
        np.save(sout / 'lg_mean.npy', np.asarray(lg_mean))
        np.save(sout / 'av_mean.npy', np.asarray(av_mean))

    # RMSE floor: irreducible residual under a perfect lag-1 linear predictor.
    # At each pixel: residual_var(i,j) = Var(x) * (1 - r²)
    # RMSE_floor = sqrt( mean_{pixels}[ varx * (1 - corr²) ] )
    finite_mask = mask & np.isfinite(varx) & np.isfinite(corr)
    varx_v      = varx[finite_mask]
    corr_v      = corr[finite_mask]
    resid_var   = varx_v * (1.0 - corr_v ** 2)
    rmse_floor  = float(np.sqrt(np.mean(resid_var))) if resid_var.size > 0 else np.nan
    mean_varx   = float(np.mean(varx_v))              if varx_v.size > 0  else np.nan
    r2_ceiling  = float(np.mean(corr_v ** 2))         if corr_v.size > 0  else np.nan

    pd.DataFrame([{
        'session_id': session.id,
        'rmse_floor': rmse_floor,
        'mean_varx':  mean_varx,
        'r2_ceiling': r2_ceiling,
        'n_pixels':   int(resid_var.size),
    }]).to_csv(sout / 'rmse_floor.csv', index=False)

    _fig1_representative_frame(fr, mask, EX_FRAME, sout)
    _fig2_mean_trace_and_acf(gmean, lg_mean, av_mean, fps, sout)
    _fig3_per_pixel_acf(corr, mask, lg_mean, av_mean, fps, sout)
    _fig3b_r2_ceiling(corr, mask, sout)
    _fig4_variance_ratio(ratio, mask, sout)
    _fig5_patch_acf_sweep(fr, mask, PATCH_SZ, fps, sout)
    _fig6_rolling_mean_std(gmean, fps, sout, window=ROLL_WIN)
    _fig7_within_patch_residual_corr(fr, mask, MODEL_PATCH_SZ, sout)

    print(f'  {session.id} → {sout}')


# ---------------------------------------------------------------------------
# Cross-session figures
# ---------------------------------------------------------------------------

def _figX_r2_ceiling_across_sessions(session_dirs: list[Path], out: Path) -> None:
    """Fig X1: violin of per-session median R² ceiling values."""
    records = []
    for sd in session_dirs:
        r2_path   = sd / 'r2_ceiling_map.npy'
        mask_path = sd / 'mask.npy'
        if not (r2_path.exists() and mask_path.exists()):
            continue
        r2   = np.load(r2_path)
        mask = np.load(mask_path).astype(bool)
        vals = r2[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        records.append({'session_id': sd.name, 'median_r2': float(np.median(vals)),
                        'mean_r2': float(np.mean(vals)), 'vals': vals})

    if not records:
        print('  FigX1: no sessions with cached R² maps, skipping.')
        return

    pd.DataFrame([{k: v for k, v in r.items() if k != 'vals'} for r in records]
                 ).to_csv(out / 'cross_session_r2_summary.csv', index=False)

    labels  = [r['session_id'] for r in records]
    data    = [r['vals']       for r in records]
    medians = [r['median_r2']  for r in records]
    order   = np.argsort(medians)
    labels  = [labels[i] for i in order]
    data    = [data[i]   for i in order]

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, max(3.0, len(records) * 0.35)),
                           constrained_layout=True)
    parts = ax.violinplot(data, orientation='horizontal', showmedians=True, positions=range(len(data)))
    for pc in parts['bodies']:
        pc.set_facecolor(_STEEL_BLUE)
        pc.set_alpha(0.6)
    parts['cmedians'].set_color(_NAVY)
    parts['cmedians'].set_linewidth(1.5)
    for key in ('cbars', 'cmins', 'cmaxes'):
        parts[key].set_color(_STEEL_BLUE)
        parts[key].set_linewidth(0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('R² ceiling  (lag-1 autocorrelation²)')
    ax.set_title('Per-session R² ceiling distribution', fontsize=9)
    ax.axvline(0, color='0.6', lw=0.7, ls='--')
    savefig(fig, out / 'figX1_r2_ceiling_across_sessions')
    plt.close(fig)


def _figX_r2_spatial_consistency(session_dirs: list[Path], out: Path) -> None:
    """Fig X2: mean R² map + per-session spatial correlation with the mean."""
    maps, masks, sids = [], [], []
    for sd in session_dirs:
        r2_path   = sd / 'r2_ceiling_map.npy'
        mask_path = sd / 'mask.npy'
        if not (r2_path.exists() and mask_path.exists()):
            continue
        maps.append(np.load(r2_path).astype(np.float32))
        masks.append(np.load(mask_path).astype(bool))
        sids.append(sd.name)

    if len(maps) < 2:
        print('  FigX2: need ≥2 sessions with cached R² maps, skipping.')
        return

    stack    = np.stack([np.where(m, r, np.nan) for r, m in zip(maps, masks)], axis=0)
    mean_map = np.nanmean(stack, axis=0)
    np.save(out / 'r2_ceiling_mean_map.npy', mean_map.astype(np.float32))

    shared_mask = np.all(np.isfinite(stack), axis=0)
    corrs = []
    for i in range(len(sids)):
        x = stack[i][shared_mask]
        y = mean_map[shared_mask]
        corrs.append(float(np.corrcoef(x, y)[0, 1]) if (x.std() > 0 and y.std() > 0) else np.nan)

    vmax = min(1.0, float(np.nanpercentile(mean_map[shared_mask], 99)) * 1.1)
    fig, axes = plt.subplots(1, 2, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.45),
                             constrained_layout=True)

    im = axes[0].imshow(mean_map, cmap='hot', vmin=0, vmax=vmax, aspect='equal')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    cb = plt.colorbar(im, ax=axes[0], fraction=.046, pad=.04)
    cb.set_label('Mean R² ceiling')
    cb.ax.tick_params(labelsize=7)
    axes[0].set_title(f'Mean R² map  (n={len(maps)} sessions)', fontsize=8)
    _add_scale_bar(axes[0])

    finite_corrs = [(s, c) for s, c in zip(sids, corrs) if np.isfinite(c)]
    if finite_corrs:
        sorted_pairs = sorted(finite_corrs, key=lambda x: x[1])
        fc_sids = [p[0] for p in sorted_pairs]
        fc_vals = [p[1] for p in sorted_pairs]
        axes[1].barh(range(len(fc_vals)), fc_vals, color=_STEEL_BLUE, height=0.7)
        axes[1].set_yticks(range(len(fc_sids)))
        axes[1].set_yticklabels(fc_sids, fontsize=7)
        axes[1].set_xlabel('Pearson r  (session vs. mean R² map)')
        axes[1].axvline(0, color='0.5', lw=0.7)
        axes[1].set_xlim(-0.1, 1.05)
        axes[1].set_title('Spatial consistency', fontsize=8)

    savefig(fig, out / 'figX2_r2_spatial_consistency')
    plt.close(fig)


def _figX_acf_mean_map(session_dirs: list[Path], out: Path) -> None:
    """Fig X4: cross-session mean and std of the per-pixel lag-1 ACF map."""
    maps, sids = [], []
    for sd in session_dirs:
        corr_path = sd / 'corr_map.npy'
        mask_path = sd / 'mask.npy'
        if not (corr_path.exists() and mask_path.exists()):
            continue
        corr = np.load(corr_path).astype(np.float32)
        mask = np.load(mask_path).astype(bool)
        corr[~mask] = np.nan
        maps.append(corr)
        sids.append(sd.name)

    if len(maps) < 2:
        print('  FigX4: need ≥2 sessions with cached ACF maps, skipping.')
        return

    stack    = np.stack(maps, axis=0)
    mean_map = np.nanmean(stack, axis=0)
    std_map  = np.nanstd(stack,  axis=0)
    cv_map   = np.clip(np.where(np.abs(mean_map) > 1e-6, std_map / np.abs(mean_map), np.nan), 0, 3)

    np.save(out / 'acf_mean_map.npy', mean_map.astype(np.float32))
    np.save(out / 'acf_std_map.npy',  std_map.astype(np.float32))

    shared_mask = np.sum(np.isfinite(stack), axis=0) >= max(2, len(maps) // 2)
    pd.DataFrame([{
        'n_sessions':       len(maps),
        'grand_mean_acf':   float(np.nanmean(mean_map[shared_mask])),
        'grand_median_acf': float(np.nanmedian(mean_map[shared_mask])),
        'mean_std_acf':     float(np.nanmean(std_map[shared_mask])),
        'mean_cv_acf':      float(np.nanmean(cv_map[shared_mask])),
    }]).to_csv(out / 'acf_mean_map_summary.csv', index=False)

    v_mean = float(np.nanpercentile(mean_map[shared_mask], 99))
    v_std  = float(np.nanpercentile(std_map[shared_mask],  99))

    fig, axes = plt.subplots(1, 3,
                             figsize=(_DOUBLE_COL * 1.05, _DOUBLE_COL * 0.38),
                             constrained_layout=True)
    for ax, data, title, cmap, vmin, vmax, label in [
        (axes[0], mean_map, f'Mean ACF  (n={len(maps)})', 'seismic', -v_mean, v_mean, 'Mean lag-1 ACF'),
        (axes[1], std_map,  'Std ACF  (session variability)', 'hot', 0, v_std, 'Std lag-1 ACF'),
        (axes[2], cv_map,   'CV = std / |mean|  (clipped at 3)', 'viridis', 0, 3, 'CV'),
    ]:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
        cb.set_label(label)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=8)
        _add_scale_bar(ax)

    savefig(fig, out / 'figX4_acf_mean_map')
    plt.close(fig)


def _figX_variance_ratio_across_sessions(session_dirs: list[Path], out: Path) -> None:
    """Fig X5: cross-session variance ratio Var(Δx)/Var(x) distributions."""
    from scipy.stats import gaussian_kde

    records = []
    for sd in session_dirs:
        ratio_path = sd / 'ratio_map.npy'
        mask_path  = sd / 'mask.npy'
        if not (ratio_path.exists() and mask_path.exists()):
            continue
        ratio = np.load(ratio_path)
        mask  = np.load(mask_path).astype(bool)
        vals  = ratio[mask]
        vals  = vals[np.isfinite(vals)]
        if vals.size < 10:
            continue
        records.append({'session_id': sd.name, 'median': float(np.median(vals)), 'vals': vals})

    if not records:
        print('  FigX5: no sessions with cached ratio maps, skipping.')
        return

    pd.DataFrame([{'session_id': r['session_id'], 'median_var_ratio': r['median']}
                  for r in records]).to_csv(out / 'cross_session_var_ratio_summary.csv', index=False)

    medians      = [r['median'] for r in records]
    sids         = [r['session_id'] for r in records]
    grand_median = float(np.median(medians))
    all_vals     = np.concatenate([r['vals'] for r in records])
    x_max        = min(5.0, float(np.percentile(all_vals, 99)) * 1.1)
    x_grid       = np.linspace(0, x_max, 500)

    fig, axes = plt.subplots(1, 2, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.5),
                             constrained_layout=True)
    cmap = plt.colormaps['tab20'].resampled(len(records))

    for i, r in enumerate(records):
        try:
            kde = gaussian_kde(r['vals'], bw_method='scott')
            axes[0].plot(x_grid, kde(x_grid), lw=1.0, color=cmap(i), alpha=0.75, label=r['session_id'])
        except Exception:
            pass

    axes[0].axvline(1.0,         color=_NAVY,   ls='--', lw=0.8, label='ratio = 1')
    axes[0].axvline(grand_median, color=_CI_RED, ls='--', lw=0.8, label=f'grand median {grand_median:.2f}')
    axes[0].set_xlabel('Var(Δx) / Var(x)')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim(0, x_max)
    axes[0].set_title('Per-session distribution', fontsize=8)
    if len(records) <= 12:
        axes[0].legend(fontsize=6, ncol=2, loc='upper right')

    order = np.argsort(medians)
    axes[1].scatter([medians[i] for i in order], range(len(order)),
                    color=[cmap(i) for i in order], s=25, zorder=3)
    axes[1].axvline(1.0,          color=_NAVY,   ls='--', lw=1, label='ratio = 1')
    axes[1].axvline(grand_median, color=_CI_RED, ls='--', lw=1, label=f'grand median {grand_median:.2f}')
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels([sids[i] for i in order], fontsize=7)
    axes[1].set_xlabel('Median Var(Δx) / Var(x)')
    axes[1].legend(fontsize=7)
    axes[1].set_title('Per-session median', fontsize=8)

    savefig(fig, out / 'figX5_variance_ratio_across_sessions')
    plt.close(fig)


def _figX_acf_distribution_across_sessions(session_dirs: list[Path], out: Path) -> None:
    """Fig X3: lag-1 ACF histogram overlay across sessions."""
    from scipy.stats import gaussian_kde

    records = []
    for sd in session_dirs:
        corr_path = sd / 'corr_map.npy'
        mask_path = sd / 'mask.npy'
        if not (corr_path.exists() and mask_path.exists()):
            continue
        corr = np.load(corr_path)
        mask = np.load(mask_path).astype(bool)
        vals = corr[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size < 10:
            continue
        records.append({'session_id': sd.name, 'vals': vals, 'median': float(np.median(vals))})

    if not records:
        print('  FigX3: no sessions with cached corr maps, skipping.')
        return

    pd.DataFrame([{'session_id': r['session_id'], 'median_lag1_acf': r['median']}
                  for r in records]).to_csv(out / 'cross_session_acf_summary.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.5),
                             constrained_layout=True)
    cmap   = plt.colormaps['tab20'].resampled(len(records))
    x_grid = np.linspace(-0.5, 1.0, 400)

    for i, r in enumerate(records):
        try:
            kde = gaussian_kde(r['vals'], bw_method='scott')
            axes[0].plot(x_grid, kde(x_grid), lw=1.0, color=cmap(i), alpha=0.75, label=r['session_id'])
        except Exception:
            pass

    axes[0].axvline(0.5, color=_CI_RED, ls='--', lw=0.8)
    axes[0].set_xlabel('Lag-1 autocorrelation')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim(-0.4, 0.9)
    axes[0].set_title('ACF distribution per session', fontsize=8)
    if len(records) <= 12:
        axes[0].legend(fontsize=6, ncol=2, loc='upper right')

    medians = [r['median'] for r in records]
    sids    = [r['session_id'] for r in records]
    order   = np.argsort(medians)
    axes[1].scatter([medians[i] for i in order], range(len(order)),
                    color=[cmap(i) for i in order], s=25, zorder=3)
    axes[1].axvline(float(np.median(medians)), color=_NAVY, ls='--', lw=1,
                    label=f'grand median {np.median(medians):.3f}')
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels([sids[i] for i in order], fontsize=7)
    axes[1].set_xlabel('Median lag-1 ACF')
    axes[1].legend(fontsize=7)
    axes[1].set_title('Per-session median', fontsize=8)

    savefig(fig, out / 'figX3_acf_distribution_across_sessions')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run signal characterization.")
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Config filename inside config/ (default: config.yml).",
    )
    return parser.parse_args()


def main() -> None:
    args      = _parse_args()
    repo_root = find_repo_root()
    config    = load_project_config(repo_root, config_name=args.config)

    pa    = dict(config['persistence_analysis'])
    arcfg = config.get('ar_analysis', {})
    pa.setdefault('acf_patch_sizes',             arcfg.get('acf_patch_sizes', [5, 10, 15, 20, 25, 30]))
    pa.setdefault('patch_lag_pca_ar_patch_size', arcfg.get('patch_lag_pca_ar_patch_size', 15))

    subject = config['subjects']['all'][0]
    EXCLUDED_SESSIONS = set(
        arcfg.get(
            'within_session_exclude',
            config['subjects'].get('sessions_to_exclude', {}).get(subject, []),
        )
    )

    preproc_root     = repo_root / config['paths']['preprocessing']
    standardized_dir = preproc_root / subject / 'baseline_only_standardized'
    mask_dir         = preproc_root / subject / 'tissue_masks'

    sessions = load_sessions(standardized_dir, mask_dir=mask_dir, exclude_ids=list(EXCLUDED_SESSIONS))
    assert sessions, f"No sessions loaded from {standardized_dir}"
    print(f"Found {len(sessions)} sessions")

    MIN_VAR = pa['min_var']
    masks = {s.id: _compute_analysis_mask(s, MIN_VAR) for s in sessions}

    OUT = repo_root / 'derivatives' / 'modeling' / 'signal_characterization' / subject
    OUT.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        'session_id':    s.id,
        'shape':         str(tuple(s.frames.shape)),
        'frame_rate_hz': float(s.fps),
        'mask_pixels':   int(masks[s.id].sum()),
        'mask_frac':     float(masks[s.id].mean()),
    } for s in sessions]).to_csv(OUT / 'session_summary.csv', index=False)

    print(f"Output root: {OUT}")
    print(f"Running {len(sessions)} sessions...")

    for s in sessions:
        run_session(s, masks[s.id], OUT / s.id, pa)

    session_dirs = [OUT / s.id for s in sessions]
    print("Computing cross-session figures...")
    _figX_r2_ceiling_across_sessions(session_dirs, OUT)
    _figX_r2_spatial_consistency(session_dirs, OUT)
    _figX_acf_mean_map(session_dirs, OUT)
    _figX_variance_ratio_across_sessions(session_dirs, OUT)
    _figX_acf_distribution_across_sessions(session_dirs, OUT)

    floor_dfs = [pd.read_csv(sd / 'rmse_floor.csv')
                 for sd in session_dirs if (sd / 'rmse_floor.csv').exists()]
    if floor_dfs:
        floor_all = pd.concat(floor_dfs, ignore_index=True)
        floor_all.to_csv(OUT / 'rmse_floor_all_sessions.csv', index=False)
        summary_rows = []
        for col, label in [('rmse_floor', 'rmse_floor'), ('r2_ceiling', 'r2_ceiling')]:
            if col not in floor_all:
                continue
            vals = floor_all[col].dropna().values
            summary_rows.append({
                'metric':       label,
                'n_sessions':   int(len(vals)),
                'grand_median': float(np.median(vals)),
                'q25':          float(np.percentile(vals, 25)),
                'q75':          float(np.percentile(vals, 75)),
                'iqr':          float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                'min':          float(vals.min()),
                'max':          float(vals.max()),
            })
        pd.DataFrame(summary_rows).to_csv(OUT / 'rmse_floor_summary.csv', index=False)

    rows = []
    for metric, csv_name, col in [
        ('lag1_acf',  'cross_session_acf_summary.csv',       'median_lag1_acf'),
        ('var_ratio', 'cross_session_var_ratio_summary.csv', 'median_var_ratio'),
    ]:
        p = OUT / csv_name
        if not p.exists():
            continue
        vals = pd.read_csv(p)[col].dropna().values
        rows.append({
            'metric':       metric,
            'n_sessions':   int(len(vals)),
            'grand_median': float(np.median(vals)),
            'q25':          float(np.percentile(vals, 25)),
            'q75':          float(np.percentile(vals, 75)),
            'iqr':          float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            'min':          float(vals.min()),
            'max':          float(vals.max()),
            'range':        float(vals.max() - vals.min()),
        })
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(OUT / 'grand_summary_statistics.csv', index=False)
        print(summary.to_string(index=False))

    print("Done.")


if __name__ == '__main__':
    main()
