"""
scripts/kernel_acf_sweep.py
---------------------------
Spatial-kernel ACF sweep: apply box-filter kernels of several sizes to the
z-scored frames, compute the per-pixel temporal ACF for each kernel size,
and plot mean ACF ± std vs lag on shared axes.

Produces four figures in figures/figX_kernel_acf_sweep/:
  fig_representative_session  — mean ACF ± SD across pixels, one session
  fig_cross_session           — mean ACF ± SD across sessions, all kernel sizes
  fig_paired_diff             — paired 7×7 minus 1×1 ACF per session;
                                grand mean ± 95% CI and ± SD across sessions
  fig_spatial_panels          — temporal-mean frame at each kernel size,
                                shared colorscale (4-panel row)

Kernel sizes: 1×1 (no kernel), 3×3, 5×5, 7×7

Usage:
  python scripts/kernel_acf_sweep.py
  python scripts/kernel_acf_sweep.py --config config.yml --rep-session Se01092020
  python scripts/kernel_acf_sweep.py --task-frames   # use task frames instead of baseline
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fuspredict.data.loading import load_sessions
from fuspredict.data.session import Session
from fuspredict.plot_utils import savefig
from fuspredict.project import find_repo_root, load_project_config

matplotlib.use('Agg')

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Palatino Linotype', 'Palatino', 'Georgia', 'DejaVu Serif'],
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'axes.grid':          False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.major.size':   0,
    'ytick.major.size':   3,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'legend.frameon':     False,
})

_DOUBLE_COL = 7.0

KERNEL_SIZES = [1, 3, 5, 7]
KERNEL_COLORS = ['#A0A0A0', '#9B59B6', '#3B82C4', '#E8872A']
MAX_LAG = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_smooth(frames: np.ndarray, k: int) -> np.ndarray:
    """Apply a k×k uniform box filter spatially to each frame in (T, H, W).

    NaN pixels (outside FOV) are filled with 0 before filtering so they don't
    poison neighbouring pixels, then restored as NaN afterwards.
    """
    if k <= 1:
        return frames
    from scipy.ndimage import uniform_filter
    f = frames.astype(np.float64)
    nan_mask = np.isnan(f)
    f[nan_mask] = 0.0
    smoothed = uniform_filter(f, size=(1, k, k))
    smoothed[nan_mask] = np.nan
    return smoothed


def _per_pixel_mean_acf(frames: np.ndarray, mask: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute per-pixel temporal ACF for all masked pixels, return mean ACF
    across pixels, shape (max_lag,) — lags 1..max_lag.
    """
    T, H, W = frames.shape
    ys, xs = np.where(mask)
    n_px = len(ys)
    if n_px == 0:
        return np.full(max_lag, np.nan)

    # stack masked pixels: shape (T, n_px)
    px = frames[:, ys, xs].astype(np.float64)
    px -= px.mean(axis=0, keepdims=True)

    # compute ACF via dot products; vectorized over pixels
    c0 = np.einsum('tp,tp->p', px, px)           # (n_px,)
    acf_lags = np.empty((max_lag, n_px), dtype=np.float64)
    for lag in range(1, max_lag + 1):
        clag = np.einsum('tp,tp->p', px[lag:], px[:-lag])
        with np.errstate(invalid='ignore', divide='ignore'):
            acf_lags[lag - 1] = np.where(c0 > 0, clag / c0, np.nan)

    # mean across pixels (ignore NaN); suppress empty-slice warning for all-NaN FOV pixels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(acf_lags, axis=1)   # (max_lag,)


def _per_pixel_acf_matrix(frames: np.ndarray, mask: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Return full (n_px, max_lag) ACF matrix for std computation.
    """
    T, H, W = frames.shape
    ys, xs = np.where(mask)
    n_px = len(ys)
    if n_px == 0:
        return np.full((0, max_lag), np.nan)

    px = frames[:, ys, xs].astype(np.float64)
    px -= px.mean(axis=0, keepdims=True)
    c0 = np.einsum('tp,tp->p', px, px)
    acf_mat = np.empty((n_px, max_lag), dtype=np.float64)
    for lag in range(1, max_lag + 1):
        clag = np.einsum('tp,tp->p', px[lag:], px[:-lag])
        with np.errstate(invalid='ignore', divide='ignore'):
            acf_mat[:, lag - 1] = np.where(c0 > 0, clag / c0, np.nan)
    return acf_mat


def _compute_session_kernel_acfs(
    session: Session,
    mask: np.ndarray,
    kernel_sizes: list[int],
    max_lag: int,
) -> dict[int, np.ndarray]:
    """Return {kernel_size: mean_acf (max_lag,)} for one session."""
    results = {}
    for k in kernel_sizes:
        smoothed = _box_smooth(session.frames, k)
        results[k] = _per_pixel_mean_acf(smoothed, mask, max_lag)
    return results


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _lag_axis(max_lag: int, fps: float) -> np.ndarray:
    return np.arange(1, max_lag + 1, dtype=float) / fps


def _plot_kernel_acf(
    ax: plt.Axes,
    lag_s: np.ndarray,
    mean_acf: np.ndarray,
    std_acf: np.ndarray | None,
    color: str,
    label: str,
) -> None:
    ax.plot(lag_s, mean_acf, color=color, lw=1.8, label=label)
    if std_acf is not None:
        ax.fill_between(lag_s, mean_acf - std_acf, mean_acf + std_acf,
                        color=color, alpha=0.18)


def _finalize_ax(ax: plt.Axes, fps: float, max_lag: int) -> None:
    ax.axhline(0, color='0.65', lw=0.7, ls='--')
    ci = 1.96 / np.sqrt(1)   # placeholder line; each session varies
    ax.set_ylabel('Mean per-pixel ACF')
    ax.set_xlabel('Lag (s)')
    ax.set_xlim(0, max_lag / fps)
    ax.set_ylim(-0.15, 0.80)


# ---------------------------------------------------------------------------
# Figure: representative session
# ---------------------------------------------------------------------------

def _fig_representative(
    session: Session,
    mask: np.ndarray,
    out: Path,
) -> None:
    fps = session.fps
    lag_s = _lag_axis(MAX_LAG, fps)

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.45), constrained_layout=True)

    for k, color in zip(KERNEL_SIZES, KERNEL_COLORS):
        smoothed = _box_smooth(session.frames, k)
        acf_mat  = _per_pixel_acf_matrix(smoothed, mask, MAX_LAG)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_acf = np.nanmean(acf_mat, axis=0)
            std_acf  = np.nanstd(acf_mat,  axis=0)
        label = 'no kernel (1×1)' if k == 1 else f'{k}×{k} box'
        _plot_kernel_acf(ax, lag_s, mean_acf, std_acf, color, label)

    _finalize_ax(ax, fps, MAX_LAG)
    ax.set_title(f'Per-pixel ACF by kernel size — session {session.id}  (shading = ±1 SD across pixels)', fontsize=9)
    ax.legend(loc='upper right')
    savefig(fig, out / 'fig_representative_session')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: cross-session
# ---------------------------------------------------------------------------

def _fig_cross_session(
    sessions: list[Session],
    masks: dict[str, np.ndarray],
    out: Path,
) -> None:
    # collect per-session mean ACFs: {k: list of mean_acf arrays}
    all_acfs: dict[int, list[np.ndarray]] = {k: [] for k in KERNEL_SIZES}

    for s in sessions:
        mask = masks[s.id]
        if mask.sum() == 0:
            continue
        for k in KERNEL_SIZES:
            smoothed = _box_smooth(s.frames, k)
            mean_acf = _per_pixel_mean_acf(smoothed, mask, MAX_LAG)
            all_acfs[k].append(mean_acf)

    fps = sessions[0].fps
    lag_s = _lag_axis(MAX_LAG, fps)
    n_sessions = len(sessions)

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.45), constrained_layout=True)

    for k, color in zip(KERNEL_SIZES, KERNEL_COLORS):
        curves = np.array(all_acfs[k])   # (n_sessions, MAX_LAG)
        if curves.shape[0] == 0:
            continue
        grand_mean = np.nanmean(curves, axis=0)
        grand_std  = np.nanstd(curves,  axis=0)
        label = 'no kernel (1×1)' if k == 1 else f'{k}×{k} box'
        _plot_kernel_acf(ax, lag_s, grand_mean, grand_std, color, label)

    _finalize_ax(ax, fps, MAX_LAG)
    ax.set_title(f'Per-pixel ACF by kernel size — cross-session mean ± SD  (n={n_sessions})', fontsize=9)
    ax.legend(loc='upper right')
    savefig(fig, out / 'fig_cross_session')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: paired difference (7×7 minus 1×1)
# ---------------------------------------------------------------------------

def _fig_paired_diff(
    sessions: list[Session],
    masks: dict[str, np.ndarray],
    out: Path,
) -> None:
    """
    For each session: compute mean per-pixel ACF at k=7 and k=1, then take the
    difference (7×7 minus 1×1). Plot grand mean ± SD and ± 95% CI across
    sessions vs lag. A CI band that clears zero at early lags confirms the ACF
    lift from spatial smoothing is statistically real across sessions.
    """
    diffs: list[np.ndarray] = []

    for s in sessions:
        mask = masks[s.id]
        if mask.sum() == 0:
            continue
        acf_1 = _per_pixel_mean_acf(_box_smooth(s.frames, 1), mask, MAX_LAG)
        acf_7 = _per_pixel_mean_acf(_box_smooth(s.frames, 7), mask, MAX_LAG)
        if np.any(np.isfinite(acf_1)) and np.any(np.isfinite(acf_7)):
            diffs.append(acf_7 - acf_1)

    if not diffs:
        print("  fig_paired_diff: no valid sessions, skipping.")
        return

    D      = np.array(diffs)          # (n_sessions, MAX_LAG)
    n      = D.shape[0]
    mean_d = np.nanmean(D, axis=0)
    std_d  = np.nanstd(D,  axis=0, ddof=1)
    ci95   = 1.96 * std_d / np.sqrt(n)

    fps   = sessions[0].fps
    lag_s = _lag_axis(MAX_LAG, fps)

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.45), constrained_layout=True)

    # ± SD band (lightest)
    ax.fill_between(lag_s, mean_d - std_d, mean_d + std_d,
                    color='#E8872A', alpha=0.12, label='±1 SD')
    # ± 95% CI band (darker)
    ax.fill_between(lag_s, mean_d - ci95, mean_d + ci95,
                    color='#E8872A', alpha=0.30, label='±95% CI of mean')
    ax.plot(lag_s, mean_d, color='#E8872A', lw=1.8, label='Grand mean')

    ax.axhline(0, color='0.4', lw=1.0, ls='--')
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('ΔACF  (7×7 minus 1×1)')
    ax.set_xlim(0, MAX_LAG / fps)
    ax.set_title(
        f'Paired ACF difference: 7×7 − 1×1  (n={n} sessions)\n'
        'CI clearing zero = smoothing lift is real across sessions',
        fontsize=9,
    )
    ax.legend(loc='upper right')
    savefig(fig, out / 'fig_paired_diff')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: spatial panels (temporal mean at each kernel size)
# ---------------------------------------------------------------------------

def _fig_spatial_panels(
    session: Session,
    mask: np.ndarray,
    out: Path,
) -> None:
    """
    Show the temporal mean of z-scored frames at each kernel size as a 4-panel
    row with a shared colorscale. Reveals the spatial-detail cost paid for the
    ACF lift shown in the other figures.
    """
    mean_frames = {}
    for k in KERNEL_SIZES:
        smoothed = _box_smooth(session.frames, k)
        tmean = np.where(mask, smoothed.mean(axis=0), np.nan)
        mean_frames[k] = tmean

    # shared robust colorscale across all four panels
    all_vals = np.concatenate([f[np.isfinite(f)] for f in mean_frames.values()])
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))

    fig, axes = plt.subplots(1, 4, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.32),
                             constrained_layout=True)

    for ax, k in zip(axes, KERNEL_SIZES):
        label = 'no kernel\n(1×1)' if k == 1 else f'{k}×{k} box'
        im = ax.imshow(mean_frames[k], cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # one shared colorbar on the right
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label='Signal (z-score)')

    fig.suptitle(
        f'Temporal-mean frame at each kernel size — session {session.id}',
        fontsize=9,
    )
    savefig(fig, out / 'fig_spatial_panels')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Spatial kernel ACF sweep.')
    parser.add_argument('--config',      default='config.yml')
    parser.add_argument('--rep-session', default=None,
                        help='Session ID to use for the representative plot. '
                             'Defaults to primary_session_id in config, then first available.')
    parser.add_argument('--task-frames', action='store_true',
                        help='Load task frames (task_only_standardized) instead of baseline.')
    return parser.parse_args()


def main() -> None:
    args      = _parse_args()
    repo_root = find_repo_root()
    config    = load_project_config(repo_root, config_name=args.config)

    pa    = config['persistence_analysis']
    arcfg = config.get('ar_analysis', {})

    subject = config['subjects']['all'][0]
    excluded = set(arcfg.get(
        'within_session_exclude',
        config['subjects'].get('sessions_to_exclude', {}).get(subject, []),
    ))

    preproc_root = repo_root / config['paths']['preprocessing']
    mask_dir     = preproc_root / subject / 'tissue_masks'

    if args.task_frames:
        standardized_dir = preproc_root / subject / 'task_only_standardized'
        glob_pattern = 'task_*_unfiltered_standardized_zscore.nc'
        frame_label = 'task'
    else:
        standardized_dir = preproc_root / subject / 'baseline_only_standardized'
        glob_pattern = None  # use default baseline glob
        frame_label = 'baseline'

    sessions = load_sessions(standardized_dir, mask_dir=mask_dir,
                             exclude_ids=list(excluded), glob_pattern=glob_pattern)
    assert sessions, f"No sessions loaded from {standardized_dir}"
    print(f"Loaded {len(sessions)} {frame_label} sessions")

    # Drop sessions too short to compute ACF at the requested max lag
    min_frames = MAX_LAG + 2
    short = [s for s in sessions if s.frames.shape[0] < min_frames]
    if short:
        print(f"  Dropping {len(short)} session(s) with < {min_frames} frames: "
              f"{[s.id for s in short]}")
    sessions = [s for s in sessions if s.frames.shape[0] >= min_frames]
    assert sessions, f"No sessions with >= {min_frames} frames found."

    MIN_VAR = pa['min_var']

    def _mask(s: Session) -> np.ndarray:
        fr = s.frames
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            var = np.nanvar(fr, axis=0)
        return np.all(np.isfinite(fr), axis=0) & np.isfinite(var) & (var > MIN_VAR)

    masks = {s.id: _mask(s) for s in sessions}

    fig_name = 'figX_kernel_acf_sweep_task' if args.task_frames else 'figX_kernel_acf_sweep'
    out = repo_root / 'figures' / fig_name
    out.mkdir(parents=True, exist_ok=True)

    # Choose representative session
    rep_id = (
        args.rep_session
        or arcfg.get('primary_session_id')
        or sessions[0].id
    )
    rep_session = next((s for s in sessions if s.id == rep_id), sessions[0])
    if rep_session.id != rep_id:
        print(f"  Warning: session {rep_id!r} not found, using {rep_session.id!r}")

    print(f"Representative session: {rep_session.id}")
    print("Plotting representative session...")
    _fig_representative(rep_session, masks[rep_session.id], out)

    print("Plotting cross-session figure...")
    _fig_cross_session(sessions, masks, out)

    print("Plotting paired difference figure...")
    _fig_paired_diff(sessions, masks, out)

    print("Plotting spatial panels figure...")
    _fig_spatial_panels(rep_session, masks[rep_session.id], out)

    print(f"Done. Outputs in {out}")


if __name__ == '__main__':
    main()
