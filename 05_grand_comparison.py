"""
08_grand_comparison.py
-----------------------
Grand model comparison — all models at h in {1, 5, 10}, lag p=10.

Reads pre-computed per-session CSVs from:
  04_linear_baselines.py → linear_baselines/per_session_results.csv
  06_convlstm.py         → convlstm/unweighted/per_session_results.csv

Spatial figures (7-9) use one primary session only:
  - ConvLSTM arrays loaded from saved eval_predictions.npz (06 output)
  - Linear model arrays re-derived by fitting on the primary session

Models: zero, rolling_mean, pixel_ar, full_frame_pca_ar, patch_lag_pca_ar, convlstm

Outputs → derivatives/modeling/grand_comparison/
  per_session_combined.csv
  fig1_rmse_strip_h1.pdf
  fig2_paired_diff_h1.pdf
  fig3_rmse_vs_horizon.pdf
  fig4_skill_vs_horizon.pdf
  fig5_spatial_strip_h1.pdf
  fig6_wilcoxon_table_h{1,5,10}.pdf
  wilcoxon_stats.csv
  fig7_spatial_comparison.pdf
  fig7b_spatial_comparison_persistence.pdf
  fig8_spatial_rmse_diff.pdf
  fig9_rmse_vs_time.pdf
  fig10_pred_std_table.pdf
  fig11_signal_vs_time.pdf
"""

import argparse
import importlib.util
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

plt.rcParams.update({
    'font.family':    'serif',
    'font.serif':     ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':      9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi':     300,
    'savefig.dpi':    300,
    'savefig.bbox':   'tight',
    'savefig.format': 'pdf',
})

from fuspredict.modeling.autoregressive import (
    fit_direct_pixel_ar_models_by_horizon,
    fit_fixed_pca_basis,
    fit_fixed_pca_ar_models_by_horizon,
    predict_direct_pixel_ar,
    predict_latent_ar,
    project_to_latent,
    reconstruct_from_latent,
)
from fuspredict.plot_utils import savefig as _savefig
from fuspredict.preprocessing.io import STAGE_STANDARDIZED
from fuspredict.project import find_repo_root, load_project_config

FIG_DPI      = 300
FIG_W_SINGLE = 3.5
FIG_W_DOUBLE = 7.0

HORIZONS  = [1, 5, 10]
PRIMARY_H = 1
LAG       = 10
ROLLING_WIN = 10

EXCLUDED_SESSIONS = {'Se27072020', 'Se31012020'}

_COLORS = {
    'zero':              '#aaaaaa',
    'persistence':       '#8c564b',
    'rolling_mean':      '#9467bd',
    'pixel_ar':          '#1f77b4',
    'full_frame_pca_ar': '#ff7f0e',
    'patch_lag_pca_ar':  '#e377c2',
    'convlstm':          '#2ca02c',
}
_LABELS = {
    'zero':              'Zero',
    'persistence':       'Persistence (n+1)',
    'rolling_mean':      'Rolling Mean (w=10)',
    'pixel_ar':          'Pixel AR',
    'full_frame_pca_ar': 'Full-frame PCA-AR',
    'patch_lag_pca_ar':  'Patch-lag PCA-AR',
    'convlstm':          'ConvLSTM',
}
_FIG_ORDER   = ['zero', 'rolling_mean', 'pixel_ar', 'full_frame_pca_ar',
                'patch_lag_pca_ar', 'convlstm']
_MODEL_ORDER = ['rolling_mean', 'pixel_ar', 'full_frame_pca_ar',
                'patch_lag_pca_ar', 'convlstm']


def _despine(ax) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def savefig(fig, path: Path) -> None:
    _savefig(fig, path.with_suffix('.pdf'), dpi=FIG_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    if len(diff) < 2 or np.all(diff == 0):
        return float('nan'), float('nan')
    result = stats.wilcoxon(a, b, alternative='two-sided')
    return float(result.statistic), float(result.pvalue)


def _bootstrap_median_diff_ci(
    a: np.ndarray, b: np.ndarray,
    n_resamples: int = 9999, rng_seed: int = 0,
) -> tuple[float, float, float]:
    diff = a - b
    med  = float(np.median(diff))
    if len(diff) < 2:
        return med, float('nan'), float('nan')
    bs = stats.bootstrap(
        (diff,), statistic=np.median, n_resamples=n_resamples,
        confidence_level=0.95, method='percentile', random_state=rng_seed,
    )
    return med, float(bs.confidence_interval.low), float(bs.confidence_interval.high)


def _sig_stars(p: float) -> str:
    if np.isnan(p): return 'n/a'
    if p < 0.001:   return '***'
    if p < 0.01:    return '**'
    if p < 0.05:    return '*'
    return 'n.s.'


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_vessel_mask(tissue_dir: Path, sid: str) -> np.ndarray | None:
    """Load pre-computed vessel mask (H, W) bool, or None."""
    mask_path = tissue_dir / f"tissue_mask_{sid}.nc"
    if not mask_path.exists():
        return None
    ds = xr.open_dataset(mask_path)
    if "vessel_mask" not in ds:
        return None
    return ds["vessel_mask"].values.astype(bool)


def _rmse_masked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.square(pred - gt)[:, mask].mean()))


def _predict_rolling_mean(
    frames: np.ndarray, win: int = ROLLING_WIN, horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict frame[t+horizon] as the mean of frames[t-win+1 : t+1]."""
    T = frames.shape[0]
    preds, gts = [], []
    for t in range(win - 1, T - horizon):
        preds.append(frames[t - win + 1: t + 1].mean(axis=0))
        gts.append(frames[t + horizon])
    return np.stack(preds), np.stack(gts)


def _load_frames_nc(nc_path: Path, train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray, str]:
    """Load (frames_train, frames_test, session_id) from a standardized .nc file."""
    ds         = xr.open_dataset(nc_path)
    session_id = str(ds.attrs.get("session_id", nc_path.stem))
    frames     = ds["frames"].values.astype(np.float32)
    ds.close()
    T        = frames.shape[0]
    n_train  = int(T * train_frac)
    return frames[:n_train], frames[n_train:], session_id


# ---------------------------------------------------------------------------
# Rolling mean baseline computation
# ---------------------------------------------------------------------------

def _compute_rolling_mean_results(
    nc_paths: list[Path],
    tissue_dir: Path,
    train_frac: float = 0.8,
) -> pd.DataFrame:
    """Compute per-session rolling mean RMSE/skill at all HORIZONS."""
    rows: list[dict] = []
    for path in nc_paths:
        _, frames_test, sid = _load_frames_nc(path, train_frac)
        if sid in EXCLUDED_SESSIONS:
            continue
        T = frames_test.shape[0]
        if T <= ROLLING_WIN + max(HORIZONS):
            continue

        vessel_mask    = _load_vessel_mask(tissue_dir, sid)
        nonvessel_mask = ~vessel_mask if vessel_mask is not None else None

        for h in HORIZONS:
            pred, gt = _predict_rolling_mean(frames_test, ROLLING_WIN, horizon=h)
            rmse_full  = float(np.sqrt(np.mean((pred - gt) ** 2)))
            zero_rmse  = float(np.sqrt(np.mean(gt ** 2)))
            skill_full = 1.0 - rmse_full / zero_rmse if zero_rmse > 0 else float('nan')

            if vessel_mask is not None:
                rmse_v  = _rmse_masked(pred, gt, vessel_mask)
                rmse_nv = _rmse_masked(pred, gt, nonvessel_mask)
                zero_v  = _rmse_masked(np.zeros_like(pred), gt, vessel_mask)
                zero_nv = _rmse_masked(np.zeros_like(pred), gt, nonvessel_mask)
                skill_v  = 1.0 - rmse_v  / zero_v  if zero_v  > 0 else float('nan')
                skill_nv = 1.0 - rmse_nv / zero_nv if zero_nv > 0 else float('nan')
            else:
                rmse_v = rmse_nv = skill_v = skill_nv = float('nan')

            rows.append({
                'session_id':      sid,    'model': 'rolling_mean',
                'horizon':         h,      'lag':   LAG,
                'rmse_full':       rmse_full,  'skill_full':      skill_full,
                'rmse_vessel':     rmse_v,     'skill_vessel':    skill_v,
                'rmse_nonvessel':  rmse_nv,    'skill_nonvessel': skill_nv,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_combined(
    modeling_dir: Path,
    exclude_sids: set[str],
) -> pd.DataFrame:
    """Merge linear-baseline and ConvLSTM CSVs into one long-form DataFrame."""
    lin_path = modeling_dir / 'linear_baselines' / 'per_session_results.csv'
    cl_path  = modeling_dir / 'convlstm' / 'unweighted' / 'per_session_results.csv'

    for p in (lin_path, cl_path):
        if not p.exists():
            raise FileNotFoundError(
                f'{p} not found.\n'
                'Run 04_linear_baselines.py and 06_convlstm.py first.')

    lin = pd.read_csv(lin_path)
    cl  = pd.read_csv(cl_path)

    all_excluded = EXCLUDED_SESSIONS | exclude_sids
    lin = lin[(lin['model'] != 'persistence') & ~lin['session_id'].isin(all_excluded)].copy()
    cl  = cl[~cl['session_id'].isin(all_excluded)].copy()
    cl['lag']   = LAG
    cl['model'] = 'convlstm'

    shared_cols = ['session_id', 'model', 'horizon', 'lag',
                   'rmse_full', 'skill_full',
                   'rmse_vessel', 'skill_vessel',
                   'rmse_nonvessel', 'skill_nonvessel']
    combined = pd.concat([lin[shared_cols], cl[shared_cols]], ignore_index=True)
    return combined[combined['lag'] == LAG].copy()


# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------

def _aligned(
    df: pd.DataFrame,
    models: list[str],
    horizon: int,
    rmse_col: str = 'rmse_full',
) -> pd.DataFrame:
    parts = []
    for m in models:
        s = (
            df[(df['model'] == m) & (df['horizon'] == horizon)]
            .set_index('session_id')[rmse_col]
            .rename(m)
        )
        if s.notna().any():
            parts.append(s)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1).dropna()


def _iqr_ylim(
    arrays: list[np.ndarray],
    iqr_scale: float = 1.5,
    margin: float = 0.05,
) -> tuple[float, float]:
    all_vals = np.concatenate([a.ravel() for a in arrays if len(a)])
    q25, q75 = np.percentile(all_vals, [25, 75])
    iqr = q75 - q25
    lo, hi = q25 - iqr_scale * iqr, q75 + iqr_scale * iqr
    span = hi - lo
    return lo - margin * span, hi + margin * span


# ---------------------------------------------------------------------------
# Figures 1-6 (operate on DataFrames)
# ---------------------------------------------------------------------------

def fig1_rmse_strip(df: pd.DataFrame, out_dir: Path) -> None:
    wide = _aligned(df, _FIG_ORDER, PRIMARY_H)
    if wide.empty:
        print('  Fig 1: no aligned data, skipping.')
        return
    rng = np.random.default_rng(0)
    all_vals_list = [wide[m].values.astype(float) for m in _FIG_ORDER]
    ylo, yhi = _iqr_ylim(all_vals_list)
    n_clipped = sum(int(np.sum((v < ylo) | (v > yhi))) for v in all_vals_list)
    zero_mean = float(np.mean(wide['zero'].values.astype(float)))

    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.5), constrained_layout=True)
    ax.axhline(zero_mean, color=_COLORS['zero'], lw=1.5, ls='--',
               label=f'Zero mean ({zero_mean:.3f})', zorder=1)
    for xi, m in enumerate(_FIG_ORDER):
        vals   = wide[m].values.astype(float)
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(xi + jitter, vals, color=_COLORS[m], s=18, alpha=0.75,
                   linewidths=0.3, edgecolors='white', zorder=3)
        ax.plot([xi - 0.25, xi + 0.25], [np.mean(vals), np.mean(vals)],
                color=_COLORS[m], lw=2.0, zorder=4, solid_capstyle='round')
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(range(len(_FIG_ORDER)))
    ax.set_xticklabels([_LABELS[m] for m in _FIG_ORDER], rotation=20, ha='right')
    ax.set_ylabel('RMSE (z-score)')
    if n_clipped:
        ax.annotate(f'{n_clipped} pts outside axis', xy=(0.99, 0.01),
                    xycoords='axes fraction', ha='right', va='bottom', fontsize=7, color='grey')
    ax.legend()
    ax.grid(axis='y', alpha=0.25, zorder=0)
    _despine(ax)
    savefig(fig, out_dir / 'fig1_rmse_strip_h1')
    print(f'  Fig 1 → fig1_rmse_strip_h1.pdf  (n={len(wide)} sessions)')


def fig2_paired_diff(df: pd.DataFrame, out_dir: Path) -> None:
    wide = _aligned(df, ['zero'] + _MODEL_ORDER, PRIMARY_H)
    if wide.empty:
        print('  Fig 2: no aligned data, skipping.')
        return
    zero_vals = wide['zero'].values.astype(float)
    rng = np.random.default_rng(1)
    diffs_list = [wide[m].values.astype(float) - zero_vals for m in _MODEL_ORDER]
    ylo, yhi   = _iqr_ylim(diffs_list)
    n_clipped  = sum(int(np.sum((d < ylo) | (d > yhi))) for d in diffs_list)

    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.5), constrained_layout=True)
    ax.axhline(0, color='black', lw=0.8, zorder=1)
    for xi, m in enumerate(_MODEL_ORDER):
        diff   = wide[m].values.astype(float) - zero_vals
        jitter = rng.uniform(-0.18, 0.18, size=len(diff))
        dot_colors = ['#2ca02c' if d < 0 else '#d62728' for d in diff]
        ax.scatter(xi + jitter, diff, color=dot_colors, s=18, alpha=0.75,
                   linewidths=0.3, edgecolors='white', zorder=3)
        ax.plot([xi - 0.25, xi + 0.25], [np.mean(diff), np.mean(diff)],
                color=_COLORS[m], lw=2.0, zorder=4, solid_capstyle='round')
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(range(len(_MODEL_ORDER)))
    ax.set_xticklabels([_LABELS[m] for m in _MODEL_ORDER], rotation=20, ha='right')
    ax.set_ylabel('RMSE − zero RMSE (z-score)')
    if n_clipped:
        ax.annotate(f'{n_clipped} pts outside axis', xy=(0.99, 0.01),
                    xycoords='axes fraction', ha='right', va='bottom', fontsize=7, color='grey')
    ax.grid(axis='y', alpha=0.25, zorder=0)
    _despine(ax)
    savefig(fig, out_dir / 'fig2_paired_diff_h1')
    print('  Fig 2 → fig2_paired_diff_h1.pdf')


def fig3_rmse_vs_horizon(df: pd.DataFrame, out_dir: Path) -> None:
    ls_map = {'zero': ':', 'rolling_mean': '--', 'pixel_ar': '-',
              'full_frame_pca_ar': '--', 'patch_lag_pca_ar': '-.', 'convlstm': '-'}
    mk_map = {'zero': 's', 'rolling_mean': '^', 'pixel_ar': 'o',
              'full_frame_pca_ar': 'o', 'patch_lag_pca_ar': 'o', 'convlstm': 'D'}
    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.5), constrained_layout=True)
    for m in _FIG_ORDER:
        means, stds = [], []
        for h in HORIZONS:
            sub = df[(df['model'] == m) & (df['horizon'] == h)]['rmse_full'].dropna()
            means.append(float(sub.mean()) if len(sub) else float('nan'))
            stds.append(float(sub.std())   if len(sub) else float('nan'))
        means_arr = np.array(means)
        stds_arr  = np.array(stds)
        ax.plot(HORIZONS, means_arr, marker=mk_map[m], ls=ls_map[m],
                color=_COLORS[m], lw=1.8, label=_LABELS[m])
        if m != 'zero':
            ax.fill_between(HORIZONS, means_arr - stds_arr, means_arr + stds_arr,
                            color=_COLORS[m], alpha=0.07)
    ax.set_xlabel('Prediction horizon (frames)')
    ax.set_ylabel('Mean RMSE (z-score) ± std')
    ax.legend()
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _despine(ax)
    savefig(fig, out_dir / 'fig3_rmse_vs_horizon')
    print('  Fig 3 → fig3_rmse_vs_horizon.pdf')


def fig4_skill_vs_horizon(df: pd.DataFrame, out_dir: Path) -> None:
    ls_map = {'rolling_mean': '--', 'pixel_ar': '-', 'full_frame_pca_ar': '--',
              'patch_lag_pca_ar': '-.', 'convlstm': '-'}
    mk_map = {'rolling_mean': '^', 'pixel_ar': 'o', 'full_frame_pca_ar': 'o',
              'patch_lag_pca_ar': 'o', 'convlstm': 'D'}
    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.5), constrained_layout=True)
    for m in _MODEL_ORDER:
        means, stds = [], []
        for h in HORIZONS:
            sub = df[(df['model'] == m) & (df['horizon'] == h)]['skill_full'].dropna()
            means.append(float(sub.mean()) if len(sub) else float('nan'))
            stds.append(float(sub.std())   if len(sub) else float('nan'))
        means_arr = np.array(means)
        stds_arr  = np.array(stds)
        ax.plot(HORIZONS, means_arr, marker=mk_map[m], ls=ls_map[m],
                color=_COLORS[m], lw=1.8, label=_LABELS[m])
        ax.fill_between(HORIZONS, means_arr - stds_arr, means_arr + stds_arr,
                        color=_COLORS[m], alpha=0.07)
    ax.axhline(0, color='black', lw=0.8, ls=':', label='Zero reference')
    ax.set_xlabel('Prediction horizon (frames)')
    ax.set_ylabel('Skill vs zero ± std\n(1 − RMSE / RMSE₀)')
    ax.legend()
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _despine(ax)
    savefig(fig, out_dir / 'fig4_skill_vs_horizon')
    print('  Fig 4 → fig4_skill_vs_horizon.pdf')


def fig5_spatial_strip(df: pd.DataFrame, out_dir: Path) -> None:
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, 3.5),
                             constrained_layout=True, sharey=False)
    for ax, rmse_col, region_label in zip(
        axes, ['rmse_vessel', 'rmse_nonvessel'], ['Vessel pixels', 'Non-vessel pixels'],
    ):
        wide = _aligned(df, _FIG_ORDER, PRIMARY_H, rmse_col=rmse_col)
        if wide.empty:
            ax.set_xlabel(f'{region_label}: no data')
            continue
        all_finite = np.concatenate([wide[m].values.astype(float)
                                     for m in _FIG_ORDER if m in wide.columns])
        all_finite = all_finite[np.isfinite(all_finite)]
        ylo, yhi   = _iqr_ylim([all_finite])
        zero_mean  = float(np.mean(wide['zero'].values.astype(float)))
        ax.axhline(zero_mean, color=_COLORS['zero'], lw=1.5, ls='--',
                   label=f'Zero mean ({zero_mean:.3f})', zorder=1)
        n_clipped = 0
        for xi, m in enumerate(_FIG_ORDER):
            if m not in wide.columns:
                continue
            vals   = wide[m].values.astype(float)
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(xi + jitter, vals, color=_COLORS[m], s=18, alpha=0.75,
                       linewidths=0.3, edgecolors='white', zorder=3)
            ax.plot([xi - 0.25, xi + 0.25], [np.mean(vals), np.mean(vals)],
                    color=_COLORS[m], lw=2.0, zorder=4, solid_capstyle='round')
            n_clipped += int(np.sum((vals < ylo) | (vals > yhi)))
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(range(len(_FIG_ORDER)))
        ax.set_xticklabels([_LABELS[m] for m in _FIG_ORDER], rotation=20, ha='right')
        ax.set_ylabel('RMSE (z-score)')
        ax.set_xlabel(region_label)
        if n_clipped:
            ax.annotate(f'{n_clipped} pts outside axis', xy=(0.99, 0.01),
                        xycoords='axes fraction', ha='right', va='bottom', fontsize=7, color='grey')
        ax.legend()
        ax.grid(axis='y', alpha=0.25, zorder=0)
        _despine(ax)
    savefig(fig, out_dir / 'fig5_spatial_strip_h1')
    print('  Fig 5 → fig5_spatial_strip_h1.pdf')


# ---------------------------------------------------------------------------
# Wilcoxon statistics
# ---------------------------------------------------------------------------

def _build_wilcoxon(
    df: pd.DataFrame, horizon: int,
    rmse_col: str = 'rmse_full', region_label: str = 'full',
) -> list[dict]:
    wide = _aligned(df, _FIG_ORDER, horizon, rmse_col=rmse_col)
    if wide.empty:
        return []
    rows: list[dict] = []
    zero_vals = wide['zero'].values.astype(float)
    for m in _MODEL_ORDER:
        if m not in wide.columns:
            continue
        vals = wide[m].values.astype(float)
        W, p = _wilcoxon(vals, zero_vals)
        med, ci_lo, ci_hi = _bootstrap_median_diff_ci(vals, zero_vals)
        rows.append({'region': region_label, 'horizon': horizon,
                     'model_A': m, 'model_B': 'zero',
                     'n_sessions': len(vals), 'W': W, 'p_value': p,
                     'median_diff': med, 'ci_low': ci_lo, 'ci_high': ci_hi})
    linear_present = [m for m in ['pixel_ar', 'full_frame_pca_ar', 'patch_lag_pca_ar']
                      if m in wide.columns]
    if linear_present and 'convlstm' in wide.columns:
        best_linear = min(linear_present,
                          key=lambda m: float(np.median(wide[m].values.astype(float))))
        a = wide['convlstm'].values.astype(float)
        b = wide[best_linear].values.astype(float)
        W, p = _wilcoxon(a, b)
        med, ci_lo, ci_hi = _bootstrap_median_diff_ci(a, b)
        rows.append({'region': region_label, 'horizon': horizon,
                     'model_A': 'convlstm', 'model_B': f'{best_linear} (best linear)',
                     'n_sessions': len(a), 'W': W, 'p_value': p,
                     'median_diff': med, 'ci_low': ci_lo, 'ci_high': ci_hi})
    return rows


def compute_wilcoxon_all(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for h in HORIZONS:
        rows.extend(_build_wilcoxon(df, h, 'rmse_full',      'full'))
        rows.extend(_build_wilcoxon(df, h, 'rmse_vessel',    'vessel'))
        rows.extend(_build_wilcoxon(df, h, 'rmse_nonvessel', 'non_vessel'))
    return pd.DataFrame(rows)


def _render_wilcoxon_table(stats_df: pd.DataFrame, horizon: int, out_dir: Path) -> None:
    mask = (stats_df['horizon'] == horizon) & (stats_df['region'] == 'full')
    sub  = stats_df[mask].copy()
    if sub.empty:
        return
    display = sub.copy()
    display['p_value']     = display['p_value'].map(
        lambda p: '<0.001' if p < 0.001 else f'{p:.3f}')
    display['stars']       = sub['p_value'].map(_sig_stars)
    display['median_diff'] = display['median_diff'].map(lambda x: f'{x:+.4f}')
    display['ci_95']       = sub.apply(
        lambda r: f'[{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}]', axis=1)
    display['W']           = display['W'].map(lambda x: f'{x:.0f}')
    display['model_A']     = display['model_A'].map(lambda m: _LABELS.get(m, m))
    cols     = ['model_A', 'model_B', 'n_sessions', 'median_diff', 'ci_95', 'W', 'p_value', 'stars']
    labels   = ['Model A', 'vs Model B', 'N', 'Median(A−B)', '95% CI', 'W', 'p-value', '']
    present  = [c for c in cols if c in display.columns]
    p_labels = [labels[cols.index(c)] for c in present]
    n_rows, n_cols = len(sub), len(present)
    fig, ax = plt.subplots(figsize=(max(FIG_W_DOUBLE, n_cols * 1.4), 0.35 * (n_rows + 1) + 0.4),
                           constrained_layout=True)
    ax.axis('off')
    ax.set_title(f'Wilcoxon tests — full frame, h={horizon}', fontsize=8, pad=4)
    tbl = ax.table(cellText=display[present].values.tolist(), colLabels=p_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    for j in range(n_cols):
        tbl[0, j].set_facecolor('white')
        tbl[0, j].set_text_props(fontweight='bold')
        tbl[0, j].set_edgecolor('#cccccc')
    for i, row in enumerate(sub.itertuples(), start=1):
        for j in range(n_cols):
            tbl[i, j].set_facecolor('#f2f2f2' if row.p_value < 0.05 else 'white')
            tbl[i, j].set_edgecolor('#cccccc')
    savefig(fig, out_dir / f'fig6_wilcoxon_table_h{horizon}')
    print(f'  Wilcoxon table h={horizon} → fig6_wilcoxon_table_h{horizon}.pdf')


def fig6_wilcoxon_table(stats_df: pd.DataFrame, out_dir: Path) -> None:
    for h in HORIZONS:
        _render_wilcoxon_table(stats_df, h, out_dir)


# ---------------------------------------------------------------------------
# Spatial figures (single primary session)
# ---------------------------------------------------------------------------

_SPATIAL_MODELS = ['rolling_mean', 'pixel_ar', 'full_frame_pca_ar',
                   'patch_lag_pca_ar', 'convlstm']
_SMOOTH_WIN = 5


def _rolling(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rolling(_SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy()


def _load_spatial_arrays(
    repo_root: Path,
    primary_nc: Path,
    primary_sid: str,
    config: dict,
) -> dict[str, dict]:
    """Return {model: {'pred': (T,H,W), 'gt': (T,H,W)}} for the primary session at h=1."""
    # Load _fit_patch_lag_pca_ar and _eval_patch_lag_pca_ar from 04_linear_baselines.py
    script_path = Path(__file__).parent / '04_linear_baselines.py'
    spec    = importlib.util.spec_from_file_location('_04', script_path)
    mod04   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod04)

    mod_cfg      = config.get('modeling', {})
    N_COMPONENTS = int(mod_cfg.get('full_frame_pca_ar', {}).get('n_components', 10))
    PATCH_K      = int(mod_cfg.get('patch_lag_pca_ar', {}).get('n_components', 10))
    PATCH_SIZE   = int(mod_cfg.get('patch_lag_pca_ar', {}).get('patch_size', 14))
    RIDGE        = float(mod_cfg.get('pixel_ar', {}).get('ridge_lambda', 1e-2))
    TRAIN_FRAC   = float(mod_cfg.get('train_frac', 0.8))

    frames_train, frames_test, _ = _load_frames_nc(primary_nc, TRAIN_FRAC)
    T_t, H_, W_ = frames_test.shape
    arrays: dict[str, dict] = {}

    # Zero
    arrays['zero'] = {'pred': np.zeros_like(frames_test[LAG:]), 'gt': frames_test[LAG:]}

    # Persistence
    arrays['persistence'] = {
        'pred': frames_test[:-1].astype(np.float32),
        'gt':   frames_test[1:].astype(np.float32),
    }

    # Pixel AR
    try:
        pix_params, _ = fit_direct_pixel_ar_models_by_horizon(
            [frames_train], p=LAG, horizons=[1], ridge_lambda=RIDGE)
        preds = np.stack([predict_direct_pixel_ar(frames_test[t - LAG: t], pix_params[1])
                          for t in range(LAG, T_t)])
        arrays['pixel_ar'] = {'pred': preds, 'gt': frames_test[LAG:]}
    except Exception as exc:
        print(f'  pixel_ar arrays FAILED: {exc}')

    # Full-frame PCA-AR
    try:
        k_eff = max(1, min(N_COMPONENTS, frames_train.shape[0] // (2 * LAG)))
        basis = fit_fixed_pca_basis([frames_train], k_eff)
        pca_params, _ = fit_fixed_pca_ar_models_by_horizon(
            [frames_train], fixed_basis=basis, ar_lag=LAG,
            horizons=[1], ridge_lambda=RIDGE)
        latent_test = project_to_latent(frames_test, basis)
        preds = np.stack([
            reconstruct_from_latent(
                predict_latent_ar(latent_test[:t], pca_params[1])[np.newaxis],
                basis, (H_, W_))[0]
            for t in range(LAG, T_t)
        ])
        arrays['full_frame_pca_ar'] = {'pred': preds, 'gt': frames_test[LAG:]}
    except Exception as exc:
        print(f'  full_frame_pca_ar arrays FAILED: {exc}')

    # Patch-lag PCA-AR
    try:
        fit_pl = mod04._fit_patch_lag_pca_ar(
            frames_train, patch_size=PATCH_SIZE, ar_lag=LAG,
            k=PATCH_K, horizons=[1], ridge_lambda=RIDGE)
        pred_by_h, _ = mod04._eval_patch_lag_pca_ar(fit_pl, frames_test)
        if 1 in pred_by_h:
            n_valid = pred_by_h[1].shape[0]
            ts      = LAG - 1 + 1
            arrays['patch_lag_pca_ar'] = {
                'pred': pred_by_h[1],
                'gt':   frames_test[ts: ts + n_valid],
            }
    except Exception as exc:
        print(f'  patch_lag_pca_ar arrays FAILED: {exc}')

    # ConvLSTM — load from saved npz
    npz_path = (repo_root / 'derivatives' / 'modeling'
                / 'convlstm' / 'unweighted' / primary_sid / 'h1' / 'eval_predictions.npz')
    if npz_path.exists():
        with np.load(str(npz_path)) as z:
            arrays['convlstm'] = {
                'pred': z['pred'].astype(np.float32),
                'gt':   z['gt'].astype(np.float32),
            }
    else:
        print(f'  ConvLSTM npz not found: {npz_path}')

    # Rolling mean
    pred_rm, gt_rm = _predict_rolling_mean(frames_test, ROLLING_WIN, horizon=1)
    arrays['rolling_mean'] = {'pred': pred_rm.astype(np.float32), 'gt': gt_rm.astype(np.float32)}

    return arrays


def fig7_spatial_comparison(arrays: dict, primary_sid: str, out_dir: Path) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    models   = [m for m in _SPATIAL_MODELS if m in arrays]
    if not models:
        print('  Fig 7: no spatial arrays, skipping.')
        return
    CMAP = 'RdBu_r'
    computed, sig_vals = [], []
    for m in models:
        gt   = arrays[m]['gt'].astype(np.float64)
        pred = arrays[m]['pred'].astype(np.float64)
        n    = min(gt.shape[0], pred.shape[0])
        gt, pred  = gt[:n], pred[:n]
        gt_mean   = gt.mean(axis=0)
        pred_mean = pred.mean(axis=0)
        computed.append((m, gt_mean, pred_mean, pred_mean - gt_mean))
        sig_vals += [np.percentile(np.abs(gt_mean), 98), np.percentile(np.abs(pred_mean), 98)]
    sig_lim  = float(np.percentile(sig_vals, 98))
    n_models = len(computed)
    fig, axes = plt.subplots(n_models, 3,
                             figsize=(FIG_W_DOUBLE, n_models * FIG_W_DOUBLE / 3),
                             constrained_layout=True)
    if n_models == 1:
        axes = axes[np.newaxis, :]
    for ri, (m, gt_mean, pred_mean, residual) in enumerate(computed):
        res_lim = float(np.percentile(np.abs(residual), 98))
        for ci, (data, lim) in enumerate([(gt_mean, sig_lim), (pred_mean, sig_lim), (residual, res_lim)]):
            axes[ri, ci].imshow(data, cmap=CMAP, vmin=-lim, vmax=lim)
            axes[ri, ci].axis('off')
        if ri == 0:
            for ci, t in enumerate(['Ground truth (mean)', 'Prediction (mean)', 'Residual: pred − GT']):
                axes[ri, ci].set_title(t, fontweight='bold')
        axes[ri, 0].text(-0.04, 0.5, _LABELS.get(m, m), transform=axes[ri, 0].transAxes,
                         ha='right', va='center', fontweight='bold', rotation=90)
    sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=-sig_lim, vmax=sig_lim))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.01, label='z-score')
    savefig(fig, out_dir / 'fig7_spatial_comparison')
    print('  Fig 7 → fig7_spatial_comparison.pdf')


def fig7b_spatial_comparison_persistence(arrays: dict, primary_sid: str, out_dir: Path) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if 'persistence' not in arrays:
        return
    CMAP = 'RdBu_r'
    gt   = arrays['persistence']['gt'].astype(np.float64)
    pred = arrays['persistence']['pred'].astype(np.float64)
    n    = min(gt.shape[0], pred.shape[0])
    gt, pred  = gt[:n], pred[:n]
    gt_mean   = gt.mean(axis=0)
    pred_mean = pred.mean(axis=0)
    residual  = pred_mean - gt_mean
    sig_lim   = float(np.percentile([np.percentile(np.abs(gt_mean), 98),
                                     np.percentile(np.abs(pred_mean), 98)], 98))
    res_lim   = float(np.percentile(np.abs(residual), 98))
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, FIG_W_DOUBLE / 3), constrained_layout=True)
    for ci, (data, lim, title) in enumerate(
        [(gt_mean, sig_lim, 'Ground truth (mean)'),
         (pred_mean, sig_lim, 'Prediction (mean)'),
         (residual, res_lim, 'Residual: pred − GT')]):
        axes[ci].imshow(data, cmap=CMAP, vmin=-lim, vmax=lim)
        axes[ci].axis('off')
        axes[ci].set_title(title, fontweight='bold')
    axes[0].text(-0.04, 0.5, _LABELS['persistence'], transform=axes[0].transAxes,
                 ha='right', va='center', fontweight='bold', rotation=90)
    sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=-sig_lim, vmax=sig_lim))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.01, label='z-score')
    savefig(fig, out_dir / 'fig7b_spatial_comparison_persistence')
    print('  Fig 7b → fig7b_spatial_comparison_persistence.pdf')


def fig8_spatial_rmse_diff(arrays: dict, primary_sid: str, out_dir: Path) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    models = [m for m in _SPATIAL_MODELS if m in arrays]
    if not models:
        return
    ref_gt    = arrays[models[0]]['gt'].astype(np.float64)
    rmse_zero = np.sqrt(np.mean(ref_gt ** 2, axis=0))
    diffs, err_lims = [], []
    for m in models:
        gt   = arrays[m]['gt'].astype(np.float64)
        pred = arrays[m]['pred'].astype(np.float64)
        n    = min(gt.shape[0], pred.shape[0])
        diff = np.sqrt(np.mean((pred[:n] - gt[:n]) ** 2, axis=0)) - rmse_zero
        diffs.append(diff)
        err_lims.append(float(np.percentile(np.abs(diff), 98)))
    err_lim  = float(np.percentile(err_lims, 98))
    n_models = len(models)
    ncols    = 3
    nrows    = (n_models + ncols - 1) // ncols
    cell     = FIG_W_DOUBLE / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell, nrows * cell), constrained_layout=True)
    axes_flat = np.array(axes).ravel()
    for idx, (m, diff) in enumerate(zip(models, diffs, strict=True)):
        axes_flat[idx].imshow(diff, cmap='bwr', vmin=-err_lim, vmax=err_lim)
        axes_flat[idx].axis('off')
        axes_flat[idx].set_title(_LABELS.get(m, m), fontweight='bold')
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    sm = ScalarMappable(cmap='bwr', norm=Normalize(vmin=-err_lim, vmax=err_lim))
    sm.set_array([])
    fig.colorbar(sm, ax=axes_flat[:n_models], orientation='horizontal',
                 shrink=0.6, pad=0.04, aspect=40,
                 label='RMSE(model) − RMSE(zero) [z-score]     blue = model better')
    savefig(fig, out_dir / 'fig8_spatial_rmse_diff')
    print('  Fig 8 → fig8_spatial_rmse_diff.pdf')


def fig9_rmse_vs_time(arrays: dict, primary_sid: str, out_dir: Path) -> None:
    models_plot = [m for m in _SPATIAL_MODELS if m in arrays]
    if not models_plot:
        return
    min_t = min(arrays[m]['gt'].shape[0] for m in models_plot)
    t     = np.arange(min_t)
    ls_map = {'rolling_mean': '--', 'pixel_ar': '-', 'full_frame_pca_ar': '--',
              'patch_lag_pca_ar': '-.', 'convlstm': '-'}
    lw_map = {'convlstm': 2.0}
    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.0), constrained_layout=True)
    ref_gt = arrays[models_plot[0]]['gt'].astype(np.float64)[:min_t]
    ax.plot(t, _rolling(np.sqrt(np.mean(ref_gt ** 2, axis=(1, 2)))),
            lw=1.2, color=_COLORS['zero'], ls=':', label=_LABELS['zero'])
    for m in models_plot:
        gt   = arrays[m]['gt'].astype(np.float64)[:min_t]
        pred = arrays[m]['pred'].astype(np.float64)[:min_t]
        ax.plot(t, _rolling(np.sqrt(np.mean((pred - gt) ** 2, axis=(1, 2)))),
                lw=lw_map.get(m, 1.6), ls=ls_map.get(m, '-'),
                color=_COLORS[m], label=_LABELS[m])
    ax.set_xlabel('Target frame (test set)')
    ax.set_ylabel('RMSE (z-score)')
    ax.legend()
    ax.grid(alpha=0.2)
    _despine(ax)
    savefig(fig, out_dir / 'fig9_rmse_vs_time')
    print('  Fig 9 → fig9_rmse_vs_time.pdf')


def fig11_signal_vs_time(arrays: dict, primary_sid: str, out_dir: Path) -> None:
    models_plot = [m for m in _SPATIAL_MODELS if m in arrays]
    if not models_plot:
        return
    min_t   = min(arrays[m]['gt'].shape[0] for m in models_plot)
    t       = np.arange(min_t)
    ls_map  = {'rolling_mean': '--', 'pixel_ar': '-', 'full_frame_pca_ar': '--',
               'patch_lag_pca_ar': '-.', 'convlstm': '-'}
    lw_map  = {'convlstm': 2.0}
    ref_gt  = arrays[models_plot[0]]['gt'].astype(np.float64)[:min_t]
    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, 3.0), constrained_layout=True)
    ax.plot(t, _rolling(ref_gt.mean(axis=(1, 2))), lw=1.8, color='black', ls='-',
            label='Ground truth', zorder=5)
    for m in models_plot:
        pred = arrays[m]['pred'].astype(np.float64)[:min_t]
        ax.plot(t, _rolling(pred.mean(axis=(1, 2))),
                lw=lw_map.get(m, 1.4), ls=ls_map.get(m, '-'),
                color=_COLORS[m], alpha=0.85, label=_LABELS[m])
    ax.set_xlabel('Target frame (test set)')
    ax.set_ylabel('Mean signal (z-score)')
    ax.legend()
    ax.grid(alpha=0.2)
    _despine(ax)
    savefig(fig, out_dir / 'fig11_signal_vs_time')
    print('  Fig 11 → fig11_signal_vs_time.pdf')


# ---------------------------------------------------------------------------
# Fig 10: pred-std table
# ---------------------------------------------------------------------------

def _collect_std_across_sessions(
    nc_paths: list[Path],
    config: dict,
    cl_base: Path,
) -> pd.DataFrame:
    """Per-session pred_std and gt_std for all models at h=1."""
    script_path = Path(__file__).parent / '04_linear_baselines.py'
    spec  = importlib.util.spec_from_file_location('_04', script_path)
    mod04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod04)

    mod_cfg      = config.get('modeling', {})
    N_COMPONENTS = int(mod_cfg.get('full_frame_pca_ar', {}).get('n_components', 10))
    PATCH_K      = int(mod_cfg.get('patch_lag_pca_ar', {}).get('n_components', 10))
    PATCH_SIZE   = int(mod_cfg.get('patch_lag_pca_ar', {}).get('patch_size', 14))
    RIDGE        = float(mod_cfg.get('pixel_ar', {}).get('ridge_lambda', 1e-2))
    TRAIN_FRAC   = float(mod_cfg.get('train_frac', 0.8))

    rows: list[dict] = []
    for path in nc_paths:
        frames_train, frames_test, sid = _load_frames_nc(path, TRAIN_FRAC)
        if sid in EXCLUDED_SESSIONS:
            continue
        print(f'    std: {sid}', end='', flush=True)
        T_t, H_, W_ = frames_test.shape
        if T_t < LAG + 2 or frames_train.shape[0] < 2 * LAG:
            print('  SKIP (too short)')
            continue

        rows.append({'session_id': sid, 'model': 'zero',
                     'pred_std': 0.0, 'gt_std': float(frames_test[LAG:].std())})

        try:
            pix_params, _ = fit_direct_pixel_ar_models_by_horizon(
                [frames_train], p=LAG, horizons=[1], ridge_lambda=RIDGE)
            preds = np.stack([predict_direct_pixel_ar(frames_test[t - LAG: t], pix_params[1])
                              for t in range(LAG, T_t)])
            rows.append({'session_id': sid, 'model': 'pixel_ar',
                         'pred_std': float(preds.std()), 'gt_std': float(frames_test[LAG:].std())})
        except Exception as exc:
            print(f'  pixel_ar FAILED: {exc}', end='')

        try:
            k_eff = max(1, min(N_COMPONENTS, frames_train.shape[0] // (2 * LAG)))
            basis = fit_fixed_pca_basis([frames_train], k_eff)
            pca_params, _ = fit_fixed_pca_ar_models_by_horizon(
                [frames_train], fixed_basis=basis, ar_lag=LAG,
                horizons=[1], ridge_lambda=RIDGE)
            latent_test = project_to_latent(frames_test, basis)
            preds = np.stack([
                reconstruct_from_latent(
                    predict_latent_ar(latent_test[:t], pca_params[1])[np.newaxis],
                    basis, (H_, W_))[0]
                for t in range(LAG, T_t)
            ])
            rows.append({'session_id': sid, 'model': 'full_frame_pca_ar',
                         'pred_std': float(preds.std()), 'gt_std': float(frames_test[LAG:].std())})
        except Exception as exc:
            print(f'  full_frame_pca_ar FAILED: {exc}', end='')

        try:
            fit_pl = mod04._fit_patch_lag_pca_ar(
                frames_train, patch_size=PATCH_SIZE, ar_lag=LAG,
                k=PATCH_K, horizons=[1], ridge_lambda=RIDGE)
            pred_by_h, _ = mod04._eval_patch_lag_pca_ar(fit_pl, frames_test)
            if 1 in pred_by_h:
                rows.append({'session_id': sid, 'model': 'patch_lag_pca_ar',
                             'pred_std': float(pred_by_h[1].std()),
                             'gt_std':   float(frames_test[LAG:].std())})
        except Exception as exc:
            print(f'  patch_lag_pca_ar FAILED: {exc}', end='')

        npz_path = cl_base / sid / 'h1' / 'eval_predictions.npz'
        if npz_path.exists():
            with np.load(str(npz_path)) as z:
                rows.append({'session_id': sid, 'model': 'convlstm',
                             'pred_std': float(z['pred'].std()),
                             'gt_std':   float(z['gt'].std())})
        else:
            print('  convlstm npz missing', end='')
        print()

    return pd.DataFrame(rows)


def fig10_pred_std_table(
    nc_paths: list[Path], config: dict, cl_base: Path, out_dir: Path,
) -> None:
    print('  Collecting pred/GT std across sessions (fits linear models)...')
    std_df = _collect_std_across_sessions(nc_paths, config, cl_base)
    if std_df.empty:
        print('  Fig 10: no data, skipping.')
        return

    counts = std_df.groupby('session_id')['model'].nunique()
    n_exp  = std_df['model'].nunique()
    complete_sids = counts[counts == n_exp].index
    n_dropped = std_df['session_id'].nunique() - len(complete_sids)
    if n_dropped:
        print(f'  Dropping {n_dropped} session(s) missing at least one model result')
    std_df = std_df[std_df['session_id'].isin(complete_sids)].copy()
    std_df.to_csv(out_dir / 'pred_std_by_session.csv', index=False)

    table_rows = []
    for m in _FIG_ORDER:
        sub = std_df[std_df['model'] == m]
        if sub.empty:
            continue
        table_rows.append({
            'model':        _LABELS[m], 'n':             len(sub),
            'mean_pred_std': float(sub['pred_std'].mean()),
            'sd_pred_std':   float(sub['pred_std'].std()),
            'mean_gt_std':   float(sub['gt_std'].mean()),
            'sd_gt_std':     float(sub['gt_std'].std()),
        })

    col_labels = ['Model', 'N', 'Mean pred σ', '± SD', 'Mean GT σ', '± SD']
    table_data = [[r['model'], str(r['n']), f"{r['mean_pred_std']:.3f}",
                   f"{r['sd_pred_std']:.3f}", f"{r['mean_gt_std']:.3f}",
                   f"{r['sd_gt_std']:.3f}"] for r in table_rows]

    n_rows, n_cols = len(table_data), len(col_labels)
    fig, ax = plt.subplots(figsize=(max(FIG_W_DOUBLE, n_cols * 1.3), 0.38 * (n_rows + 1) + 0.5),
                           constrained_layout=True)
    ax.axis('off')
    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)
    for j in range(n_cols):
        tbl[0, j].set_facecolor('white')
        tbl[0, j].set_text_props(fontweight='bold')
        tbl[0, j].set_edgecolor('#cccccc')
    for i in range(1, n_rows + 1):
        bg = '#f2f2f2' if i % 2 == 0 else 'white'
        for j in range(n_cols):
            tbl[i, j].set_facecolor(bg)
            tbl[i, j].set_edgecolor('#cccccc')
    ax.set_title('Prediction σ vs GT σ — secundo sessions, h=1  (GT z-scored → expected σ ≈ 1)',
                 fontsize=8, pad=6)
    savefig(fig, out_dir / 'fig10_pred_std_table')
    print('  Fig 10 → fig10_pred_std_table.pdf')


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    print(f'\n{"-"*100}')
    print(f'{"Model":<22} {"Horizon":>7} {"N":>4} {"Mean RMSE":>10} {"Std":>8} {"Mean skill":>11}')
    print(f'{"-"*100}')
    for h in HORIZONS:
        for m in _FIG_ORDER:
            sub = df[(df['model'] == m) & (df['horizon'] == h)]['rmse_full'].dropna()
            sk  = df[(df['model'] == m) & (df['horizon'] == h)]['skill_full'].dropna()
            if sub.empty:
                continue
            print(f'{_LABELS[m]:<22} {h:>7} {len(sub):>4} '
                  f'{sub.mean():>10.4f} {sub.std():>8.4f} {sk.mean():>+11.4f}')
        print()


def _print_wilcoxon(stats_df: pd.DataFrame) -> None:
    print(f'\n{"-"*110}')
    print(f'{"Region":<12} {"h":>3} {"Model A":<22} {"vs Model B":<26} {"N":>4} '
          f'{"W":>8} {"p-value":>10} {"Median diff":>13}')
    print(f'{"-"*110}')
    for _, r in stats_df.iterrows():
        p_str = '<0.001' if r['p_value'] < 0.001 else f'{r["p_value"]:.3f}'
        print(f'{r["region"]:<12} {int(r["horizon"]):>3} '
              f'{_LABELS.get(r["model_A"], r["model_A"]):<22} '
              f'{r["model_B"]:<26} {int(r["n_sessions"]):>4} '
              f'{r["W"]:>8.0f} {p_str:>10} {_sig_stars(r["p_value"]):>4} '
              f'{r["median_diff"]:>+13.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Grand model comparison for fUS prediction')
    parser.add_argument('--primary-sid',    default=None,
                        help='Session ID for spatial figures (default: from config)')
    parser.add_argument('--skip-spatial',   action='store_true',
                        help='Skip figs 7-9 (no model re-fitting needed)')
    parser.add_argument('--skip-std-table', action='store_true',
                        help='Skip fig 10 pred-std table')
    args = parser.parse_args()

    repo_root    = find_repo_root()
    config       = load_project_config(repo_root)
    ar_cfg       = config.get('ar_analysis', {})
    TRAIN_FRAC   = float(config.get('modeling', {}).get('train_frac', 0.8))
    EXCLUDE_SIDS = set(str(s) for s in ar_cfg.get('within_session_exclude', []))
    primary_sid  = args.primary_sid or str(ar_cfg.get('primary_session_id', 'Se25082020'))

    preproc_root     = repo_root / config['paths']['preprocessing']
    modeling_dir     = repo_root / config['paths']['modeling']
    standardized_dir = preproc_root / 'secundo' / 'baseline_only_standardized'
    tissue_dir       = preproc_root / 'secundo' / 'tissue_masks'
    cl_base          = modeling_dir / 'convlstm' / 'unweighted'

    nc_paths = sorted(
        p for p in standardized_dir.glob(f'baseline_*_unfiltered_{STAGE_STANDARDIZED}.nc')
        if not any(ex in p.stem for ex in (EXCLUDED_SESSIONS | EXCLUDE_SIDS))
    )
    assert nc_paths, f'No .nc sessions found in {standardized_dir}'

    out_dir = repo_root / 'derivatives' / 'modeling' / 'grand_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading combined CSV data...')
    df = _load_combined(modeling_dir, EXCLUDE_SIDS)
    print(f'  Models:   {sorted(df["model"].unique())}')
    print(f'  Horizons: {sorted(df["horizon"].unique())}')
    print(f'  Sessions: {df["session_id"].nunique()}')
    df.to_csv(out_dir / 'per_session_combined.csv', index=False)

    print('\n-- Computing rolling mean baseline --')
    rm_df = _compute_rolling_mean_results(nc_paths, tissue_dir, TRAIN_FRAC)
    print(f'  Rolling mean: {rm_df["session_id"].nunique()} sessions')
    df = pd.concat([df, rm_df], ignore_index=True)
    _print_summary(df)

    # Summary CSVs
    summary_rows = []
    for h in HORIZONS:
        wide_h = _aligned(df, _FIG_ORDER, h)
        for m in _FIG_ORDER:
            if wide_h.empty or m not in wide_h.columns:
                continue
            sub = wide_h[m].dropna()
            sk  = (df[(df['model'] == m) & (df['horizon'] == h)]
                   .set_index('session_id')['skill_full']
                   .reindex(wide_h.index).dropna())
            if sub.empty:
                continue
            summary_rows.append({
                'model': m, 'label': _LABELS[m], 'horizon': h,
                'n': len(sub), 'mean_rmse': float(sub.mean()), 'std_rmse': float(sub.std()),
                'mean_skill': float(sk.mean()), 'std_skill': float(sk.std()),
            })
    pd.DataFrame(summary_rows).to_csv(out_dir / 'summary_rmse.csv', index=False)

    print('\n-- Generating figures 1-6 --')
    fig1_rmse_strip(df, out_dir)
    fig2_paired_diff(df, out_dir)
    fig3_rmse_vs_horizon(df, out_dir)
    fig4_skill_vs_horizon(df, out_dir)
    fig5_spatial_strip(df, out_dir)

    print('\n-- Computing Wilcoxon statistics --')
    stats_df = compute_wilcoxon_all(df)
    stats_df.to_csv(out_dir / 'wilcoxon_stats.csv', index=False)
    _print_wilcoxon(stats_df[stats_df['region'] == 'full'])
    fig6_wilcoxon_table(stats_df, out_dir)

    if not args.skip_spatial:
        print(f'\n-- Spatial figures (primary session: {primary_sid}) --')
        primary_nc = next(
            (p for p in nc_paths if primary_sid in p.stem), None)
        if primary_nc is None:
            print(f'  WARNING: {primary_sid} not found, skipping spatial figs.')
        else:
            print('  Fitting/loading model arrays...')
            arrays = _load_spatial_arrays(repo_root, primary_nc, primary_sid, config)
            print(f'  Arrays available: {list(arrays.keys())}')
            fig7_spatial_comparison(arrays, primary_sid, out_dir)
            fig7b_spatial_comparison_persistence(arrays, primary_sid, out_dir)
            fig8_spatial_rmse_diff(arrays, primary_sid, out_dir)
            fig9_rmse_vs_time(arrays, primary_sid, out_dir)
            fig11_signal_vs_time(arrays, primary_sid, out_dir)

    if not args.skip_std_table:
        print('\n-- Fig 10: prediction std table (all sessions) --')
        fig10_pred_std_table(nc_paths, config, cl_base, out_dir)

    print(f'\nAll outputs → {out_dir}')
    print('Done.')


if __name__ == '__main__':
    main()