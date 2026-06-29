"""
04_linear_baselines.py
----------------------
Linear baseline evaluation.

Models: zero predictor, persistence, pixel-AR, full-frame PCA-AR (k=10),
patch-lag PCA-AR (patch_size × patch_size patches). All use 80/20 chronological
within-session train/test split. RMSE in z-score units.

Outputs → derivatives/modeling/linear_baselines/
  per_session_results.csv
  lag_sweep_fullframe_pca.csv
  aggregate_summary.csv
  table_1.csv / table_1_figure.png
  fig_a_rmse_ladder.png
  fig_b_spatial_diff_map.png
  fig_c_skill_by_pixel_type.png
  fig_combined_spatial_comparison.png
  fig_combined_spatial_rmse_diff.png
  fig_rmse_over_time.png
  appendix/
    {model}/rmse_vs_time.png
    {model}/spatial_comparison.png
    {model}/spatial_error_map.png
    combined_barplot.png
    horizon_degradation.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr

from fuspredict.modeling.autoregressive import (
    _tile_patches,
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

FIG_DPI = 160

_MODEL_COLORS = {
    'zero':              '#aaaaaa',
    'persistence':       '#888888',
    'pixel_ar':          '#1f77b4',
    'full_frame_pca_ar': '#ff7f0e',
    'patch_lag_pca_ar':  '#e377c2',
}
_MODEL_LABELS = {
    'zero':              'Zero',
    'persistence':       'Persistence',
    'pixel_ar':          'Pixel AR',
    'full_frame_pca_ar': 'Full-frame PCA-AR',
    'patch_lag_pca_ar':  'Patch-lag PCA-AR',
}
_COMPLEXITY_ORDER = ['zero', 'persistence', 'pixel_ar', 'full_frame_pca_ar', 'patch_lag_pca_ar']
_FIG_MODEL_ORDER  = ['zero', 'pixel_ar', 'full_frame_pca_ar', 'patch_lag_pca_ar']
_AR_MODELS        = ['pixel_ar', 'full_frame_pca_ar', 'patch_lag_pca_ar']


def savefig(fig, path: Path) -> None:
    _savefig(fig, path, dpi=FIG_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Vessel mask loading
# ---------------------------------------------------------------------------

def _load_vessel_mask(tissue_dir: Path, session_id: str) -> np.ndarray | None:
    """Return bool (H, W) vessel mask from a .nc tissue mask file, or None."""
    mask_path = tissue_dir / f"tissue_mask_{session_id}.nc"
    if not mask_path.exists():
        return None
    ds = xr.open_dataset(mask_path)
    if "vessel_mask" not in ds:
        return None
    return ds["vessel_mask"].values.astype(bool)


# ---------------------------------------------------------------------------
# RMSE helpers
# ---------------------------------------------------------------------------

def _rmse_full(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def _rmse_masked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    sq = np.square(pred - gt)
    return float(np.sqrt(sq[:, mask].mean()))


def _skill(rmse_model: float, rmse_zero: float) -> float:
    if rmse_zero > 0:
        return float(1.0 - rmse_model / rmse_zero)
    return float('nan')


# ---------------------------------------------------------------------------
# Patch-lag PCA-AR (batched PyTorch implementation)
# ---------------------------------------------------------------------------

def _fit_patch_lag_pca_ar(
    frames_train: np.ndarray,
    patch_size: int,
    ar_lag: int,
    k: int,
    horizons: list[int],
    ridge_lambda: float = 1e-2,
) -> dict:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T, H, W = frames_train.shape
    patch_radius = max(1, patch_size // 2)
    patches      = _tile_patches(H, W, patch_radius)
    P            = len(patches)
    horizons_list = sorted(set(int(h) for h in horizons))

    patch_pixel_list: list[np.ndarray] = []
    patch_idx_np    = np.zeros((H, W), dtype=np.int64)
    within_idx_np   = np.zeros((H, W), dtype=np.int64)
    patch_sizes: list[int] = []

    for pi, (rs, cs) in enumerate(patches):
        px   = frames_train[:, rs, cs].reshape(T, -1).T.astype(np.float32)
        S_p  = px.shape[0]
        nr, nc = rs.stop - rs.start, cs.stop - cs.start
        gr, gc = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        patch_pixel_list.append(px)
        patch_sizes.append(S_p)
        patch_idx_np[rs, cs]   = pi
        within_idx_np[rs, cs]  = (gr * nc + gc).reshape(nr, nc)

    S = max(patch_sizes)
    patch_pixels_np      = np.zeros((P, S, T), dtype=np.float32)
    patch_pixel_means_np = np.zeros((P, S),    dtype=np.float32)
    for pi, px in enumerate(patch_pixel_list):
        S_p  = px.shape[0]
        mu_p = px.mean(axis=1)
        patch_pixels_np[pi, :S_p, :]   = px - mu_p[:, None]
        patch_pixel_means_np[pi, :S_p] = mu_p

    patch_pixels = torch.from_numpy(patch_pixels_np).to(device)
    px_unfolded  = patch_pixels.unfold(2, ar_lag, 1)[:, :, :-1, :]
    N_lag = px_unfolded.shape[2]
    X_lag = px_unfolded.permute(0, 2, 1, 3).reshape(P, N_lag, S * ar_lag)
    k_eff = min(k, S * ar_lag - 1, N_lag - 1)
    mu    = X_lag.mean(dim=1, keepdim=True)
    X_c   = X_lag - mu
    _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
    components = Vt[:, :k_eff, :]
    mu_sq  = mu.squeeze(1)
    latent = torch.bmm(X_c, components.transpose(1, 2))

    W_by_h: dict[int, np.ndarray] = {}
    eye = torch.eye(k_eff + 1, dtype=torch.float32, device=device)
    for h in horizons_list:
        n_valid = N_lag - h
        if n_valid <= k_eff + 1:
            continue
        Xs   = latent[:, :n_valid, :]
        Ys   = latent[:, h:n_valid + h, :]
        ones = torch.ones(P, n_valid, 1, dtype=torch.float32, device=device)
        Xa   = torch.cat([ones, Xs], dim=2)
        A    = torch.bmm(Xa.transpose(1, 2), Xa) + ridge_lambda * eye
        B    = torch.bmm(Xa.transpose(1, 2), Ys)
        W_by_h[h] = torch.linalg.solve(A, B).cpu().numpy()

    return {
        'patches': patches, 'S': S, 'k_eff': k_eff, 'ar_lag': ar_lag,
        'patch_size': patch_size, 'k': k,
        'components': components.cpu().numpy(),
        'mu_sq': mu_sq.cpu().numpy(),
        'patch_pixel_means': patch_pixel_means_np,
        'W_by_h': W_by_h,
        'patch_idx_np': patch_idx_np,
        'within_idx_np': within_idx_np,
        'H': H, 'W': W, 'horizons_list': horizons_list,
    }


def _eval_patch_lag_pca_ar(
    fit: dict,
    frames_test: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Return pred_by_h[h] = (N, H, W) float32 and test frames."""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patches       = fit['patches']
    S             = fit['S']
    k_eff         = fit['k_eff']
    ar_lag        = fit['ar_lag']
    components    = torch.from_numpy(fit['components']).to(device)
    mu_sq         = torch.from_numpy(fit['mu_sq']).to(device)
    pixel_means   = torch.from_numpy(fit['patch_pixel_means']).to(device)
    W_by_h_np     = fit['W_by_h']
    patch_idx_np  = fit['patch_idx_np']
    within_idx_np = fit['within_idx_np']
    H, W_         = fit['H'], fit['W']
    horizons_list = fit['horizons_list']

    T_test = frames_test.shape[0]
    P      = len(patches)

    patch_pixels_np = np.zeros((P, S, T_test), dtype=np.float32)
    for pi, (rs, cs) in enumerate(patches):
        px   = frames_test[:, rs, cs].reshape(T_test, -1).T.astype(np.float32)
        S_p  = px.shape[0]
        train_mu = fit['patch_pixel_means'][pi, :S_p]
        patch_pixels_np[pi, :S_p, :] = px - train_mu[:, None]

    patch_pixels_t = torch.from_numpy(patch_pixels_np).to(device)
    frames_t       = torch.from_numpy(frames_test).to(device)

    px_unfolded = patch_pixels_t.unfold(2, ar_lag, 1)[:, :, :-1, :]
    N_lag = px_unfolded.shape[2]
    X_lag = px_unfolded.permute(0, 2, 1, 3).reshape(P, N_lag, S * ar_lag)
    X_c   = X_lag - mu_sq.unsqueeze(1)
    latent = torch.bmm(X_c, components.transpose(1, 2))

    comp_rs    = components.reshape(P, k_eff, S, ar_lag)
    decode_pix = comp_rs[:, :, :, -1]
    mu_pixel   = mu_sq.reshape(P, S, ar_lag)[:, :, -1] + pixel_means

    pidx = torch.from_numpy(patch_idx_np).to(device).reshape(-1)
    sidx = torch.from_numpy(within_idx_np).to(device).reshape(-1)

    pred_by_h: dict[int, np.ndarray] = {}
    for h in horizons_list:
        if h not in W_by_h_np:
            continue
        n_valid = N_lag - h
        if n_valid <= 0:
            continue
        Wh      = torch.from_numpy(W_by_h_np[h]).to(device)
        lat_all  = latent[:, :n_valid, :]
        ones_t   = torch.ones(P, n_valid, 1, dtype=torch.float32, device=device)
        Xa_all   = torch.cat([ones_t, lat_all], dim=2)
        pred_lat = torch.bmm(Xa_all, Wh)
        pred_px  = torch.einsum('ptk,pks->pts', pred_lat, decode_pix) + mu_pixel.unsqueeze(1)
        pred_flat = pred_px[pidx, :, sidx]
        pred_map  = pred_flat.T.reshape(n_valid, H, W_)
        pred_by_h[h] = pred_map.cpu().float().numpy()

    return pred_by_h, frames_test


# ---------------------------------------------------------------------------
# Per-session evaluation
# ---------------------------------------------------------------------------

def _evaluate_session(
    frames_train: np.ndarray,
    frames_test: np.ndarray,
    session_id: str,
    lag: int,
    horizons: list[int],
    n_components: int,
    patch_size: int,
    patch_k: int,
    ridge_lambda: float,
    vessel_mask: np.ndarray | None,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Evaluate all linear models on one session.

    Returns
    -------
    rows : list of dicts, one per (model, horizon)
    arrays : dict of h=1 frame arrays keyed by model name
    """
    rows: list[dict] = []
    arrays: dict[str, dict] = {}

    T_t, H_, W_    = frames_test.shape
    nonvessel_mask  = ~vessel_mask if vessel_mask is not None else None

    def _row(model, horizon, pred, gt,
             z_full, z_vessel, z_nonvessel) -> dict:
        rf = _rmse_full(pred, gt)
        rv = _rmse_masked(pred, gt, vessel_mask)    if vessel_mask    is not None else float('nan')
        rn = _rmse_masked(pred, gt, nonvessel_mask) if nonvessel_mask is not None else float('nan')
        return {
            'session_id':       session_id, 'model': model,
            'horizon':          horizon,    'lag': lag,
            'rmse_full':        rf,
            'skill_full':       _skill(rf, z_full),
            'rmse_vessel':      rv,
            'skill_vessel':     _skill(rv, z_vessel)    if z_vessel    is not None else float('nan'),
            'rmse_nonvessel':   rn,
            'skill_nonvessel':  _skill(rn, z_nonvessel) if z_nonvessel is not None else float('nan'),
        }

    # Zero predictor
    zero_ref: dict[int, dict] = {}
    for h in horizons:
        gt_frames = [frames_test[t + h - 1]
                     for t in range(lag, T_t - h + 1)]
        if not gt_frames:
            continue
        gt_stack  = np.stack(gt_frames)
        zero_pred = np.zeros_like(gt_stack)
        zf = _rmse_full(zero_pred, gt_stack)
        zv = _rmse_masked(zero_pred, gt_stack, vessel_mask)    if vessel_mask    is not None else None
        zn = _rmse_masked(zero_pred, gt_stack, nonvessel_mask) if nonvessel_mask is not None else None
        zero_ref[h] = {'rmse_full': zf, 'rmse_vessel': zv, 'rmse_nonvessel': zn}
        rows.append({
            'session_id': session_id, 'model': 'zero', 'horizon': h, 'lag': lag,
            'rmse_full':       zf, 'skill_full':       0.0,
            'rmse_vessel':     zv if zv is not None else float('nan'),
            'skill_vessel':    0.0 if zv is not None else float('nan'),
            'rmse_nonvessel':  zn if zn is not None else float('nan'),
            'skill_nonvessel': 0.0 if zn is not None else float('nan'),
        })
        if h == 1:
            arrays['zero'] = {
                'pred':    zero_pred,
                'gt':      gt_stack,
                'persist': np.stack([frames_test[t - 1] for t in range(lag, T_t)]),
            }

    # Persistence
    for h in horizons:
        ref = zero_ref.get(h)
        if ref is None:
            continue
        pred_frames = [frames_test[t - 1]           for t in range(lag, T_t - h + 1)]
        gt_frames   = [frames_test[t + h - 1]        for t in range(lag, T_t - h + 1)]
        if not gt_frames:
            continue
        pred_stack = np.stack(pred_frames)
        gt_stack   = np.stack(gt_frames)
        rows.append(_row('persistence', h, pred_stack, gt_stack,
                         ref['rmse_full'], ref['rmse_vessel'], ref['rmse_nonvessel']))
        if h == 1:
            arrays['persistence'] = {'pred': pred_stack, 'gt': gt_stack, 'persist': pred_stack}

    # Pixel AR
    try:
        pix_params, _ = fit_direct_pixel_ar_models_by_horizon(
            [frames_train], p=lag, horizons=horizons, ridge_lambda=ridge_lambda)

        for h in horizons:
            ref = zero_ref.get(h)
            if ref is None or h not in pix_params:
                continue
            preds, gts = [], []
            for t in range(lag, T_t - h + 1):
                preds.append(predict_direct_pixel_ar(frames_test[t - lag: t], pix_params[h]))
                gts.append(frames_test[t + h - 1])
            if not preds:
                continue
            pred_arr = np.stack(preds)
            gt_arr   = np.stack(gts)
            rows.append(_row('pixel_ar', h, pred_arr, gt_arr,
                             ref['rmse_full'], ref['rmse_vessel'], ref['rmse_nonvessel']))
            if h == 1:
                arrays['pixel_ar'] = {
                    'pred':    pred_arr,
                    'gt':      gt_arr,
                    'persist': frames_test[lag - 1: T_t - 1],
                }
    except Exception as exc:
        print(f'    pixel_ar FAILED: {exc}')

    # Full-frame PCA-AR
    try:
        T_train = frames_train.shape[0]
        k_eff   = max(1, min(n_components, T_train // (2 * lag)))
        basis   = fit_fixed_pca_basis([frames_train], k_eff)
        pca_params, _ = fit_fixed_pca_ar_models_by_horizon(
            [frames_train], fixed_basis=basis, ar_lag=lag,
            horizons=horizons, ridge_lambda=ridge_lambda)

        latent_test = project_to_latent(frames_test, basis)

        for h in horizons:
            ref = zero_ref.get(h)
            if ref is None or h not in pca_params:
                continue
            preds, gts, pers = [], [], []
            for t in range(lag, T_t - h + 1):
                pred_lat = predict_latent_ar(latent_test[:t], pca_params[h])
                preds.append(reconstruct_from_latent(pred_lat[np.newaxis], basis, (H_, W_))[0])
                gts.append(frames_test[t + h - 1])
                if h == 1:
                    pers.append(frames_test[t - 1])
            if not preds:
                continue
            pred_arr = np.stack(preds)
            gt_arr   = np.stack(gts)
            rows.append(_row('full_frame_pca_ar', h, pred_arr, gt_arr,
                             ref['rmse_full'], ref['rmse_vessel'], ref['rmse_nonvessel']))
            if h == 1:
                arrays['full_frame_pca_ar'] = {
                    'pred':    pred_arr,
                    'gt':      gt_arr,
                    'persist': np.stack(pers),
                }
    except Exception as exc:
        print(f'    full_frame_pca_ar FAILED: {exc}')

    # Patch-lag PCA-AR
    try:
        fit_pl = _fit_patch_lag_pca_ar(
            frames_train, patch_size=patch_size, ar_lag=lag,
            k=patch_k, horizons=horizons, ridge_lambda=ridge_lambda)
        pred_by_h, _ = _eval_patch_lag_pca_ar(fit_pl, frames_test)

        for h in horizons:
            ref = zero_ref.get(h)
            if ref is None or h not in pred_by_h:
                continue
            n_valid      = pred_by_h[h].shape[0]
            target_start = lag - 1 + h
            gt_ph        = frames_test[target_start: target_start + n_valid]
            rows.append(_row('patch_lag_pca_ar', h, pred_by_h[h], gt_ph,
                             ref['rmse_full'], ref['rmse_vessel'], ref['rmse_nonvessel']))
            if h == 1:
                arrays['patch_lag_pca_ar'] = {
                    'pred':    pred_by_h[1],
                    'gt':      gt_ph,
                    'persist': frames_test[target_start - 1: target_start - 1 + n_valid],
                }
    except Exception as exc:
        print(f'    patch_lag_pca_ar FAILED: {exc}')

    return rows, arrays


# ---------------------------------------------------------------------------
# Secondary analysis: lag sweep on full-frame PCA-AR
# ---------------------------------------------------------------------------

def _lag_sweep_fullframe(
    sessions: list[tuple[np.ndarray, np.ndarray, str]],
    lag_sweep: list[int],
    horizons: list[int],
    n_components: int,
    ridge_lambda: float,
    vessel_masks: dict[str, np.ndarray | None],
) -> pd.DataFrame:
    """Full-frame PCA-AR across lag values; returns long-form per-session DataFrame."""
    all_rows: list[dict] = []

    for si, (frames_train, frames_test, session_id) in enumerate(sessions):
        vmask          = vessel_masks.get(session_id)
        nonvessel_mask = ~vmask if vmask is not None else None
        T_t, H_, W_   = frames_test.shape
        print(f'  [{si+1}/{len(sessions)}] {session_id}')

        for lag in lag_sweep:
            try:
                T_train = frames_train.shape[0]
                k_eff   = max(1, min(n_components, T_train // (2 * lag)))
                basis   = fit_fixed_pca_basis([frames_train], k_eff)
                pca_params, _ = fit_fixed_pca_ar_models_by_horizon(
                    [frames_train], fixed_basis=basis, ar_lag=lag,
                    horizons=horizons, ridge_lambda=ridge_lambda)
                latent_test = project_to_latent(frames_test, basis)

                for h in horizons:
                    if h not in pca_params:
                        continue
                    preds, gts = [], []
                    for t in range(lag, T_t - h + 1):
                        pred_lat = predict_latent_ar(latent_test[:t], pca_params[h])
                        preds.append(reconstruct_from_latent(pred_lat[np.newaxis], basis, (H_, W_))[0])
                        gts.append(frames_test[t + h - 1])
                    if not preds:
                        continue

                    pred_arr = np.stack(preds)
                    gt_arr   = np.stack(gts)
                    zero_arr = np.zeros_like(gt_arr)
                    rf = _rmse_full(pred_arr, gt_arr)
                    zf = _rmse_full(zero_arr, gt_arr)
                    rv = _rmse_masked(pred_arr, gt_arr, vmask)          if vmask          is not None else float('nan')
                    rn = _rmse_masked(pred_arr, gt_arr, nonvessel_mask) if nonvessel_mask is not None else float('nan')
                    zv = _rmse_masked(zero_arr, gt_arr, vmask)          if vmask          is not None else float('nan')
                    zn = _rmse_masked(zero_arr, gt_arr, nonvessel_mask) if nonvessel_mask is not None else float('nan')

                    all_rows.append({
                        'session_id':      session_id, 'model': 'full_frame_pca_ar',
                        'horizon':         h,           'lag':   lag,
                        'rmse_full':       rf,          'skill_full':       _skill(rf, zf),
                        'rmse_vessel':     rv,          'skill_vessel':     _skill(rv, zv) if not np.isnan(rv) else float('nan'),
                        'rmse_nonvessel':  rn,          'skill_nonvessel':  _skill(rn, zn) if not np.isnan(rn) else float('nan'),
                    })
            except Exception as exc:
                print(f'    lag={lag} FAILED: {exc}')

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _build_agg(per_session_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std RMSE + skill_full across sessions."""
    zero_lookup = (
        per_session_df[per_session_df['model'] == 'zero']
        .groupby('horizon')['rmse_full'].mean().to_dict()
    )
    rows = []
    for (model, horizon), grp in per_session_df.groupby(['model', 'horizon']):
        rmse_vals = grp.groupby('session_id')['rmse_full'].mean().dropna().to_numpy(float)
        if len(rmse_vals) == 0:
            continue
        pooled = float(np.mean(rmse_vals))
        zero   = zero_lookup.get(horizon, float('nan'))
        rows.append({
            'model':        model,           'horizon':    int(horizon),
            'n_sessions':   int(grp['session_id'].nunique()),
            'RMSE_mean':    pooled,          'RMSE_std':   float(np.std(rmse_vals)),
            'skill_vs_zero': _skill(pooled, zero),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main paper figures and tables
# ---------------------------------------------------------------------------

def make_table_1(agg_df: pd.DataFrame, horizons: list[int], out_dir: Path) -> None:
    models = [m for m in _FIG_MODEL_ORDER if m != 'zero']
    cols: dict[str, list] = {'model': []}
    for h in horizons:
        cols[f'h={h} RMSE'] = []
    cols['skill@h=1'] = []

    zero_h1  = agg_df[(agg_df['model'] == 'zero') & (agg_df['horizon'] == 1)]
    zero_r_h1 = float(zero_h1.iloc[0]['RMSE_mean']) if not zero_h1.empty else float('nan')

    for model in models:
        sub_h1 = agg_df[(agg_df['model'] == model) & (agg_df['horizon'] == 1)]
        if sub_h1.empty:
            continue
        cols['model'].append(_MODEL_LABELS[model])
        for h in horizons:
            sub = agg_df[(agg_df['model'] == model) & (agg_df['horizon'] == h)]
            if not sub.empty:
                r = sub.iloc[0]
                cols[f'h={h} RMSE'].append(f'{r["RMSE_mean"]:.4f} ± {r["RMSE_std"]:.4f}')
            else:
                cols[f'h={h} RMSE'].append('—')
        rmse_h1 = float(sub_h1.iloc[0]['RMSE_mean'])
        skill   = _skill(rmse_h1, zero_r_h1)
        cols['skill@h=1'].append(f'{skill:+.3f}' if not np.isnan(skill) else '—')

    table = pd.DataFrame(cols)
    table.to_csv(out_dir / 'table_1.csv', index=False)

    fig, ax = plt.subplots(figsize=(max(10, len(table.columns) * 2), 0.6 * (len(table) + 2)))
    ax.axis('off')
    tbl = ax.table(cellText=table.values.tolist(), colLabels=table.columns.tolist(),
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    for j in range(len(table.columns)):
        tbl[0, j].set_facecolor('#333333')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    fig.suptitle('Table 1: RMSE summary — linear baselines', fontsize=10)
    savefig(fig, out_dir / 'table_1_figure')
    print(f'Table 1 → {out_dir / "table_1.csv"}')


def make_fig_a(agg_df: pd.DataFrame, out_dir: Path) -> None:
    """RMSE ladder plot at h=1."""
    h      = 1
    labels, means, stds, model_iter = [], [], [], []
    for model in _FIG_MODEL_ORDER:
        sub = agg_df[(agg_df['model'] == model) & (agg_df['horizon'] == h)]
        if sub.empty:
            continue
        labels.append(_MODEL_LABELS[model])
        means.append(float(sub.iloc[0]['RMSE_mean']))
        stds.append(float(sub.iloc[0]['RMSE_std']))
        model_iter.append(model)
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for xi, (m, mn, sd) in enumerate(zip(model_iter, means, stds)):
        c = _MODEL_COLORS.get(m, '#333333')
        ax.scatter(xi, mn, s=100, color=c, zorder=5)
        ax.errorbar(xi, mn, yerr=sd, fmt='none', ecolor=c, capsize=5, lw=1.5, zorder=4)

    zero_sub = agg_df[(agg_df['model'] == 'zero') & (agg_df['horizon'] == h)]
    if not zero_sub.empty:
        ax.axhline(float(zero_sub.iloc[0]['RMSE_mean']), color=_MODEL_COLORS['zero'],
                   lw=1.2, ls='--', alpha=0.7, label='Zero ref')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('RMSE (z-score) — mean ± std across sessions')
    ax.set_title('Fig A: RMSE ladder plot at h=1', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    savefig(fig, out_dir / 'fig_a_rmse_ladder')
    print(f'Fig A → {out_dir / "fig_a_rmse_ladder.png"}')


def make_fig_b(avg_arrays: dict, vessel_mask: np.ndarray | None, out_dir: Path) -> None:
    """Pixel-wise RMSE(model) − RMSE(zero) map for patch-lag PCA-AR at h=1."""
    pred = avg_arrays['pred']
    gt   = avg_arrays['gt']
    diff = np.sqrt(np.mean((pred - gt) ** 2, axis=0)) - np.sqrt(np.mean(gt ** 2, axis=0))
    lim  = float(np.percentile(np.abs(diff), 98))

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(diff, cmap='bwr', vmin=-lim, vmax=lim)
    if vessel_mask is not None:
        ax.contour(vessel_mask.astype(float), levels=[0.5], colors='black', linewidths=0.8, alpha=0.7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('RMSE(model) − RMSE(zero) [z-score]\nblue = model better')
    ax.set_title('Fig B: Spatial RMSE diff — patch-lag PCA-AR, h=1\n(black contour = vessel mask)', fontsize=10)
    ax.axis('off')
    savefig(fig, out_dir / 'fig_b_spatial_diff_map')
    print(f'Fig B → {out_dir / "fig_b_spatial_diff_map.png"}')


def make_fig_c(per_session_df: pd.DataFrame, out_dir: Path) -> None:
    """Skill score (vessel vs non-vessel) for patch-lag PCA-AR at h=1."""
    sub = per_session_df[(per_session_df['model'] == 'patch_lag_pca_ar') & (per_session_df['horizon'] == 1)]
    if sub.empty:
        print('  Fig C: no data, skipping.')
        return

    v_skill  = sub['skill_vessel'].dropna().to_numpy(float)
    nv_skill = sub['skill_nonvessel'].dropna().to_numpy(float)
    v_skill  = v_skill[np.isfinite(v_skill)]
    nv_skill = nv_skill[np.isfinite(nv_skill)]
    if len(v_skill) == 0 or len(nv_skill) == 0:
        print('  Fig C: no finite skill scores, skipping.')
        return

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    parts = ax.violinplot([nv_skill, v_skill], positions=[0, 1], showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], ['#1f77b4', '#d62728']):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    for xi, vals in zip([0, 1], [nv_skill, v_skill]):
        med = float(np.median(vals))
        ax.text(xi, med + 0.01, f'med={med:+.3f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-vessel pixels', 'Vessel pixels'], fontsize=11)
    ax.set_ylabel('Skill vs zero predictor')
    ax.set_title(f'Fig C: Skill by pixel type — patch-lag PCA-AR, h=1', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    savefig(fig, out_dir / 'fig_c_skill_by_pixel_type')
    print(f'Fig C → {out_dir / "fig_c_skill_by_pixel_type.png"}')


def make_fig_combined_spatial(primary_arrays: dict, session_id: str, out_dir: Path) -> None:
    """Combined: models × (GT mean | pred mean | residual) and spatial RMSE diff."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    models    = ['pixel_ar', 'full_frame_pca_ar', 'patch_lag_pca_ar']
    available = [(m, primary_arrays[m]) for m in models if m in primary_arrays]
    if not available:
        return

    computed = []
    sig_vals, err_vals = [], []
    for model_key, arrs in available:
        gt = arrs['gt'].astype(np.float64)
        pr = arrs['pred'].astype(np.float64)
        gt_mean = gt.mean(axis=0)
        pr_mean = pr.mean(axis=0)
        diff    = np.sqrt(np.mean((pr - gt) ** 2, axis=0)) - np.sqrt(np.mean(gt ** 2, axis=0))
        computed.append((model_key, gt_mean, pr_mean, pr_mean - gt_mean, diff))
        sig_vals += [np.percentile(np.abs(gt_mean), 98), np.percentile(np.abs(pr_mean), 98)]
        err_vals.append(np.percentile(np.abs(diff), 98))

    sig_lim  = float(np.percentile(sig_vals, 98))
    err_lim  = float(np.percentile(err_vals, 98))
    n_models = len(computed)

    fig1, axes1 = plt.subplots(n_models, 3, figsize=(3 * 3.5, n_models * 3.2), constrained_layout=True)
    if n_models == 1:
        axes1 = axes1[np.newaxis, :]
    for ri, (mk, gt_mean, pr_mean, resid, _) in enumerate(computed):
        for ci, data in enumerate([gt_mean, pr_mean, resid]):
            axes1[ri, ci].imshow(data, cmap='RdBu_r', vmin=-sig_lim, vmax=sig_lim)
            axes1[ri, ci].axis('off')
        if ri == 0:
            for ci, t in enumerate(['Ground truth (mean)', 'Prediction (mean)', 'Residual: pred − GT']):
                axes1[ri, ci].set_title(t, fontsize=9, fontweight='bold')
        axes1[ri, 0].text(-0.04, 0.5, _MODEL_LABELS.get(mk, mk),
                          transform=axes1[ri, 0].transAxes,
                          ha='right', va='center', fontsize=9, fontweight='bold', rotation=90)
    sm = ScalarMappable(cmap='RdBu_r', norm=Normalize(vmin=-sig_lim, vmax=sig_lim))
    sm.set_array([])
    fig1.colorbar(sm, ax=axes1, shrink=0.6, pad=0.01, label='z-score')
    fig1.suptitle(f'Spatial comparison — linear baselines, h=1 ({session_id})', fontsize=11, fontweight='bold')
    savefig(fig1, out_dir / 'fig_combined_spatial_comparison')

    fig2, axes2 = plt.subplots(2, 2, figsize=(7, 6.4), constrained_layout=True)
    for ri, (mk, _, _, _, diff) in enumerate(computed):
        axes2.ravel()[ri].imshow(diff, cmap='bwr', vmin=-err_lim, vmax=err_lim)
        axes2.ravel()[ri].axis('off')
        axes2.ravel()[ri].set_title(_MODEL_LABELS.get(mk, mk), fontsize=9, fontweight='bold')
    for ri in range(len(computed), 4):
        axes2.ravel()[ri].set_visible(False)
    sm2 = ScalarMappable(cmap='bwr', norm=Normalize(vmin=-err_lim, vmax=err_lim))
    sm2.set_array([])
    fig2.colorbar(sm2, ax=axes2, shrink=0.6, pad=0.01, label='RMSE(model) − RMSE(zero)')
    fig2.suptitle(f'Spatial RMSE diff — linear baselines, h=1 ({session_id})', fontsize=11, fontweight='bold')
    savefig(fig2, out_dir / 'fig_combined_spatial_rmse_diff')
    print(f'Combined spatial figures → {out_dir}')


_SMOOTH_WIN = 5


def _rolling(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rolling(_SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy()


def make_fig_rmse_over_time(avg_arrays: dict, out_dir: Path) -> None:
    present = [m for m in _FIG_MODEL_ORDER if m in avg_arrays]
    if not present:
        return
    min_t = min(avg_arrays[m]['gt'].shape[0] for m in present)
    t     = np.arange(min_t)
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ref_gt  = avg_arrays[present[-1]]['gt'].astype(np.float64)[-min_t:]
    ax.plot(t, _rolling(np.sqrt(np.mean(ref_gt ** 2, axis=(1, 2)))),
            lw=1.2, color=_MODEL_COLORS['zero'], ls=':', label='Zero')
    for mk in _FIG_MODEL_ORDER:
        if mk == 'zero' or mk not in avg_arrays:
            continue
        gt   = avg_arrays[mk]['gt'].astype(np.float64)[-min_t:]
        pred = avg_arrays[mk]['pred'].astype(np.float64)[-min_t:]
        ax.plot(t, _rolling(np.sqrt(np.mean((pred - gt) ** 2, axis=(1, 2)))),
                lw=1.6, color=_MODEL_COLORS[mk], label=_MODEL_LABELS[mk])
    ax.set_xlabel('Target frame (test set)')
    ax.set_ylabel('RMSE (z-score)')
    ax.set_title(f'RMSE vs time — all AR models, h=1\n(avg across sessions, {_SMOOTH_WIN}-frame smooth)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, out_dir / 'fig_rmse_over_time')
    print(f'Fig RMSE over time → {out_dir / "fig_rmse_over_time.png"}')


# ---------------------------------------------------------------------------
# Appendix figures
# ---------------------------------------------------------------------------

def _app_rmse_vs_time(arrays: dict, model_key: str, out_dir: Path) -> None:
    pred    = arrays['pred']
    gt      = arrays['gt']
    persist = arrays.get('persist', arrays['gt'])
    t       = np.arange(len(gt))
    mc      = _MODEL_COLORS.get(model_key, '#333333')
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(t, _rolling(np.sqrt(np.mean((pred    - gt) ** 2, axis=(1, 2)))), lw=1.4, color=mc,
            label=_MODEL_LABELS.get(model_key, model_key))
    ax.plot(t, _rolling(np.sqrt(np.mean((persist - gt) ** 2, axis=(1, 2)))), lw=1.0,
            color=_MODEL_COLORS['persistence'], ls='--', label='Persistence')
    ax.plot(t, _rolling(np.sqrt(np.mean(gt ** 2, axis=(1, 2)))), lw=1.0,
            color=_MODEL_COLORS['zero'], ls=':', label='Zero')
    ax.set_xlabel('Target frame (test set)')
    ax.set_ylabel('RMSE (z-score)')
    ax.set_title(f'Appendix: RMSE vs time, h=1 — {_MODEL_LABELS.get(model_key, model_key)}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_dir / 'rmse_vs_time')


def _app_spatial_comparison(arrays: dict, model_key: str, out_dir: Path) -> None:
    pred     = arrays['pred']
    gt       = arrays['gt']
    gt_mean  = gt.mean(axis=0)
    pr_mean  = pred.mean(axis=0)
    residual = pr_mean - gt_mean
    vabs     = float(np.percentile(np.abs(gt_mean), 98))
    res_lim  = float(np.percentile(np.abs(residual), 98))
    rmse     = float(np.sqrt(np.mean((pred - gt) ** 2)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
    axes[0].imshow(gt_mean,   cmap='RdBu_r', vmin=-vabs,    vmax=vabs)
    axes[0].set_title('Ground truth (mean)')
    axes[1].imshow(pr_mean,   cmap='RdBu_r', vmin=-vabs,    vmax=vabs)
    axes[1].set_title(_MODEL_LABELS.get(model_key, model_key))
    im = axes[2].imshow(residual, cmap='RdBu_r', vmin=-res_lim, vmax=res_lim)
    axes[2].set_title('Residual: pred − GT')
    axes[2].text(0.97, 0.03, f'RMSE={rmse:.4f}', transform=axes[2].transAxes,
                 ha='right', va='bottom', fontsize=9, color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#333333', alpha=0.7))
    for ax in axes:
        ax.axis('off')
    fig.colorbar(im, ax=axes[2], shrink=0.75, label='z-score')
    fig.suptitle(f'Appendix: spatial comparison, h=1 — {_MODEL_LABELS.get(model_key, model_key)}')
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_dir / 'spatial_comparison')


def _app_spatial_error_map(arrays: dict, model_key: str, out_dir: Path) -> None:
    pred = arrays['pred']
    gt   = arrays['gt']
    diff = np.sqrt(np.mean((pred - gt) ** 2, axis=0)) - np.sqrt(np.mean(gt ** 2, axis=0))
    lim  = float(np.percentile(np.abs(diff), 98))
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(diff, cmap='bwr', vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='RMSE(model) − RMSE(zero) [z-score]\nblue = model better')
    ax.set_title(f'Appendix: spatial error map, h=1 — {_MODEL_LABELS.get(model_key, model_key)}')
    ax.axis('off')
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_dir / 'spatial_error_map')


def _app_combined_barplot(per_session_df: pd.DataFrame, out_dir: Path) -> None:
    z_sub     = per_session_df[(per_session_df['model'] == 'zero')        & (per_session_df['horizon'] == 1)]
    p_sub     = per_session_df[(per_session_df['model'] == 'persistence') & (per_session_df['horizon'] == 1)]
    zero_mean = float(z_sub.groupby('session_id')['rmse_full'].mean().mean()) if not z_sub.empty else float('nan')
    pers_mean = float(p_sub.groupby('session_id')['rmse_full'].mean().mean()) if not p_sub.empty else float('nan')

    model_data = [(mk, per_session_df[(per_session_df['model'] == mk) & (per_session_df['horizon'] == 1)]
                   .groupby('session_id')['rmse_full'].mean().dropna().to_numpy(float))
                  for mk in _AR_MODELS
                  if not per_session_df[(per_session_df['model'] == mk) & (per_session_df['horizon'] == 1)].empty]
    if not model_data:
        return

    x      = np.arange(len(model_data))
    labels = [_MODEL_LABELS[mk] for mk, _ in model_data]
    means  = [float(v.mean()) for _, v in model_data]
    stds   = [float(v.std())  for _, v in model_data]
    colors = [_MODEL_COLORS[mk] for mk, _ in model_data]

    fig, ax = plt.subplots(figsize=(max(7, len(model_data) * 2.5 + 2), 6), constrained_layout=True)
    for xi, (_, vals) in enumerate(model_data):
        ax.bar(xi, means[xi], width=0.5, color=colors[xi], alpha=0.7, zorder=2)
        ax.errorbar(xi, means[xi], yerr=stds[xi], fmt='none', ecolor='black', capsize=5, lw=1.5, zorder=4)
        jitter = np.random.default_rng(xi).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xi + jitter, vals, color=colors[xi], s=25, alpha=0.85,
                   linewidths=0.4, edgecolors='white', zorder=5)
    if not np.isnan(pers_mean):
        ax.axhline(pers_mean, color=_MODEL_COLORS['persistence'], lw=1.5, ls='--',
                   label=f'Persistence ({pers_mean:.3f})')
    if not np.isnan(zero_mean):
        ax.axhline(zero_mean, color='red', lw=1.5, ls=':', label=f'Zero ({zero_mean:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('RMSE (z-score)')
    ax.set_title('RMSE across sessions — AR models, h=1')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_dir / 'combined_barplot')
    print(f'App combined barplot → {out_dir / "combined_barplot.png"}')


def make_horizon_degradation(agg_df: pd.DataFrame, horizons: list[int], out_dir: Path) -> None:
    ls_map = {'zero': ':', 'pixel_ar': '-', 'full_frame_pca_ar': '-', 'patch_lag_pca_ar': '-'}
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for model in _FIG_MODEL_ORDER:
        sub = agg_df[agg_df['model'] == model].sort_values('horizon')
        if sub.empty:
            continue
        c  = _MODEL_COLORS.get(model, '#333333')
        ls = ls_map.get(model, '-')
        ax.plot(sub['horizon'], sub['RMSE_mean'], f'o{ls}', color=c, lw=1.8, label=_MODEL_LABELS[model])
        ax.fill_between(sub['horizon'],
                        sub['RMSE_mean'] - sub['RMSE_std'],
                        sub['RMSE_mean'] + sub['RMSE_std'],
                        color=c, alpha=0.12)
    ax.set_xlabel('Prediction horizon (frames)')
    ax.set_ylabel('Mean RMSE ± std across sessions')
    ax.set_title('Horizon degradation — all AR models')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    savefig(fig, out_dir / 'horizon_degradation')
    print(f'Horizon degradation → {out_dir / "horizon_degradation.png"}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Linear baseline evaluation for fUS prediction.')
    parser.add_argument('--n-sessions',    type=int, default=None, help='Limit number of sessions')
    parser.add_argument('--overwrite',     action='store_true',    help='Recompute cached CSVs')
    parser.add_argument('--skip-lag-sweep', action='store_true',   help='Skip secondary lag sweep')
    args = parser.parse_args()

    repo_root = find_repo_root()
    config    = load_project_config(repo_root)
    ar_cfg    = config.get('ar_analysis', {})
    mod_cfg   = config.get('modeling', {})

    PRIMARY_LAG  = int(mod_cfg.get('n_lags', 10))
    LAG_SWEEP    = [int(v) for v in ar_cfg.get('within_session_lag_sweep', [5, 10, 20])]
    HORIZONS     = [int(h) for h in mod_cfg.get('horizons', [1, 5, 10])]
    RIDGE        = float(mod_cfg.get('pixel_ar', {}).get('ridge_lambda', 1e-2))
    N_COMPONENTS = int(mod_cfg.get('full_frame_pca_ar', {}).get('n_components', 10))
    PATCH_K      = int(mod_cfg.get('patch_lag_pca_ar', {}).get('n_components', 10))
    PATCH_SIZE   = int(mod_cfg.get('patch_lag_pca_ar', {}).get('patch_size', 14))
    EXCLUDE_SIDS = [str(s) for s in ar_cfg.get('within_session_exclude', [])]
    PRIMARY_SID  = str(ar_cfg.get('primary_session_id', 'Se25082020'))
    TRAIN_FRAC   = float(mod_cfg.get('train_frac', 0.8))

    preproc_root     = repo_root / config['paths']['preprocessing']
    standardized_dir = preproc_root / 'secundo' / 'baseline_only_standardized'
    tissue_dir       = preproc_root / 'secundo' / 'tissue_masks'

    OUT     = repo_root / 'derivatives' / 'modeling' / 'linear_baselines'
    APP_DIR = OUT / 'appendix'
    OUT.mkdir(parents=True, exist_ok=True)
    APP_DIR.mkdir(parents=True, exist_ok=True)

    # Discover session files
    nc_paths = sorted(
        p for p in standardized_dir.glob(f'baseline_*_unfiltered_{STAGE_STANDARDIZED}.nc')
        if not any(ex in p.stem for ex in EXCLUDE_SIDS)
    )
    if args.n_sessions is not None:
        nc_paths = nc_paths[:args.n_sessions]
    assert nc_paths, f'No .nc sessions found in {standardized_dir}'
    print(f'Sessions: {len(nc_paths)}  |  lag: {PRIMARY_LAG}  |  horizons: {HORIZONS}')

    # Load all sessions
    sessions: list[tuple[np.ndarray, np.ndarray, str]] = []
    for path in nc_paths:
        ds         = xr.open_dataset(path)
        da         = ds['frames']
        session_id = str(ds.attrs.get('session_id', path.stem))
        T          = da.sizes["time"]
        n_train    = int(T * TRAIN_FRAC)
        frames_train = da.values[:n_train]
        frames_test  = da.values[n_train:]
        sessions.append((frames_train, frames_test, session_id))
        ds.close()
    print(f'Loaded {len(sessions)} sessions')

    # Vessel masks
    print('\nLoading vessel masks...')
    vessel_masks: dict[str, np.ndarray | None] = {}
    for _, _, sid in sessions:
        vessel_masks[sid] = _load_vessel_mask(tissue_dir, sid)
        status = f'{vessel_masks[sid].sum()} vessel px' if vessel_masks[sid] is not None else 'not found'
        print(f'  {sid}: {status}')

    results_csv   = OUT / 'per_session_results.csv'
    lag_sweep_csv = OUT / 'lag_sweep_fullframe_pca.csv'

    # Primary per-session results
    if not args.overwrite and results_csv.exists():
        print(f'\nLoading cached results from {results_csv}')
        per_session_df = pd.read_csv(results_csv)
    else:
        all_rows: list[dict] = []
        for si, (frames_train, frames_test, session_id) in enumerate(sessions):
            vmask = vessel_masks.get(session_id)
            print(f'\n[{si+1}/{len(sessions)}] {session_id}')
            try:
                rows, _ = _evaluate_session(
                    frames_train, frames_test, session_id,
                    lag=PRIMARY_LAG, horizons=HORIZONS,
                    n_components=N_COMPONENTS,
                    patch_size=PATCH_SIZE, patch_k=PATCH_K,
                    ridge_lambda=RIDGE, vessel_mask=vmask,
                )
                all_rows.extend(rows)
            except Exception as exc:
                print(f'  FAILED: {exc}')
        per_session_df = pd.DataFrame(all_rows)
        per_session_df.to_csv(results_csv, index=False)
        print(f'\nPer-session results → {results_csv}')

    agg_df = _build_agg(per_session_df)
    agg_df.to_csv(OUT / 'aggregate_summary.csv', index=False)

    # Secondary lag sweep
    if not args.skip_lag_sweep:
        if not args.overwrite and lag_sweep_csv.exists():
            print(f'\nLoading cached lag sweep from {lag_sweep_csv}')
            lag_sweep_df = pd.read_csv(lag_sweep_csv)
        else:
            print('\nRunning lag sweep on full-frame PCA-AR...')
            lag_sweep_df = _lag_sweep_fullframe(
                sessions, lag_sweep=LAG_SWEEP, horizons=HORIZONS,
                n_components=N_COMPONENTS, ridge_lambda=RIDGE,
                vessel_masks=vessel_masks,
            )
            lag_sweep_df.to_csv(lag_sweep_csv, index=False)

    # Spatial arrays from primary session
    primary_session = next(
        ((ft, fs, sid) for ft, fs, sid in sessions if PRIMARY_SID in sid),
        sessions[0],
    )
    frames_train_p, frames_test_p, primary_sid = primary_session
    print(f'\nDeriving h=1 frame arrays from primary session ({primary_sid})...')
    _, primary_arrays = _evaluate_session(
        frames_train_p, frames_test_p, primary_sid,
        lag=PRIMARY_LAG, horizons=[1],
        n_components=N_COMPONENTS, patch_size=PATCH_SIZE, patch_k=PATCH_K,
        ridge_lambda=RIDGE, vessel_mask=vessel_masks.get(primary_sid),
    )

    # Averaged spatial arrays across all sessions
    print('\nDeriving averaged h=1 arrays across all sessions...')
    all_session_arrays: dict[str, list[dict]] = {}
    for frames_train, frames_test, sid in sessions:
        vmask = vessel_masks.get(sid)
        try:
            _, arrs = _evaluate_session(
                frames_train, frames_test, sid,
                lag=PRIMARY_LAG, horizons=[1],
                n_components=N_COMPONENTS, patch_size=PATCH_SIZE, patch_k=PATCH_K,
                ridge_lambda=RIDGE, vessel_mask=vmask,
            )
            for mk, arr in arrs.items():
                all_session_arrays.setdefault(mk, []).append(arr)
            print(f'  {sid}: ok')
        except Exception as exc:
            print(f'  {sid}: FAILED {exc}')

    def _avg_arrs(sess_list: list[dict]) -> dict | None:
        if not sess_list:
            return None
        keys  = [k for k in ('pred', 'gt', 'persist') if k in sess_list[0]]
        min_t = min(a[keys[0]].shape[0] for a in sess_list)
        return {k: np.mean(np.stack([a[k][:min_t].astype(np.float64)
                                     for a in sess_list]), axis=0).astype(np.float32)
                for k in keys}

    avg_arrays = {mk: _avg_arrs(v) for mk, v in all_session_arrays.items()}
    vmask_primary_fig = vessel_masks.get(primary_sid)

    # Figures
    print('\n-- Main figures --')
    make_table_1(agg_df, HORIZONS, OUT)
    make_fig_a(agg_df, OUT)
    if avg_arrays.get('patch_lag_pca_ar') is not None:
        make_fig_b(avg_arrays['patch_lag_pca_ar'], vmask_primary_fig, OUT)
        make_fig_c(per_session_df, OUT)
    make_fig_combined_spatial(primary_arrays, primary_sid, OUT)
    make_fig_rmse_over_time(avg_arrays, OUT)

    print('\n-- Appendix figures --')
    for model_key in _AR_MODELS:
        model_dir = APP_DIR / model_key
        arr_avg   = avg_arrays.get(model_key)
        arr_prim  = primary_arrays.get(model_key)
        if arr_avg:
            _app_rmse_vs_time(arr_avg, model_key, model_dir)
        if arr_prim:
            _app_spatial_comparison(arr_prim, model_key, model_dir)
            _app_spatial_error_map(arr_prim, model_key, model_dir)

    _app_combined_barplot(per_session_df, APP_DIR)
    make_horizon_degradation(agg_df, HORIZONS, APP_DIR)

    print(f'\nAll outputs → {OUT}')
    print('Done.')


if __name__ == '__main__':
    main()