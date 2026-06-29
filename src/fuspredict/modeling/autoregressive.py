"""
autoregressive.py
-----------------
Autoregressive model fitting, prediction, and evaluation for fUS frame forecasting.

All functions are pure — they take numpy arrays and return results.
No file I/O. No path loading. The caller loads frames (e.g. via xarray) and
passes them in directly.

Typical workflow
----------------
    import xarray as xr
    from fus_predict.preprocessing.preprocessing import train_test_split
    from fus_predict.modeling.autoregressive import (
        fit_direct_pixel_ar_models_by_horizon,
        evaluate_direct_pixel_ar_pooled,
    )

    # Load and split sessions
    sessions = []
    for path in nc_paths:
        ds    = xr.open_dataset(path)
        train, test = train_test_split(ds["frames"])
        sessions.append((test.values, ds.attrs["session_id"]))

    # Fit (training frames only)
    train_frames = [xr.open_dataset(p)["frames"].isel(
                        time=slice(None, int(0.8 * xr.open_dataset(p).dims["time"]))
                    ).values for p in nc_paths]

    params, fit_df = fit_direct_pixel_ar_models_by_horizon(
        train_frames, p=10, horizons=[1, 5, 10]
    )

    # Evaluate (test frames)
    per_session_df, agg_df = evaluate_direct_pixel_ar_pooled(
        sessions, params, context_frames=10
    )
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Type aliases
# Fitting functions take:  list[np.ndarray]            — frames per session
# Evaluation functions take: list[tuple[np.ndarray, str]] — (frames, name) per session
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """RMSE and MAE over all elements."""
    r = np.asarray(pred, np.float64) - np.asarray(target, np.float64)
    return {
        "RMSE": float(np.sqrt(np.mean(np.square(r)))),
        "MAE":  float(np.mean(np.abs(r))),
    }


def skill_vs_zero(model_rmse: float, zero_rmse: float) -> float:
    """Percent improvement over zero predictor (positive = better)."""
    if zero_rmse == 0.0:
        return 0.0
    return 100.0 * (zero_rmse - model_rmse) / zero_rmse


def skill_vs_persistence(model_rmse: float, persistence_rmse: float) -> float:
    """Skill score relative to persistence: positive = better than persistence."""
    if persistence_rmse == 0.0:
        return 0.0
    return 1.0 - model_rmse / persistence_rmse


def predict_persistence(frames: np.ndarray, t: int) -> np.ndarray:
    """Persistence baseline: ŷₜ₊ₕ = yₜ (last observed frame)."""
    return frames[t - 1].copy()


def compute_pixel_mse_map(
    pred_frames: np.ndarray,
    target_frames: np.ndarray,
) -> np.ndarray:
    """Mean squared error per pixel averaged over time. Returns (H, W)."""
    diff = np.asarray(pred_frames, np.float64) - np.asarray(target_frames, np.float64)
    return np.mean(np.square(diff), axis=0)


def _session_metrics_row(
    name: str,
    target_frames: np.ndarray,
    pred_frames: np.ndarray,
    zero_frames: np.ndarray,
    horizon: int,
    model_name: str,
) -> list[dict]:
    """Two rows: zero predictor and model."""
    zm = compute_metrics(zero_frames, target_frames)
    mm = compute_metrics(pred_frames, target_frames)
    base = {"session_name": name, "horizon": horizon}
    return [
        {**base, "model": "zero predictor", **zm, "skill_vs_zero_pct": 0.0},
        {**base, "model": model_name, **mm,
         "skill_vs_zero_pct": skill_vs_zero(mm["RMSE"], zm["RMSE"])},
    ]


def aggregate_session_summaries(
    per_session_df: pd.DataFrame,
    model_col: str = "model",
) -> pd.DataFrame:
    """Pool and aggregate per-session metric rows into a compact summary table."""
    rows = []
    for (model, horizon), grp in per_session_df.groupby([model_col, "horizon"], sort=False):
        zero_grp = per_session_df[
            (per_session_df[model_col] == "zero predictor")
            & (per_session_df["horizon"] == horizon)
        ]
        rmse_vals      = grp["RMSE"].to_numpy(float)
        mae_vals       = grp["MAE"].to_numpy(float)
        zero_rmse_vals = zero_grp["RMSE"].to_numpy(float)
        pooled_rmse      = float(np.mean(rmse_vals))
        pooled_zero_rmse = float(np.mean(zero_rmse_vals)) if zero_rmse_vals.size > 0 else 0.0
        rows.append({
            model_col:          model,
            "horizon":          int(horizon),
            "n_sessions":       int(grp["session_name"].nunique()),
            "RMSE_mean":        pooled_rmse,
            "RMSE_std":         float(np.std(rmse_vals)),
            "MAE_mean":         float(np.mean(mae_vals)),
            "MAE_std":          float(np.std(mae_vals)),
            "skill_vs_zero_pct": skill_vs_zero(pooled_rmse, pooled_zero_rmse),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def build_output_dir(
    repo_root: str | Path,
    section: str,
    suffix: str = "",
) -> Path:
    """Return and create the artifact directory for a modeling section."""
    base = Path(repo_root) / "derivatives" / "modeling" / str(section)
    if suffix:
        base = base / suffix
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Section 1: Direct pixel AR
# ---------------------------------------------------------------------------

def _fit_pixel_ar_single_horizon(
    session_frames: list[np.ndarray],
    p: int,
    ridge_lambda: float,
    target_horizon: int,
    max_steps: int | None = None,
) -> tuple[dict, dict]:
    """Fit per-pixel ridge AR for a single horizon across sessions."""
    p              = int(p)
    target_horizon = int(target_horizon)
    max_steps      = None if max_steps is None else int(max_steps)

    XTX = XTy = None
    H = W = 0
    steps_used = sessions_used = 0
    t0 = perf_counter()

    for frames in session_frames:
        frames = np.asarray(frames, np.float32)
        T      = frames.shape[0]
        session_had_steps = False

        for t in range(p, T - target_horizon + 1):
            if max_steps is not None and steps_used >= max_steps:
                break
            lags   = frames[t - p: t]                      # (p, H, W)
            y      = frames[t + target_horizon - 1]        # (H, W)
            H, W   = y.shape
            V      = H * W
            x_lags = lags.reshape(p, V).astype(np.float32)
            y_flat = y.reshape(V).astype(np.float32)
            x_aug  = np.vstack([np.ones((1, V), np.float32), x_lags])  # (p+1, V)

            if XTX is None:
                XTX = np.zeros((V, p + 1, p + 1), np.float64)
                XTy = np.zeros((V, p + 1),         np.float64)
            XTX += np.einsum("iv,jv->vij", x_aug, x_aug)
            XTy += np.einsum("iv,v->vi",   x_aug, y_flat)
            steps_used       += 1
            session_had_steps = True

        if session_had_steps:
            sessions_used += 1
        if max_steps is not None and steps_used >= max_steps:
            break

    if XTX is None:
        raise RuntimeError(f"No training examples for pixel AR at horizon={target_horizon}.")

    reg   = np.diag([0.0] + [ridge_lambda] * p)
    A_reg = XTX + reg[np.newaxis]   # (V, p+1, p+1)

    if _TORCH_AVAILABLE:
        try:
            W_hat = (
                _torch.linalg.solve(
                    _torch.from_numpy(A_reg),
                    _torch.from_numpy(XTy[..., np.newaxis]),
                ).squeeze(-1).numpy()
            )
        except Exception:
            W_hat = np.linalg.solve(A_reg, XTy[..., np.newaxis]).squeeze(-1)
    else:
        W_hat = np.linalg.solve(A_reg, XTy[..., np.newaxis]).squeeze(-1)

    params = {
        "A":               W_hat[:, 1:].reshape(H, W, p).astype(np.float32),
        "b":               W_hat[:, 0].reshape(H, W).astype(np.float32),
        "p":               p,
        "target_horizon":  target_horizon,
    }
    summary = {
        "target_horizon":  target_horizon,
        "model":           f"pixel_ar_p{p}_h{target_horizon}",
        "steps_used":      steps_used,
        "sessions_used":   sessions_used,
        "fit_seconds":     float(perf_counter() - t0),
    }
    return params, summary


def fit_direct_pixel_ar_models_by_horizon(
    session_frames: list[np.ndarray],
    p: int,
    horizons: list[int],
    ridge_lambda: float = 1e-2,
    max_steps: int | None = None,
) -> tuple[dict[int, dict], pd.DataFrame]:
    """
    Fit one per-pixel direct-horizon ridge AR model per horizon.

    Parameters
    ----------
    session_frames : list of np.ndarray, each (T, H, W)
        Training frames per session.
    p : int
        AR lag order.
    horizons : list of int
        Forecast horizons to fit.
    ridge_lambda : float
        Ridge regularisation.
    max_steps : int, optional
        Stop after accumulating this many lag-window steps.

    Returns
    -------
    params_by_horizon : dict[horizon → params dict]
    fit_summary_df    : DataFrame with fit metadata per horizon
    """
    params_by_horizon: dict[int, dict] = {}
    rows: list[dict] = []
    for horizon in sorted(set(int(h) for h in horizons)):
        params, summary = _fit_pixel_ar_single_horizon(
            session_frames, p=p, ridge_lambda=ridge_lambda,
            target_horizon=horizon, max_steps=max_steps,
        )
        params_by_horizon[horizon] = params
        rows.append(summary)
    fit_df = pd.DataFrame(rows).sort_values("target_horizon", kind="stable").reset_index(drop=True)
    return params_by_horizon, fit_df


def predict_direct_pixel_ar(context: np.ndarray, params: dict) -> np.ndarray:
    """
    Predict a single frame from context using pre-fitted per-pixel AR params.

    Parameters
    ----------
    context : np.ndarray, shape (T, H, W) with T >= p
        Uses the last p frames.
    params : dict with keys A (H, W, p), b (H, W), p.

    Returns
    -------
    np.ndarray, shape (H, W), float32
    """
    ctx  = np.asarray(context, np.float32)
    p    = int(params["p"])
    A    = np.asarray(params["A"], np.float32)
    b    = np.asarray(params["b"], np.float32)
    lags = ctx[-p:]  # (p, H, W)
    pred = b.copy()
    for i in range(p):
        pred += A[:, :, i] * lags[i]
    return pred


def evaluate_direct_pixel_ar_session_summary(
    frames: np.ndarray,
    name: str,
    params_by_horizon: dict[int, dict],
    context_frames: int,
) -> pd.DataFrame:
    """Per-horizon aggregate summary for one session."""
    T   = frames.shape[0]
    rows: list[dict] = []

    for horizon, params in sorted(params_by_horizon.items()):
        model_name = f"pixel_ar_p{params['p']}_h{horizon}"
        targets, preds, zeros = [], [], []
        for t in range(context_frames, T - horizon + 1):
            gt   = frames[t + horizon - 1]
            pred = predict_direct_pixel_ar(frames[t - context_frames: t], params)
            targets.append(gt)
            preds.append(pred)
            zeros.append(np.zeros_like(gt))
        if not targets:
            continue
        rows.extend(_session_metrics_row(
            name,
            np.stack(targets), np.stack(preds), np.stack(zeros),
            horizon, model_name,
        ))
    return pd.DataFrame(rows).reset_index(drop=True)


def evaluate_direct_pixel_ar_pooled(
    sessions: list[tuple[np.ndarray, str]],
    params_by_horizon: dict[int, dict],
    context_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-session summary + aggregate across all eval sessions."""
    all_rows: list[dict] = []
    for frames, name in sessions:
        try:
            sess_df = evaluate_direct_pixel_ar_session_summary(
                frames, name, params_by_horizon, context_frames
            )
            all_rows.extend(sess_df.to_dict("records"))
        except Exception:
            continue
    per_session_df = pd.DataFrame(all_rows).reset_index(drop=True)
    aggregate_df   = aggregate_session_summaries(per_session_df)
    return per_session_df, aggregate_df


def compute_spatial_error_maps(
    frames: np.ndarray,
    params_by_horizon: dict[int, dict],
    context_frames: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Mean pixel-wise squared error for persistence and pixel AR per horizon."""
    T      = frames.shape[0]
    result: dict[int, dict[str, np.ndarray]] = {}

    for horizon, params in sorted(params_by_horizon.items()):
        p_sq, m_sq = [], []
        for t in range(context_frames, T - horizon + 1):
            gt      = frames[t + horizon - 1].astype(np.float64)
            persist = frames[t - 1].astype(np.float64)
            pred    = predict_direct_pixel_ar(frames[t - context_frames: t], params).astype(np.float64)
            p_sq.append(np.square(persist - gt))
            m_sq.append(np.square(pred - gt))
        if p_sq:
            result[horizon] = {
                "persistence": np.mean(np.stack(p_sq), axis=0).astype(np.float32),
                "model":       np.mean(np.stack(m_sq), axis=0).astype(np.float32),
            }
    return result


# ---------------------------------------------------------------------------
# Section 2: Patch AR
# ---------------------------------------------------------------------------

def _tile_patches(H: int, W: int, patch_radius: int) -> list[tuple[slice, slice]]:
    """Return non-overlapping patch slices that tile (H, W)."""
    size    = 2 * patch_radius
    patches = []
    r0      = 0
    while r0 < H:
        r1 = min(r0 + size, H)
        c0 = 0
        while c0 < W:
            c1 = min(c0 + size, W)
            patches.append((slice(r0, r1), slice(c0, c1)))
            c0 = c1
        r0 = r1
    return patches


def fit_patch_ar_models_by_horizon(
    session_frames: list[np.ndarray],
    patch_radius: int,
    p: int,
    horizons: list[int],
    ridge_lambda: float = 1e-2,
) -> tuple[dict[int, dict], pd.DataFrame]:
    """
    Fit one multivariate direct-horizon ridge AR per non-overlapping patch per horizon.

    Parameters
    ----------
    session_frames : list of np.ndarray, each (T, H, W)
        Training frames per session.
    patch_radius : int
        Half-size of each square patch.
    p : int
        AR lag order.
    horizons : list of int
        Forecast horizons to fit.
    ridge_lambda : float
        Ridge regularisation.

    Returns
    -------
    params_by_horizon : dict[horizon → params dict]
    fit_summary_df    : DataFrame with fit metadata per horizon
    """
    horizons_list = sorted(set(int(h) for h in horizons))
    max_h         = max(horizons_list)
    p             = int(p)
    t0            = perf_counter()

    # Determine frame shape from first session
    sample        = np.asarray(session_frames[0], np.float32)
    H, W          = sample.shape[1], sample.shape[2]
    patches       = _tile_patches(H, W, patch_radius)
    n_patches     = len(patches)

    XTX: dict[int, list] = {h: [None] * n_patches for h in horizons_list}
    XTy: dict[int, list] = {h: [None] * n_patches for h in horizons_list}

    sessions_used = steps_used = 0
    for frames in session_frames:
        frames      = np.asarray(frames, np.float32)
        T           = frames.shape[0]
        session_had = False
        for t in range(p, T - max_h + 1):
            for pi, (rs, cs) in enumerate(patches):
                ctx_patch = frames[t - p: t, rs, cs]       # (p, ph, pw)
                Pv        = ctx_patch.shape[1] * ctx_patch.shape[2]
                x_lags    = ctx_patch.reshape(p * Pv).astype(np.float64)
                x_aug     = np.concatenate([[1.0], x_lags])  # (p*Pv + 1,)
                D1        = len(x_aug)
                for h in horizons_list:
                    target_t = t + h - 1
                    if target_t >= T:
                        continue
                    y_flat = frames[target_t, rs, cs].reshape(Pv).astype(np.float64)
                    if XTX[h][pi] is None:
                        XTX[h][pi] = np.zeros((D1, D1), np.float64)
                        XTy[h][pi] = np.zeros((D1, Pv), np.float64)
                    XTX[h][pi] += np.outer(x_aug, x_aug)
                    XTy[h][pi] += np.outer(x_aug, y_flat)
            steps_used += 1
            session_had = True
        if session_had:
            sessions_used += 1

    rows: list[dict] = []
    params_by_horizon: dict[int, dict] = {}
    for h in horizons_list:
        W_list = []
        for pi in range(n_patches):
            xtx = XTX[h][pi]
            xty = XTy[h][pi]
            if xtx is None:
                W_list.append(None)
                continue
            D1  = xtx.shape[0]
            reg = np.zeros((D1, D1), np.float64)
            reg[1:, 1:] = ridge_lambda * np.eye(D1 - 1, dtype=np.float64)
            W_hat = np.linalg.solve(xtx + reg, xty)    # (D+1, Pv)
            W_list.append(W_hat.T.astype(np.float32))  # (Pv, D+1)
        params_by_horizon[h] = {
            "patches":        patches,
            "W_list":         W_list,
            "p":              p,
            "patch_radius":   patch_radius,
            "target_horizon": h,
        }
        rows.append({
            "target_horizon":  h,
            "model":           f"patch_ar_r{patch_radius}_p{p}_h{h}",
            "n_patches":       n_patches,
            "steps_used":      steps_used,
            "sessions_used":   sessions_used,
            "fit_seconds":     float(perf_counter() - t0),
        })

    fit_df = pd.DataFrame(rows).sort_values("target_horizon", kind="stable").reset_index(drop=True)
    return params_by_horizon, fit_df


def predict_patch_ar(context: np.ndarray, params: dict) -> np.ndarray:
    """
    Predict a full frame using pre-fitted patch AR params.

    Parameters
    ----------
    context : np.ndarray, shape (T, H, W) with T >= p
        Uses the last p frames.

    Returns
    -------
    np.ndarray, shape (H, W), float32
    """
    ctx  = np.asarray(context, np.float32)
    p    = int(params["p"])
    lags = ctx[-p:]
    H, W = lags.shape[1], lags.shape[2]
    pred = np.zeros((H, W), np.float32)
    for (rs, cs), W_mat in zip(params["patches"], params["W_list"]):
        if W_mat is None:
            continue
        patch_ctx = lags[:, rs, cs]
        Pv        = patch_ctx.shape[1] * patch_ctx.shape[2]
        x_aug     = np.concatenate([[1.0], patch_ctx.reshape(p * Pv)]).astype(np.float32)
        ph        = rs.stop - rs.start
        pw        = cs.stop - cs.start
        pred[rs, cs] = (W_mat @ x_aug).reshape(ph, pw)
    return pred


def evaluate_patch_ar_pooled_with_persistence(
    sessions: list[tuple[np.ndarray, str]],
    params_by_horizon: dict[int, dict],
    context_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-session and aggregate metrics (persistence + patch AR) across sessions."""
    all_rows: list[dict] = []
    for frames, name in sessions:
        frames = np.asarray(frames, np.float32)
        T      = frames.shape[0]
        try:
            for horizon, params in sorted(params_by_horizon.items()):
                p_rmses, m_rmses = [], []
                for t in range(context_frames, T - horizon + 1):
                    gt      = frames[t + horizon - 1]
                    persist = frames[t - 1]
                    pred    = predict_patch_ar(frames[t - context_frames: t], params)
                    p_rmses.append(float(np.sqrt(np.mean(np.square(persist - gt), np.float64))))
                    m_rmses.append(float(np.sqrt(np.mean(np.square(pred - gt),    np.float64))))
                if not p_rmses:
                    continue
                pr    = int(params["patch_radius"])
                mname = f"patch_ar_r{pr}_p{params['p']}_h{horizon}"
                base  = {"session_name": name, "horizon": horizon}
                all_rows.append({**base, "model": "persistence", "RMSE": float(np.mean(p_rmses)), "MAE": float(np.mean(p_rmses))})
                all_rows.append({**base, "model": mname,         "RMSE": float(np.mean(m_rmses)), "MAE": float(np.mean(m_rmses))})
        except Exception:
            continue

    per_session_df = pd.DataFrame(all_rows).reset_index(drop=True)
    agg_rows: list[dict] = []
    for (model, horizon), grp in per_session_df.groupby(["model", "horizon"], sort=False):
        persist_grp  = per_session_df[(per_session_df["model"] == "persistence") & (per_session_df["horizon"] == horizon)]
        rmse_vals    = grp["RMSE"].to_numpy(float)
        p_rmse_vals  = persist_grp["RMSE"].to_numpy(float)
        pooled_rmse  = float(np.mean(rmse_vals))
        pooled_p     = float(np.mean(p_rmse_vals)) if p_rmse_vals.size > 0 else 0.0
        agg_rows.append({"model": model, "horizon": int(horizon),
                         "n_sessions": int(grp["session_name"].nunique()),
                         "RMSE_mean": pooled_rmse, "RMSE_std": float(np.std(rmse_vals)),
                         "skill_vs_persistence": skill_vs_persistence(pooled_rmse, pooled_p)})
    return per_session_df, pd.DataFrame(agg_rows).reset_index(drop=True)


def compute_spatial_error_maps_patch_ar(
    frames: np.ndarray,
    params_by_horizon: dict[int, dict],
    context_frames: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Mean pixel-wise squared error for persistence and patch AR per horizon."""
    T      = frames.shape[0]
    result: dict[int, dict[str, np.ndarray]] = {}
    for horizon, params in sorted(params_by_horizon.items()):
        p_sq, m_sq = [], []
        for t in range(context_frames, T - horizon + 1):
            gt      = frames[t + horizon - 1].astype(np.float64)
            persist = frames[t - 1].astype(np.float64)
            pred    = predict_patch_ar(frames[t - context_frames: t], params).astype(np.float64)
            p_sq.append(np.square(persist - gt))
            m_sq.append(np.square(pred - gt))
        if p_sq:
            result[horizon] = {
                "persistence": np.mean(np.stack(p_sq), axis=0).astype(np.float32),
                "model":       np.mean(np.stack(m_sq), axis=0).astype(np.float32),
            }
    return result


# ---------------------------------------------------------------------------
# PCA helpers (shared across Section 3 and Section 4)
# ---------------------------------------------------------------------------

def project_to_latent(frames: np.ndarray, basis: dict) -> np.ndarray:
    """Project (T, H, W) frames onto PCA basis → latent (T, k)."""
    T, H, W = frames.shape
    X = frames.reshape(T, H * W).astype(np.float32) - basis["mean"]
    return X @ basis["components"].T


def reconstruct_from_latent(
    latent: np.ndarray,
    basis: dict,
    shape: tuple[int, int],
) -> np.ndarray:
    """Reconstruct frames from latent coords → (T, H, W) float32."""
    recon = (latent @ basis["components"]) + basis["mean"]
    return recon.reshape((len(latent),) + tuple(shape)).astype(np.float32)


# ---------------------------------------------------------------------------
# Section 3: Fixed PCA basis + AR
# ---------------------------------------------------------------------------

def fit_fixed_pca_basis(
    session_frames: list[np.ndarray],
    n_components: int,
) -> dict:
    """
    Fit a single PCA basis on all concatenated training frames.

    Parameters
    ----------
    session_frames : list of np.ndarray, each (T, H, W)
        Training frames per session.
    n_components : int
        Number of PCA components.

    Returns
    -------
    dict with keys:
        components (k, H*W), mean (H*W), n_components, n_train_frames,
        explained_variance_ratio (k,)
    """
    from sklearn.decomposition import PCA as _SKLearnPCA

    chunks = [np.asarray(f, np.float32).reshape(f.shape[0], -1) for f in session_frames]
    X      = np.vstack(chunks).astype(np.float64)
    k      = min(int(n_components), X.shape[0] - 1, X.shape[1])

    pca = _SKLearnPCA(n_components=k, svd_solver="randomized", random_state=0)
    pca.fit(X)

    return {
        "components":               pca.components_.astype(np.float32),
        "mean":                     pca.mean_.astype(np.float32),
        "n_components":             k,
        "n_train_frames":           int(X.shape[0]),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
    }


def fit_fixed_pca_ar_models_by_horizon(
    session_frames: list[np.ndarray],
    fixed_basis: dict,
    ar_lag: int,
    horizons: list[int],
    ridge_lambda: float = 1e-2,
) -> tuple[dict[int, dict], pd.DataFrame]:
    """
    Fit direct-horizon ridge AR in fixed PCA latent space.

    Parameters
    ----------
    session_frames : list of np.ndarray, each (T, H, W)
        Training frames per session.
    fixed_basis : dict
        Output of fit_fixed_pca_basis.
    ar_lag : int
        AR lag order.
    horizons : list of int
        Forecast horizons.
    ridge_lambda : float
        Ridge regularisation.

    Returns
    -------
    params_by_horizon : dict[horizon → params dict]
    fit_summary_df
    """
    k             = fixed_basis["n_components"]
    horizons_list = sorted(set(int(h) for h in horizons))
    max_h         = max(horizons_list)
    t0            = perf_counter()

    Xs: dict[int, list[np.ndarray]] = {h: [] for h in horizons_list}
    Ys: dict[int, list[np.ndarray]] = {h: [] for h in horizons_list}

    for frames in session_frames:
        frames = np.asarray(frames, np.float32)
        T      = frames.shape[0]
        latent = project_to_latent(frames, fixed_basis)  # (T, k)
        for t in range(ar_lag, T - max_h + 1):
            x_lags = latent[t - ar_lag: t].reshape(-1).astype(np.float32)
            for h in horizons_list:
                target_t = t + h - 1
                if target_t >= T:
                    continue
                Xs[h].append(x_lags)
                Ys[h].append(latent[target_t])

    rows: list[dict] = []
    params_by_horizon: dict[int, dict] = {}
    for h in horizons_list:
        if not Xs[h]:
            continue
        X     = np.vstack(Xs[h]).astype(np.float64)   # (N, k*lag)
        Y     = np.vstack(Ys[h]).astype(np.float64)   # (N, k)
        N, D  = X.shape
        X_aug = np.hstack([np.ones((N, 1), np.float64), X])
        A     = X_aug.T @ X_aug + ridge_lambda * np.eye(D + 1, dtype=np.float64)
        W     = np.linalg.solve(A, X_aug.T @ Y).T     # (k, D+1)
        params_by_horizon[h] = {
            "W":              W.astype(np.float32),
            "ar_lag":         ar_lag,
            "n_components":   k,
            "target_horizon": h,
        }
        rows.append({
            "target_horizon":    h,
            "model":             f"fixed_pca_ar_d{k}_p{ar_lag}_h{h}",
            "n_train_examples":  N,
            "fit_seconds":       float(perf_counter() - t0),
        })

    fit_df = pd.DataFrame(rows).sort_values("target_horizon", kind="stable").reset_index(drop=True)
    return params_by_horizon, fit_df


def predict_latent_ar(latent_context: np.ndarray, params: dict) -> np.ndarray:
    """
    Predict a single latent vector using pre-fitted PCA-AR params.

    Parameters
    ----------
    latent_context : np.ndarray, shape (T, k)
        Uses the last ar_lag frames.

    Returns
    -------
    np.ndarray, shape (k,), float32
    """
    W      = np.asarray(params["W"],      np.float32)   # (k, k*lag+1)
    ar_lag = int(params["ar_lag"])
    lags   = latent_context[-ar_lag:].reshape(-1).astype(np.float32)
    x_aug  = np.concatenate([[1.0], lags])
    return W @ x_aug


def evaluate_fixed_pca_ar_pooled_with_persistence(
    sessions: list[tuple[np.ndarray, str]],
    latent_params_by_horizon: dict[int, dict],
    fixed_basis: dict,
    context_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-session + aggregate (persistence + fixed PCA-AR) across all sessions."""
    k        = fixed_basis["n_components"]
    max_h    = max(int(h) for h in latent_params_by_horizon)
    all_rows: list[dict] = []

    for frames, name in sessions:
        frames = np.asarray(frames, np.float32)
        T, H, W = frames.shape
        try:
            latent = project_to_latent(frames, fixed_basis)
            for h, params in sorted(latent_params_by_horizon.items()):
                ar_lag     = int(params["ar_lag"])
                model_name = f"fixed_pca_ar_d{k}_p{ar_lag}_h{h}"
                p_rmses, m_rmses = [], []
                for t in range(context_frames, T - max_h + 1):
                    target_t   = t + h - 1
                    if target_t >= T:
                        continue
                    pred_lat   = predict_latent_ar(latent[:t], params)
                    pred_frame = reconstruct_from_latent(pred_lat[np.newaxis], fixed_basis, (H, W))[0]
                    gt         = frames[target_t]
                    persist    = frames[t - 1]
                    p_rmses.append(float(np.sqrt(np.mean(np.square(persist - gt),    np.float64))))
                    m_rmses.append(float(np.sqrt(np.mean(np.square(pred_frame - gt), np.float64))))
                if not p_rmses:
                    continue
                base = {"session_name": name, "horizon": h}
                all_rows.append({**base, "model": "persistence", "RMSE": float(np.mean(p_rmses)), "MAE": float(np.mean(p_rmses))})
                all_rows.append({**base, "model": model_name,    "RMSE": float(np.mean(m_rmses)), "MAE": float(np.mean(m_rmses))})
        except Exception:
            continue

    per_session_df = pd.DataFrame(all_rows).reset_index(drop=True)
    agg_rows: list[dict] = []
    for (model, horizon), grp in per_session_df.groupby(["model", "horizon"], sort=False):
        persist_grp = per_session_df[(per_session_df["model"] == "persistence") & (per_session_df["horizon"] == horizon)]
        rmse_vals   = grp["RMSE"].to_numpy(float)
        p_rmse_vals = persist_grp["RMSE"].to_numpy(float)
        pooled_rmse = float(np.mean(rmse_vals))
        pooled_p    = float(np.mean(p_rmse_vals)) if p_rmse_vals.size > 0 else 0.0
        agg_rows.append({"model": model, "horizon": int(horizon),
                         "n_sessions":    int(grp["session_name"].nunique()),
                         "RMSE_mean":     pooled_rmse, "RMSE_std": float(np.std(rmse_vals)),
                         "skill_vs_persistence": skill_vs_persistence(pooled_rmse, pooled_p)})
    return per_session_df, pd.DataFrame(agg_rows).reset_index(drop=True)


def compute_spatial_error_maps_fixed_pca_ar(
    frames: np.ndarray,
    latent_params_by_horizon: dict[int, dict],
    fixed_basis: dict,
    context_frames: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Mean pixel-wise squared error for persistence and fixed PCA-AR per horizon."""
    T, H, W = frames.shape
    max_h   = max(int(h) for h in latent_params_by_horizon)
    latent  = project_to_latent(frames, fixed_basis)
    result: dict[int, dict[str, np.ndarray]] = {}

    for h, params in sorted(latent_params_by_horizon.items()):
        p_sq, m_sq = [], []
        for t in range(context_frames, T - max_h + 1):
            target_t = t + h - 1
            if target_t >= T:
                continue
            pred_lat   = predict_latent_ar(latent[:t], params)
            pred_frame = reconstruct_from_latent(pred_lat[np.newaxis], fixed_basis, (H, W))[0].astype(np.float64)
            gt         = frames[target_t].astype(np.float64)
            persist    = frames[t - 1].astype(np.float64)
            p_sq.append(np.square(persist - gt))
            m_sq.append(np.square(pred_frame - gt))
        if p_sq:
            result[h] = {
                "persistence": np.mean(np.stack(p_sq), axis=0).astype(np.float32),
                "model":       np.mean(np.stack(m_sq), axis=0).astype(np.float32),
            }
    return result


# ---------------------------------------------------------------------------
# Section 4: Kalman filter in PCA latent space
# ---------------------------------------------------------------------------

def estimate_kalman_params_from_latent(
    latent_traces: list[np.ndarray],
    n_components: int,
) -> np.ndarray:
    """
    Estimate per-component (F, Q, R) from aligned training latent traces.

    Returns
    -------
    np.ndarray, shape (k, 3) — each row is (F, Q, R) for one component.
    """
    kalman_params = np.zeros((n_components, 3), np.float64)
    for c in range(n_components):
        traces = [tr[:, c] for tr in latent_traces if tr.shape[0] > 2]
        if not traces:
            kalman_params[c] = [0.9, 0.1, 0.1]
            continue
        num = denom = 0.0
        for tr in traces:
            num   += float(np.sum(tr[:-1] * tr[1:]))
            denom += float(np.sum(tr[:-1] ** 2))
        F = float(np.clip(num / denom if denom > 1e-12 else 0.9, -0.999, 0.999))
        residuals_sq = [float(np.mean((tr[1:] - F * tr[:-1]) ** 2)) for tr in traces]
        Q = float(np.mean(residuals_sq)) if residuals_sq else 0.1
        R = float(Q * 0.5)
        kalman_params[c] = [F, max(Q, 1e-6), max(R, 1e-6)]
    return kalman_params


def fit_fixed_pca_kalman(
    session_frames: list[np.ndarray],
    fixed_basis: dict,
    context_frames: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Fit per-component Kalman parameters using fixed PCA projections.

    Returns
    -------
    kalman_params : np.ndarray, shape (k, 3) — (F, Q, R) per component
    fit_summary_df : pd.DataFrame
    """
    k          = fixed_basis["n_components"]
    all_traces = []
    for frames in session_frames:
        frames = np.asarray(frames, np.float32)
        if frames.shape[0] < max(context_frames, k + 2):
            continue
        all_traces.append(project_to_latent(frames, fixed_basis))

    if not all_traces:
        raise RuntimeError("No valid sessions for Kalman parameter estimation.")

    kalman_params = estimate_kalman_params_from_latent(all_traces, k)
    summary_rows  = [
        {"component": c, "F": float(kalman_params[c, 0]),
         "Q": float(kalman_params[c, 1]), "R": float(kalman_params[c, 2])}
        for c in range(k)
    ]
    return kalman_params, pd.DataFrame(summary_rows)


def predict_kalman_h_step(
    latent_context: np.ndarray,
    kalman_params: np.ndarray,
    h: int,
) -> np.ndarray:
    """
    h-step-ahead Kalman prediction for each PCA component independently.

    Parameters
    ----------
    latent_context : np.ndarray, shape (T, k)
    kalman_params  : np.ndarray, shape (k, 3) — (F, Q, R) per component
    h              : int — forecast horizon in frames

    Returns
    -------
    np.ndarray, shape (k,), float32
    """
    _, k = latent_context.shape
    pred = np.empty(k, np.float32)
    for c in range(k):
        F, Q, R = float(kalman_params[c, 0]), float(kalman_params[c, 1]), float(kalman_params[c, 2])
        obs     = latent_context[:, c].astype(np.float64)
        x_hat, P = 0.0, Q + R
        for z in obs:
            x_pred = F * x_hat
            P_pred = F * F * P + Q
            K      = P_pred / (P_pred + R)
            x_hat  = x_pred + K * (z - x_pred)
            P      = (1.0 - K) * P_pred
        pred[c] = float((F ** h) * x_hat)
    return pred


def evaluate_fixed_pca_kalman_pooled_with_persistence(
    sessions: list[tuple[np.ndarray, str]],
    kalman_params: np.ndarray,
    fixed_basis: dict,
    horizons: list[int],
    context_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-session + aggregate (persistence + fixed PCA-Kalman) across all sessions."""
    k        = fixed_basis["n_components"]
    max_h    = max(int(h) for h in horizons)
    all_rows: list[dict] = []

    for frames, name in sessions:
        frames = np.asarray(frames, np.float32)
        T, H, W = frames.shape
        try:
            latent = project_to_latent(frames, fixed_basis)
            for h in sorted(int(hh) for hh in horizons):
                model_name    = f"fixed_pca_kalman_d{k}_h{h}"
                p_rmses, m_rmses = [], []
                for t in range(context_frames, T - max_h + 1):
                    target_t   = t + h - 1
                    if target_t >= T:
                        continue
                    pred_lat   = predict_kalman_h_step(latent[:t], kalman_params, h)
                    pred_frame = reconstruct_from_latent(pred_lat[np.newaxis], fixed_basis, (H, W))[0]
                    gt         = frames[target_t]
                    persist    = frames[t - 1]
                    p_rmses.append(float(np.sqrt(np.mean(np.square(persist - gt),    np.float64))))
                    m_rmses.append(float(np.sqrt(np.mean(np.square(pred_frame - gt), np.float64))))
                if not p_rmses:
                    continue
                base = {"session_name": name, "horizon": h}
                all_rows.append({**base, "model": "persistence", "RMSE": float(np.mean(p_rmses)), "MAE": float(np.mean(p_rmses))})
                all_rows.append({**base, "model": model_name,    "RMSE": float(np.mean(m_rmses)), "MAE": float(np.mean(m_rmses))})
        except Exception:
            continue

    per_session_df = pd.DataFrame(all_rows).reset_index(drop=True)
    agg_rows: list[dict] = []
    for (model, horizon), grp in per_session_df.groupby(["model", "horizon"], sort=False):
        persist_grp = per_session_df[(per_session_df["model"] == "persistence") & (per_session_df["horizon"] == horizon)]
        rmse_vals   = grp["RMSE"].to_numpy(float)
        p_rmse_vals = persist_grp["RMSE"].to_numpy(float)
        pooled_rmse = float(np.mean(rmse_vals))
        pooled_p    = float(np.mean(p_rmse_vals)) if p_rmse_vals.size > 0 else 0.0
        agg_rows.append({"model": model, "horizon": int(horizon),
                         "n_sessions":    int(grp["session_name"].nunique()),
                         "RMSE_mean":     pooled_rmse, "RMSE_std": float(np.std(rmse_vals)),
                         "skill_vs_persistence": skill_vs_persistence(pooled_rmse, pooled_p)})
    return per_session_df, pd.DataFrame(agg_rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 5: Within-session latent AR
# ---------------------------------------------------------------------------

def evaluate_within_session_latent_ar(
    frames: np.ndarray,
    name: str,
    basis: dict,
    ar_lag: int,
    horizons: list[int],
    model_label: str,
    ridge_lambda: float = 1e-2,
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]], np.ndarray]:
    """
    Fit a direct-horizon ridge AR in a pre-computed latent space and evaluate.

    PCA (or any other linear basis) is fit externally and passed in.
    AR weights are fit on all valid lag windows of the session latent trace.

    Parameters
    ----------
    frames      : np.ndarray, shape (T, H, W) — full session frames
    name        : str — session identifier used in DataFrame output
    basis       : dict with keys components (k, H*W), mean (H*W), n_components
    ar_lag      : int — AR lag order
    horizons    : list of int — forecast horizons
    model_label : str — model name in output DataFrames
    ridge_lambda: float — ridge regularisation

    Returns
    -------
    trace_df       : columns [session_name, target_frame, horizon, model, RMSE]
    spatial_maps   : {horizon → {'persistence': (H,W), 'model': (H,W)}}
    pred_frames_h1 : (N, H, W) float32 — predicted frames at h=1
    """
    frames        = np.asarray(frames, np.float32)
    T, H, W       = frames.shape
    horizons_list = sorted(set(int(h) for h in horizons))
    k             = int(basis["n_components"])
    latent        = project_to_latent(frames, basis)   # (T, k)

    # Fit AR in latent space
    ar_params_by_horizon: dict[int, dict] = {}
    for h in horizons_list:
        Xs, Ys = [], []
        for t in range(ar_lag, T - h + 1):
            Xs.append(latent[t - ar_lag: t].reshape(-1).astype(np.float32))
            Ys.append(latent[t + h - 1].astype(np.float32))
        if not Xs:
            continue
        Xm = np.vstack(Xs).astype(np.float64)
        Ym = np.vstack(Ys).astype(np.float64)
        N, D = Xm.shape
        Xa   = np.hstack([np.ones((N, 1), np.float64), Xm])
        Wh   = np.linalg.solve(Xa.T @ Xa + ridge_lambda * np.eye(D + 1, dtype=np.float64), Xa.T @ Ym).T.astype(np.float32)
        ar_params_by_horizon[h] = {"W": Wh, "ar_lag": ar_lag, "n_components": k, "target_horizon": h}

    # Evaluate
    rows: list[dict] = []
    spatial_acc: dict[int, dict[str, list]] = {h: {"p_sq": [], "m_sq": []} for h in horizons_list}
    pred_frames_h1: list[np.ndarray] = []

    for h in horizons_list:
        if h not in ar_params_by_horizon:
            continue
        Wh = ar_params_by_horizon[h]["W"]
        for t in range(ar_lag, T - h + 1):
            target_t   = t + h - 1
            x_lag      = latent[t - ar_lag: t].reshape(-1).astype(np.float32)
            x_aug      = np.concatenate([[1.0], x_lag])
            pred_lat   = (Wh @ x_aug).astype(np.float32)
            pred_frame = reconstruct_from_latent(pred_lat[np.newaxis], basis, (H, W))[0]
            gt         = frames[target_t]
            persist    = frames[t - 1]
            base       = {"session_name": name, "target_frame": target_t, "horizon": h}
            p_rmse     = float(np.sqrt(np.mean(np.square(persist - gt),    np.float64)))
            m_rmse     = float(np.sqrt(np.mean(np.square(pred_frame - gt), np.float64)))
            rows.append({**base, "model": "persistence", "RMSE": p_rmse})
            rows.append({**base, "model": model_label,   "RMSE": m_rmse})
            spatial_acc[h]["p_sq"].append(np.square(persist.astype(np.float64) - gt.astype(np.float64)))
            spatial_acc[h]["m_sq"].append(np.square(pred_frame.astype(np.float64) - gt.astype(np.float64)))
            if h == 1:
                pred_frames_h1.append(pred_frame.copy())

    trace_df     = pd.DataFrame(rows).reset_index(drop=True)
    spatial_maps = {
        h: {"persistence": np.mean(np.stack(acc["p_sq"]), axis=0).astype(np.float32),
            "model":       np.mean(np.stack(acc["m_sq"]), axis=0).astype(np.float32)}
        for h, acc in spatial_acc.items() if acc["p_sq"]
    }
    pred_frames_h1_arr = (
        np.stack(pred_frames_h1).astype(np.float32)
        if pred_frames_h1
        else np.empty((0, H, W), np.float32)
    )
    return trace_df, spatial_maps, pred_frames_h1_arr


def evaluate_within_session_pca_ar(
    frames: np.ndarray,
    name: str,
    n_components: int,
    ar_lag: int,
    horizons: list[int],
    ridge_lambda: float = 1e-2,
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]], np.ndarray]:
    """
    Fit PCA on the session's own frames, then fit and evaluate AR in that space.

    PCA is fit on all T frames of the session. AR weights are fit once on all
    valid lag windows of the full latent trace.

    Parameters
    ----------
    frames       : np.ndarray, shape (T, H, W)
    name         : str — session identifier
    n_components : int — PCA components
    ar_lag       : int — AR lag order
    horizons     : list of int — forecast horizons
    ridge_lambda : float — ridge regularisation
    """
    from sklearn.decomposition import PCA as _SKLearnPCA

    frames  = np.asarray(frames, np.float32)
    T, H, W = frames.shape
    k        = min(int(n_components), T - 1, H * W)

    pca = _SKLearnPCA(n_components=k, svd_solver="randomized", random_state=0)
    pca.fit(frames.reshape(T, H * W).astype(np.float64))

    basis = {
        "components":               pca.components_.astype(np.float32),
        "mean":                     pca.mean_.astype(np.float32),
        "n_components":             k,
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
    }
    model_label = f"within_session_pca_ar_k{k}_p{ar_lag}"
    return evaluate_within_session_latent_ar(
        frames, name, basis, ar_lag, horizons, model_label, ridge_lambda
    )


def evaluate_within_session_pca_ar_pooled(
    sessions: list[tuple[np.ndarray, str]],
    n_components: int,
    ar_lag: int,
    horizons: list[int],
    ridge_lambda: float = 1e-2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run evaluate_within_session_pca_ar for every session and aggregate."""
    import warnings
    horizons_list = sorted(set(int(h) for h in horizons))
    all_rows: list[dict] = []

    for frames, name in sessions:
        try:
            trace_df, _, _ = evaluate_within_session_pca_ar(
                frames, name, n_components=n_components,
                ar_lag=ar_lag, horizons=horizons_list, ridge_lambda=ridge_lambda,
            )
            for h in horizons_list:
                sub = trace_df[trace_df["horizon"] == h]
                for model, grp in sub.groupby("model"):
                    all_rows.append({
                        "session_name": name,
                        "horizon":      h,
                        "model":        model,
                        "RMSE":         float(grp["RMSE"].mean()),
                    })
        except Exception as e:
            warnings.warn(f"Session {name} failed: {e}")

    per_session_df = pd.DataFrame(all_rows).reset_index(drop=True)
    if per_session_df.empty or "model" not in per_session_df.columns:
        cols = ["model", "horizon", "n_sessions", "RMSE_mean", "RMSE_std", "skill_vs_persistence"]
        return per_session_df, pd.DataFrame(columns=cols)

    agg_rows: list[dict] = []
    for (model, horizon), grp in per_session_df.groupby(["model", "horizon"], sort=False):
        persist_grp = per_session_df[(per_session_df["model"] == "persistence") & (per_session_df["horizon"] == horizon)]
        rmse_vals   = grp["RMSE"].to_numpy(float)
        p_rmse_vals = persist_grp["RMSE"].to_numpy(float)
        pooled_rmse = float(np.mean(rmse_vals))
        pooled_p    = float(np.mean(p_rmse_vals)) if p_rmse_vals.size > 0 else 0.0
        agg_rows.append({"model": model, "horizon": int(horizon),
                         "n_sessions":    int(grp["session_name"].nunique()),
                         "RMSE_mean":     pooled_rmse, "RMSE_std": float(np.std(rmse_vals)),
                         "skill_vs_persistence": skill_vs_persistence(pooled_rmse, pooled_p)})
    return per_session_df, pd.DataFrame(agg_rows).reset_index(drop=True)