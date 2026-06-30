"""
autoregressive.py
-----------------
Autoregressive model fitting, prediction, and evaluation for fUS frame forecasting.

All functions are pure — they take numpy arrays and return results.
No file I/O. No path loading. The caller loads frames (e.g. via xarray) and
passes them in directly.
"""

from time import perf_counter

import numpy as np
import pandas as pd


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


# ---------------------------------------------------------------------------
# Section 1: Direct pixel AR
# ---------------------------------------------------------------------------

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
# PCA helpers (shared across Section 3)
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

