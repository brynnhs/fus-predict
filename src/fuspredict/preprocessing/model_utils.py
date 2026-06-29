"""
model_utils.py
--------------
Model fitting, prediction, and evaluation utilities for fUS frame prediction.

All fitting functions are pure — they take numpy arrays and return parameter
dicts. No dataset internals are accessed.

Fitting workflow
----------------
    from fus_predict.model_utils import (
        sample_train_frames_for_pca,
        fit_pca_basis,
        fit_pca_ar_diag,
        evaluate_model_on_dataset,
    )

    # 1. Sample flat frames for PCA from the training dataset
    frames_flat = sample_train_frames_for_pca(train_ds, max_frames=2000, seed=42)

    # 2. Fit PCA basis
    basis = fit_pca_basis(frames_flat, d=10)

    # 3. Load per-session training frames (T, H, W) for AR fitting
    session_frames = [xr.open_dataset(p)["frames"].values for p in train_paths]

    # 4. Fit AR model on PCA latents
    params = fit_pca_ar_diag(session_frames, basis, p=10, ridge_lambda=0.01)

    # 5. Evaluate
    agg = evaluate_model_on_dataset(lambda ctx: predict_pca_ar_diag(ctx, params), test_ds)
"""

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert a tensor or array-like to a numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Dataset iteration
# ---------------------------------------------------------------------------

def iter_windows(dataset, max_items: int | None = None):
    """
    Yield (context, target) from a FUSForecastWindowDataset as numpy arrays.

    Supports dataset items of (x, y) or (x, y, meta).
    """
    n = len(dataset)
    if max_items is not None:
        n = min(n, int(max_items))
    for i in range(n):
        item = dataset[i]
        if not (isinstance(item, (list, tuple)) and len(item) >= 2):
            raise ValueError("Dataset item must be (context, target) or (context, target, meta)")
        yield _to_numpy(item[0]), _to_numpy(item[1])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_frame_metrics(
    y_true,
    y_pred,
    standardize: bool = False,
    eps: float = 1e-8,
    decimals: int = 3,
) -> dict:
    """
    Compute MSE, RMSE, MAE, R² between two arrays (any shape, flattened).

    Parameters
    ----------
    y_true, y_pred : array-like
        Matching shapes; will be squeezed and flattened.
    standardize : bool
        If True, z-score each frame using y_true statistics before computing
        metrics. Useful for comparing across sessions with different scales.
    eps : float
        Minimum std to avoid division by zero when standardizing.
    decimals : int
        Rounding precision for returned values.
    """
    yt = np.asarray(y_true).squeeze().astype(np.float32)
    yp = np.asarray(y_pred).squeeze().astype(np.float32)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if np.issubdtype(yt.dtype, np.integer) or np.issubdtype(yp.dtype, np.integer):
        warnings.warn(
            "compute_frame_metrics received integer inputs; expected float arrays.",
            RuntimeWarning,
        )

    if standardize:
        yt_f = yt.reshape(1, -1) if yt.ndim <= 2 else yt.reshape(yt.shape[0], -1)
        yp_f = yp.reshape(1, -1) if yp.ndim <= 2 else yp.reshape(yp.shape[0], -1)
        mu    = yt_f.mean(axis=1, keepdims=True)
        sigma = yt_f.std(axis=1,  keepdims=True)
        sigma = np.where(sigma < float(eps), 1.0, sigma)
        yt = ((yt_f - mu) / sigma).reshape(-1)
        yp = ((yp_f - mu) / sigma).reshape(-1)
    else:
        yt = yt.reshape(-1)
        yp = yp.reshape(-1)

    err   = yp - yt
    mse   = float(np.mean(err ** 2))
    rmse  = float(np.sqrt(mse))
    mae   = float(np.mean(np.abs(err)))
    denom = float(np.sum((yt - yt.mean()) ** 2))
    r2    = 1.0 - (float(np.sum(err ** 2)) / denom) if denom > 0 else float("nan")

    d = int(decimals)
    return {
        "MSE":  float(np.round(mse,  d)),
        "RMSE": float(np.round(rmse, d)),
        "MAE":  float(np.round(mae,  d)),
        "R2":   float(np.round(r2,   d)),
    }


def evaluate_model_on_dataset(
    predict_fn,
    test_ds,
    max_items: int | None = None,
    return_per_window: bool = False,
    standardize: bool = False,
    decimals: int = 3,
) -> dict | tuple[dict, list[dict]]:
    """
    Evaluate predict_fn on test_ds and aggregate metrics across windows.

    Parameters
    ----------
    predict_fn : callable
        Takes context (numpy array) and returns prediction of the same shape
        as target.
    test_ds : FUSForecastWindowDataset
    max_items : int, optional
        Cap the number of windows evaluated.
    return_per_window : bool
        If True, also return the list of per-window metric dicts.
    standardize : bool
        Passed to compute_frame_metrics.
    decimals : int
        Rounding precision.

    Returns
    -------
    agg : dict
        {metric_mean, metric_std, n_windows} for each metric.
    per_window : list of dict (only if return_per_window=True)
    """
    per_window = []
    for context, target in iter_windows(test_ds, max_items=max_items):
        pred = np.asarray(predict_fn(context))
        if pred.shape != target.shape:
            raise ValueError(f"predict_fn returned {pred.shape}, expected {target.shape}")
        per_window.append(
            compute_frame_metrics(target, pred, standardize=standardize, decimals=decimals)
        )

    if not per_window:
        raise RuntimeError("No windows evaluated.")

    keys = per_window[0].keys()
    agg  = {
        f"{k}_mean": float(np.round(np.mean([m[k] for m in per_window]), decimals))
        for k in keys
    }
    agg.update({
        f"{k}_std": float(np.round(np.std([m[k] for m in per_window]), decimals))
        for k in keys
    })
    agg["n_windows"] = len(per_window)

    if return_per_window:
        return agg, per_window
    return agg


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def residual_acf_latent(residual_latents: np.ndarray, max_lag: int) -> dict:
    """
    Compute ACF for each PCA component and a summary of mean |ACF| by lag.

    Parameters
    ----------
    residual_latents : np.ndarray, shape (T, d)
    max_lag : int

    Returns
    -------
    dict with keys:
        acf               : (d, max_lag+1)
        mean_abs_acf_by_lag : (max_lag+1,)
    """
    x = np.asarray(residual_latents)
    if x.ndim != 2:
        raise ValueError("residual_latents must be (T, d)")
    T, d    = x.shape
    max_lag = int(max_lag)
    if not (0 <= max_lag < T):
        raise ValueError(f"max_lag must be in [0, T-1]; got max_lag={max_lag}, T={T}")

    x_c   = x - x.mean(axis=0, keepdims=True)
    denom = np.sum(x_c ** 2, axis=0)
    denom = np.where(denom == 0, np.nan, denom)

    acf = np.zeros((d, max_lag + 1), dtype=float)
    acf[:, 0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[:, lag] = np.sum(x_c[lag:] * x_c[:-lag], axis=0) / denom

    return {
        "acf":                  acf,
        "mean_abs_acf_by_lag":  np.nanmean(np.abs(acf), axis=0),
    }


def ljung_box_test(residual_series: np.ndarray, lags: list[int]) -> list | None:
    """
    Run the Ljung-Box test on each column of residual_series.

    Requires statsmodels. Returns None if not available.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        warnings.warn("statsmodels not available; skipping Ljung-Box test.")
        return None

    x = np.asarray(residual_series)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return [acorr_ljungbox(x[:, i], lags=lags, return_df=True) for i in range(x.shape[1])]


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------

def predict_persistence(context, K: int = 1) -> np.ndarray:
    """
    Persistence baseline: repeat the last context frame K times.

    Returns shape (K, *frame_shape).
    """
    last = _to_numpy(context)[-1]
    return np.repeat(last[np.newaxis], int(K), axis=0)


# ---------------------------------------------------------------------------
# Per-pixel AR
# ---------------------------------------------------------------------------

def fit_pixel_ar(
    train_ds,
    p: int,
    ridge_lambda: float,
    max_items: int | None = None,
) -> dict:
    """
    Fit per-pixel ridge AR(p) from training windows.

    Parameters
    ----------
    train_ds : FUSForecastWindowDataset
    p : int
        AR lag order.
    ridge_lambda : float
        Ridge regularisation parameter.
    max_items : int, optional
        Maximum number of windows to use.

    Returns
    -------
    dict with keys: A (H, W, p), b (H, W), p
    """
    p            = int(p)
    ridge_lambda = float(ridge_lambda)
    XTX = XTy = None
    H = W = None

    for context, target in iter_windows(train_ds, max_items=max_items):
        if context.shape[0] < p:
            raise ValueError("Context shorter than p.")
        lags   = context[-p:, 0]       # (p, H, W)
        y      = target[0, 0]          # (H, W)
        H, W   = y.shape
        V      = H * W
        x_lags = lags.reshape(p, V)
        y_flat = y.reshape(V)
        x_aug  = np.vstack([np.ones((1, V), dtype=x_lags.dtype), x_lags])  # (p+1, V)

        if XTX is None:
            XTX = np.zeros((V, p + 1, p + 1), dtype=np.float64)
            XTy = np.zeros((V, p + 1),         dtype=np.float64)
        XTX += np.einsum("iv,jv->vij", x_aug, x_aug)
        XTy += np.einsum("iv,v->vi",   x_aug, y_flat)

    if XTX is None:
        raise RuntimeError("No training windows for pixel AR.")

    reg     = np.diag([0.0] + [ridge_lambda] * p)
    W_hat   = np.linalg.solve(XTX + reg[np.newaxis], XTy[..., np.newaxis]).squeeze(-1)
    return {
        "b": W_hat[:, 0].reshape(H, W).astype(np.float32),
        "A": W_hat[:, 1:].reshape(H, W, p).astype(np.float32),
        "p": p,
    }


def predict_pixel_ar(context, params: dict, K: int = 1) -> np.ndarray:
    """
    Predict K frames using fitted per-pixel AR parameters.

    Returns shape (K, 1, H, W).
    """
    x  = _to_numpy(context)
    p  = int(params["p"])
    A  = params["A"]
    b  = params["b"]
    if x.shape[0] < p:
        raise ValueError("Context shorter than p.")

    history = list(np.asarray(x[-p:, 0], dtype=np.float32))
    preds   = []
    for _ in range(int(K)):
        lags = np.stack(history[-p:], axis=0)
        pred = b.astype(np.float32, copy=True)
        for i in range(p):
            pred += A[:, :, i] * lags[i]
        preds.append(pred)
        history.append(pred)
    return np.stack(preds, axis=0)[:, np.newaxis]


# ---------------------------------------------------------------------------
# PCA basis fitting
# ---------------------------------------------------------------------------

def sample_train_frames_for_pca(
    train_ds,
    max_frames: int,
    seed: int,
) -> np.ndarray:
    """
    Sample up to max_frames flattened frames from training windows.

    Returns
    -------
    np.ndarray, shape (N, H*W)
    """
    rng       = np.random.default_rng(int(seed))
    max_frames = int(max_frames)
    frames    = []

    for context, target in iter_windows(train_ds):
        if len(frames) >= max_frames:
            break
        all_frames = np.concatenate([context, target], axis=0)  # (W+K, 1, H, W)
        T          = all_frames.shape[0]
        remaining  = max_frames - len(frames)
        pick       = np.arange(T) if T <= remaining else rng.choice(T, size=remaining, replace=False)
        for idx in pick:
            frames.append(all_frames[idx, 0].reshape(-1))

    if not frames:
        raise RuntimeError("No frames sampled for PCA.")
    return np.stack(frames, axis=0)


def _fit_pca_sklearn(frames_flat: np.ndarray, d: int):
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("scikit-learn is required for PCA baselines.") from e
    pca = PCA(n_components=int(d), svd_solver="randomized")
    pca.fit(frames_flat)
    return pca.mean_, pca.components_


def _fit_pca_torch(frames_flat: np.ndarray, d: int, device: str):
    import torch
    x          = torch.as_tensor(frames_flat, device=device, dtype=torch.float32)
    mean       = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    _, _, v    = torch.pca_lowrank(x_centered, q=int(d), center=False)
    return mean.squeeze(0).cpu().numpy(), v.T.contiguous().cpu().numpy()


def fit_pca_basis(
    frames_flat: np.ndarray,
    d: int,
    use_torch: bool = True,
    device: str = "cuda",
) -> dict:
    """
    Fit a PCA basis on a (N, V) frame matrix.

    Parameters
    ----------
    frames_flat : np.ndarray, shape (N, H*W)
        Flattened training frames, e.g. from sample_train_frames_for_pca.
    d : int
        Number of PCA components.
    use_torch : bool
        Use torch.pca_lowrank (faster on GPU). Falls back to sklearn on failure.
    device : str
        PyTorch device string.

    Returns
    -------
    dict with keys: d, mean (V,), components (d, V)
    """
    import torch

    mean = components = None

    if use_torch:
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        try:
            mean, components = _fit_pca_torch(frames_flat, d, device=device)
        except RuntimeError as e:
            warnings.warn(f"Torch PCA failed: {e}. Falling back to sklearn.")
            use_torch = False

    if not use_torch or mean is None:
        mean, components = _fit_pca_sklearn(frames_flat, d)

    return {
        "d":          int(d),
        "mean":       np.asarray(mean,       dtype=np.float32),
        "components": np.asarray(components, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# PCA-VAR model
# ---------------------------------------------------------------------------

def fit_pca_var(
    session_frames: list[np.ndarray],
    basis: dict,
    p: int,
    ridge_lambda: float,
    max_samples: int | None = None,
) -> dict:
    """
    Fit a full VAR(p) on PCA latents, accumulating statistics across sessions.

    Parameters
    ----------
    session_frames : list of np.ndarray, each shape (T, H, W)
        Per-session training frames (training split only).
    basis : dict
        Output of fit_pca_basis.
    p : int
        VAR lag order.
    ridge_lambda : float
        Ridge regularisation.
    max_samples : int, optional
        Stop after accumulating this many target samples.

    Returns
    -------
    dict with keys: p, d, mean, components, var_weights
    """
    p            = int(p)
    ridge_lambda = float(ridge_lambda)
    mean         = np.asarray(basis["mean"],       dtype=np.float32)
    components   = np.asarray(basis["components"], dtype=np.float32)

    XTX = XTy = None
    n_used = 0

    for frames in session_frames:
        arr  = np.asarray(frames, dtype=np.float32)
        T    = arr.shape[0]
        if T <= p:
            continue
        Z  = (arr.reshape(T, -1) - mean) @ components.T  # (T, d)
        Zt = Z[p:]
        X  = np.concatenate([Z[p - i - 1: -i - 1] for i in range(p)], axis=1)
        X  = np.concatenate([np.ones((X.shape[0], 1), dtype=X.dtype), X], axis=1)
        XTX = X.T @ X if XTX is None else XTX + X.T @ X
        XTy = X.T @ Zt if XTy is None else XTy + X.T @ Zt
        n_used += Zt.shape[0]
        if max_samples is not None and n_used >= int(max_samples):
            break

    if XTX is None:
        raise RuntimeError("No training sequences for PCA-VAR.")

    reg = np.diag([0.0] + [ridge_lambda] * (XTX.shape[0] - 1))
    W   = np.linalg.solve(XTX + reg, XTy).astype(np.float32)

    return {"p": p, "d": int(basis["d"]), "mean": mean, "components": components, "var_weights": W}


def predict_pca_var(context, params: dict, K: int = 1) -> np.ndarray:
    """
    Predict K frames using PCA + full VAR parameters.

    Returns shape (K, 1, H, W).
    """
    x          = _to_numpy(context)
    p          = int(params["p"])
    mean       = params["mean"]
    components = params["components"]
    W          = params["var_weights"]
    H, Wd      = x.shape[-2], x.shape[-1]

    if x.shape[0] < p:
        raise ValueError("Context shorter than p.")

    history = list(np.asarray(x[-p:, 0], dtype=np.float32))
    preds   = []
    for _ in range(int(K)):
        ctx        = np.stack(history[-p:]).reshape(p, -1)
        latents    = (ctx - mean) @ components.T
        x_feat     = latents.reshape(-1)
        x_aug      = np.concatenate([np.ones(1, dtype=x_feat.dtype), x_feat])
        pred_flat  = (x_aug @ W) @ components + mean
        pred       = pred_flat.reshape(H, Wd).astype(np.float32)
        preds.append(pred)
        history.append(pred)
    return np.stack(preds)[:, np.newaxis]


# ---------------------------------------------------------------------------
# PCA-AR (diagonal / per-component) model
# ---------------------------------------------------------------------------

def fit_pca_ar_diag(
    session_frames: list[np.ndarray],
    basis: dict,
    p: int,
    ridge_lambda: float,
    max_samples: int | None = None,
) -> dict:
    """
    Fit independent AR(p) per PCA component, accumulating across sessions.

    Parameters
    ----------
    session_frames : list of np.ndarray, each shape (T, H, W)
        Per-session training frames (training split only).
    basis : dict
        Output of fit_pca_basis.
    p : int
        AR lag order.
    ridge_lambda : float
        Ridge regularisation.
    max_samples : int, optional
        Stop after accumulating this many target samples.

    Returns
    -------
    dict with keys: p, d, mean, components, ar_weights (d, p+1)
    """
    p            = int(p)
    ridge_lambda = float(ridge_lambda)
    mean         = np.asarray(basis["mean"],       dtype=np.float32)
    components   = np.asarray(basis["components"], dtype=np.float32)

    XTX = XTy = None
    n_used = 0

    for frames in session_frames:
        arr  = np.asarray(frames, dtype=np.float32)
        T    = arr.shape[0]
        if T <= p:
            continue
        Z     = (arr.reshape(T, -1) - mean) @ components.T  # (T, d)
        Zt    = Z[p:]
        lags  = [Z[p - i - 1: -i - 1] for i in range(p)]
        L     = np.stack(lags, axis=1)                       # (T-p, p, d)
        ones  = np.ones((L.shape[0], 1, L.shape[2]), dtype=L.dtype)
        X_aug = np.concatenate([ones, L], axis=1)            # (T-p, p+1, d)
        batch_XTX = np.einsum("tfd,tgd->dfg", X_aug, X_aug)
        batch_XTy = np.einsum("tfd,td->df",   X_aug, Zt)
        XTX = batch_XTX if XTX is None else XTX + batch_XTX
        XTy = batch_XTy if XTy is None else XTy + batch_XTy
        n_used += Zt.shape[0]
        if max_samples is not None and n_used >= int(max_samples):
            break

    if XTX is None:
        raise RuntimeError("No training sequences for PCA-AR-diag.")

    reg = np.diag([0.0] + [ridge_lambda] * p)
    W   = np.zeros((XTX.shape[0], p + 1), dtype=np.float32)
    for j in range(XTX.shape[0]):
        W[j] = np.linalg.solve(XTX[j] + reg, XTy[j]).astype(np.float32)

    return {"p": p, "d": int(basis["d"]), "mean": mean, "components": components, "ar_weights": W}


def predict_pca_ar_diag(context, params: dict, K: int = 1) -> np.ndarray:
    """
    Predict K frames using PCA + per-component AR parameters.

    Returns shape (K, 1, H, W).
    """
    x          = _to_numpy(context)
    p          = int(params["p"])
    mean       = params["mean"]
    components = params["components"]
    W          = params["ar_weights"]     # (d, p+1)
    H, Wd      = x.shape[-2], x.shape[-1]

    if x.shape[0] < p:
        raise ValueError("Context shorter than p.")

    history = list(np.asarray(x[-p:, 0], dtype=np.float32))
    preds   = []
    for _ in range(int(K)):
        ctx     = np.stack(history[-p:]).reshape(p, -1)
        latents = (ctx - mean) @ components.T   # (p, d)
        z_pred  = W[:, 0].copy()
        for i in range(p):
            z_pred += W[:, i + 1] * latents[-(i + 1)]
        pred = (z_pred @ components + mean).reshape(H, Wd).astype(np.float32)
        preds.append(pred)
        history.append(pred)
    return np.stack(preds)[:, np.newaxis]