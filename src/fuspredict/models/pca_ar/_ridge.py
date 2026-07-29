"""
_ridge.py
---------
Shared ridge-regularized direct-horizon AR fitting helpers used by the
frozen-basis predictors.
"""

from __future__ import annotations

import numpy as np


def _fit_ridge_ar(
    Z: np.ndarray,
    horizon: int,
    lag: int,
    ridge_lambda: float,
) -> dict:
    """
    Fit ridge-regularized direct-horizon AR weights on a latent time series.

    Parameters
    ----------
    Z : np.ndarray, shape (T, C)
        Latent (e.g. PCA-projected) time series for one session.
    horizon : int
        Prediction horizon.
    lag : int
        Number of autoregressive lags, flattened across components.
    ridge_lambda : float
        L2 regularization strength (bias excluded).

    Returns
    -------
    dict with keys "XtX" and "XtY" : np.ndarray
        Partial accumulators to be summed across sessions before solving.
    """
    T, C = Z.shape
    n_samples = T - lag - horizon + 1
    dim = lag * C + 1
    if n_samples <= 0:
        return {
            "XtX": np.zeros((dim, dim), dtype=np.float64),
            "XtY": np.zeros((dim, C), dtype=np.float64),
        }

    X = np.stack(
        [Z[lag - 1 - k : lag - 1 - k + n_samples] for k in range(lag)],
        axis=1,
    )  # (N, lag, C)
    X = X.reshape(n_samples, lag * C)
    ones = np.ones((n_samples, 1), dtype=np.float64)
    Xb = np.concatenate([X, ones], axis=1)  # (N, dim)

    Y = Z[lag - 1 + horizon : lag - 1 + horizon + n_samples]  # (N, C)

    XtX = Xb.T @ Xb
    XtY = Xb.T @ Y
    return {"XtX": XtX, "XtY": XtY}


def _solve_ridge_ar(
    XtX: np.ndarray,
    XtY: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the regularized normal equations for AR weights.

    Parameters
    ----------
    XtX : np.ndarray, shape (dim, dim)
        Accumulated design Gram matrix.
    XtY : np.ndarray, shape (dim, C)
        Accumulated design/target cross product.
    ridge_lambda : float
        L2 regularization strength (bias excluded).

    Returns
    -------
    W : np.ndarray, shape (dim - 1, C)
        AR weight matrix (excluding bias row).
    bias : np.ndarray, shape (C,)
        Bias term.
    """
    dim = XtX.shape[0]
    reg = ridge_lambda * np.eye(dim, dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(XtX + reg, XtY)  # (dim, C)
    return weights[:-1], weights[-1]
