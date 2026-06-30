"""
autocorrelation.py
------------------
Autocorrelation and spatial correlation analysis utilities for fUS data.

All functions are pure — no global state, no file I/O, no side effects.
Inputs and outputs are numpy arrays.

Author: Brynn Harris-Shanks, 2026
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def finite_values(a: np.ndarray) -> np.ndarray:
    """Return a flat array of all finite values in a."""
    x = np.asarray(a)
    return x[np.isfinite(x)]


def robust_limits(
    a: np.ndarray,
    pctl: tuple[float, float] = (2.0, 98.0),
    symmetric: bool = False,
    nonnegative: bool = False,
    eps: float = 1e-8,
) -> tuple[float, float]:
    """
    Compute robust display limits from percentiles of finite values.

    Parameters
    ----------
    a : array-like
        Input array (any shape).
    pctl : (lo_pct, hi_pct)
        Percentile pair for lower and upper bounds.
    symmetric : bool
        If True, return (-m, m) where m = max(|lo|, |hi|).
    nonnegative : bool
        If True, clamp lo to 0.
    eps : float
        Minimum range — if hi <= lo + eps, hi is set to lo + 1.

    Returns
    -------
    (lo, hi) : tuple of float
    """
    x = finite_values(a)
    if x.size == 0:
        return (0.0, 1.0)

    lo = float(np.percentile(x, pctl[0]))
    hi = float(np.percentile(x, pctl[1]))

    if symmetric:
        m = max(abs(lo), abs(hi), eps)
        return (-m, m)
    if nonnegative:
        lo = max(0.0, lo)
    if hi <= lo + eps:
        hi = lo + 1.0

    return (lo, hi)


def session_global_signal(frames_3d: np.ndarray) -> np.ndarray | None:
    """
    Compute the mean-trace global signal from (T, H, W) frames.

    Returns the mean-centered finite trace, or None if fewer than 2
    finite frames exist or the signal has near-zero variance.
    """
    arr = np.asarray(frames_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected (T, H, W), got shape {arr.shape}")

    x      = np.nanmean(arr, axis=(1, 2)).astype(np.float64)
    finite = np.isfinite(x)
    x      = x[finite]

    if x.size < 2:
        return None

    x = x - np.mean(x)
    if not np.isfinite(np.var(x)) or np.var(x) < 1e-12:
        return None

    return x


# ---------------------------------------------------------------------------
# Temporal autocorrelation
# ---------------------------------------------------------------------------

def standardized_acf(
    x: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Compute the standardized (lag-0 = 1) autocorrelation function of x.

    Parameters
    ----------
    x : np.ndarray, 1D
        Mean-centered time series.
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    lags : np.ndarray of int, shape (L+1,)
    acf  : np.ndarray of float, shape (L+1,)
    or (None, None) if computation is not possible.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return None, None

    corr_full = np.correlate(x, x, mode="full")
    mid       = corr_full.size // 2
    corr      = corr_full[mid:]

    if corr[0] == 0 or not np.isfinite(corr[0]):
        return None, None

    L    = int(min(max_lag, x.size - 1))
    lags = np.arange(L + 1, dtype=np.int32)
    acf  = corr[: L + 1] / corr[0]

    return lags, acf


def safe_standardized_acf(
    x: np.ndarray,
    max_lag: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Mean-center and variance-check x, then compute its standardized ACF.

    Drops non-finite values before computing. Returns (None, None) if
    fewer than 2 finite values remain or variance is near zero.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size < 2:
        return None, None

    x   = x - np.mean(x)
    var = np.var(x)

    if not np.isfinite(var) or var < float(eps):
        return None, None

    return standardized_acf(x, max_lag)


# ---------------------------------------------------------------------------
# Spatial correlation
# ---------------------------------------------------------------------------

def safe_temporal_corr_map(
    A: np.ndarray,
    B: np.ndarray,
    eps: float = 1e-8,
    min_samples: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel Pearson correlation between two (T, H, W) arrays.

    Missing values (NaN/Inf) are excluded per pixel. Pixels with fewer than
    min_samples finite pairs are returned as NaN.

    Parameters
    ----------
    A, B : np.ndarray, shape (T, H, W)
    eps : float
        Minimum denominator to avoid division by near-zero.
    min_samples : int
        Minimum number of finite frame pairs required per pixel.

    Returns
    -------
    corr    : np.ndarray, shape (H, W), float64 — Pearson r, NaN where invalid
    n_valid : np.ndarray, shape (H, W), int32  — number of finite pairs used
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.shape != B.shape:
        raise ValueError(f"A and B must have identical shape; got {A.shape} vs {B.shape}")
    if A.ndim != 3:
        raise ValueError(f"Expected (T, H, W), got shape {A.shape}")

    finite  = np.isfinite(A) & np.isfinite(B)
    n_valid = finite.sum(axis=0).astype(np.int32)

    A0 = np.where(finite, A, 0.0)
    B0 = np.where(finite, B, 0.0)

    mean_a = np.divide(A0.sum(axis=0), n_valid, out=np.zeros(n_valid.shape), where=n_valid > 0)
    mean_b = np.divide(B0.sum(axis=0), n_valid, out=np.zeros(n_valid.shape), where=n_valid > 0)

    Ac = np.where(finite, A - mean_a[None], 0.0)
    Bc = np.where(finite, B - mean_b[None], 0.0)

    cov   = np.sum(Ac * Bc, axis=0)
    denom = np.sqrt(np.sum(Ac * Ac, axis=0) * np.sum(Bc * Bc, axis=0))

    corr = np.full(n_valid.shape, np.nan, dtype=np.float64)
    ok   = (n_valid >= min_samples) & np.isfinite(denom) & (denom > eps)
    corr[ok] = np.clip(cov[ok] / denom[ok], -1.0, 1.0)

    return corr, n_valid


