"""
autocorrelation.py
------------------
Autocorrelation and spatial correlation analysis utilities for fUS data.

All functions are pure — no global state, no file I/O, no side effects.
Inputs and outputs are numpy arrays.

Author: Brynn Harris-Shanks, 2026
"""

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

def spatial_neighbor_offsets(neighborhood: int) -> list[tuple[int, int]]:
    """
    Return (dy, dx) offsets for a 4- or 8-connected neighbourhood.

    Parameters
    ----------
    neighborhood : int
        4 for cardinal neighbours, 8 for all neighbours including diagonals.
    """
    if neighborhood == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighborhood == 8:
        return [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    raise ValueError(f"Unsupported neighborhood={neighborhood!r}; expected 4 or 8")


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


def seed_patch_mask(
    shape_hw: tuple[int, int],
    seed_center_yx: tuple[int, int],
    seed_radius: int = 0,
) -> np.ndarray:
    """
    Build a boolean (H, W) mask for a square seed patch.

    Parameters
    ----------
    shape_hw : (H, W)
    seed_center_yx : (y, x) pixel index of the patch centre.
    seed_radius : int
        Half-width of the patch. 0 means a single pixel.

    Returns
    -------
    np.ndarray, shape (H, W), dtype bool
    """
    H, W = int(shape_hw[0]), int(shape_hw[1])
    y, x = int(seed_center_yx[0]), int(seed_center_yx[1])
    r    = int(seed_radius)

    if H <= 0 or W <= 0:
        raise ValueError(f"shape_hw must be positive, got {shape_hw}")
    if r < 0:
        raise ValueError(f"seed_radius must be >= 0, got {seed_radius}")
    if not (0 <= y < H and 0 <= x < W):
        raise ValueError(f"Seed centre {(y, x)} out of bounds for shape {(H, W)}")

    mask = np.zeros((H, W), dtype=bool)
    mask[max(0, y - r): min(H, y + r + 1),
         max(0, x - r): min(W, x + r + 1)] = True

    if not np.any(mask):
        raise ValueError(f"Seed patch is empty for centre {(y, x)}, radius={r}, shape={(H, W)}")

    return mask


def seed_temporal_corr_map(
    frames_3d: np.ndarray,
    seed_center_yx: tuple[int, int],
    seed_radius: int = 0,
    eps: float = 1e-8,
    min_samples: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Correlate every pixel's time series with a seed-patch mean time series.

    Parameters
    ----------
    frames_3d : np.ndarray, shape (T, H, W)
    seed_center_yx : (y, x) centre of the seed patch.
    seed_radius : int
        Half-width of the seed patch (0 = single pixel).
    eps, min_samples : forwarded to safe_temporal_corr_map.

    Returns
    -------
    corr_map : np.ndarray, shape (H, W) — Pearson r with seed trace
    n_valid  : np.ndarray, shape (H, W) — finite sample counts
    seed_ts  : np.ndarray, shape (T,)   — seed patch mean time series
    seed_mask: np.ndarray, shape (H, W) — boolean seed region mask
    """
    arr = np.asarray(frames_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected (T, H, W), got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"Not enough frames: T={arr.shape[0]}")

    T, H, W   = arr.shape
    seed_mask = seed_patch_mask((H, W), seed_center_yx, seed_radius=seed_radius)
    seed_ts   = np.nanmean(arr[:, seed_mask], axis=1).astype(np.float64)

    n_finite = int(np.sum(np.isfinite(seed_ts)))
    if n_finite < max(2, min_samples):
        raise ValueError(
            f"Seed trace has insufficient finite samples: {n_finite} (min {max(2, min_samples)})"
        )

    seed_var = float(np.var(seed_ts[np.isfinite(seed_ts)]))
    if not np.isfinite(seed_var) or seed_var <= eps:
        raise ValueError(f"Seed trace is near-constant (variance={seed_var})")

    seed_3d           = np.broadcast_to(seed_ts.reshape(T, 1, 1), arr.shape)
    corr_map, n_valid = safe_temporal_corr_map(arr, seed_3d, eps=eps, min_samples=min_samples)

    return corr_map, n_valid, seed_ts, seed_mask


def spatial_autocorr_map(
    frames_3d: np.ndarray,
    neighborhood: int = 4,
    eps: float = 1e-8,
    min_samples: int = 8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Per-pixel mean Pearson correlation with spatial neighbours.

    Parameters
    ----------
    frames_3d : np.ndarray, shape (T, H, W)
    neighborhood : int
        4 (cardinal) or 8 (all neighbours including diagonals).
    eps, min_samples : forwarded to safe_temporal_corr_map.

    Returns
    -------
    sac_map     : np.ndarray, shape (H, W) — mean neighbour correlation
    count_map   : np.ndarray, shape (H, W), int32 — number of valid neighbours
    global_mean : float — nanmean of sac_map
    """
    arr = np.asarray(frames_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected (T, H, W), got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"Not enough frames: T={arr.shape[0]}")

    T, H, W   = arr.shape
    sum_map   = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.int32)

    for dy, dx in spatial_neighbor_offsets(neighborhood):
        y0, y1 = max(0, -dy), min(H, H - dy)
        x0, x1 = max(0, -dx), min(W, W - dx)

        if y1 <= y0 or x1 <= x0:
            continue

        A = arr[:, y0:y1, x0:x1]
        B = arr[:, y0 + dy: y1 + dy, x0 + dx: x1 + dx]

        corr_local, _ = safe_temporal_corr_map(A, B, eps=eps, min_samples=min_samples)
        finite_local  = np.isfinite(corr_local)

        if not np.any(finite_local):
            continue

        sum_map[y0:y1, x0:x1][finite_local]   += corr_local[finite_local]
        count_map[y0:y1, x0:x1][finite_local] += 1

    sac_map         = np.full((H, W), np.nan, dtype=np.float64)
    valid           = count_map > 0
    sac_map[valid]  = sum_map[valid] / count_map[valid].astype(np.float64)
    sac_map[np.isfinite(sac_map)] = np.clip(sac_map[np.isfinite(sac_map)], -1.0, 1.0)

    global_mean = float(np.nanmean(sac_map)) if np.any(np.isfinite(sac_map)) else np.nan

    return sac_map, count_map, global_mean


def spatial_limits(map_: np.ndarray) -> tuple[float, float]:
    """Return fixed display limits for a correlation map."""
    return (-1.0, 1.0)