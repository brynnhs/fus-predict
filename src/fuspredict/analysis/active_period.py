"""
active_period.py
----------------
Pure computation functions for active-period analysis of fUS data.

All functions are stateless and side-effect-free: no I/O, no plotting.
Inputs and outputs are numpy arrays or plain Python scalars/dicts.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage


# ---------------------------------------------------------------------------
# Z-score standardization
# ---------------------------------------------------------------------------

_ZSCORE_EPS = 1e-8


def zscore_frames(
    frames_log10: np.ndarray,
    baseline_mean_map: np.ndarray,
    baseline_std_map: np.ndarray,
) -> np.ndarray:
    """Z-score task frames using per-pixel baseline mean and std.

    Parameters
    ----------
    frames_log10 : np.ndarray, shape (T, H, W)
        Log10-transformed Power Doppler frames.
    baseline_mean_map : np.ndarray, shape (H, W)
        Per-pixel baseline log10 mean (from standardized .nc mean_map).
    baseline_std_map : np.ndarray, shape (H, W)
        Per-pixel baseline log10 std (from standardized .nc std_map).

    Returns
    -------
    np.ndarray, shape (T, H, W), float32
        Frames in z-score units relative to baseline.
    """
    arr   = frames_log10.astype(np.float32)
    mean  = baseline_mean_map.astype(np.float32)
    scale = np.where(baseline_std_map > _ZSCORE_EPS, baseline_std_map, _ZSCORE_EPS).astype(np.float32)
    return (arr - mean[None]) / scale[None]


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

def activation_delta(
    baseline_mean_map: np.ndarray,
    task_frames_log10: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Compute a spatial activation delta map for ROI selection.

    Delta = mean(first ``window`` task frames) - baseline_mean_map,
    both in log10 space.

    Parameters
    ----------
    baseline_mean_map : np.ndarray, shape (H, W)
        Per-pixel baseline log10 mean.
    task_frames_log10 : np.ndarray, shape (T, H, W)
        Log10 task frames in acquisition order.
    window : int
        Number of task frames averaged to form the post-onset mean.

    Returns
    -------
    np.ndarray, shape (H, W), float32
    """
    post = task_frames_log10[: min(window, task_frames_log10.shape[0])].mean(axis=0)
    return (post - baseline_mean_map).astype(np.float32)


def auto_roi(delta_map: np.ndarray, n_pixels: int = 125) -> np.ndarray:
    """Select a spatially compact ROI around the peak activation.

    Starts from the top-n_pixels seed, keeps the largest connected component,
    then grows outward (4-connectivity) until ``n_pixels`` is reached.

    Parameters
    ----------
    delta_map : np.ndarray, shape (H, W)
    n_pixels : int
        Target ROI size in pixels.

    Returns
    -------
    np.ndarray, shape (H, W), bool
    """
    flat = delta_map.ravel()
    positive = flat[flat > 0]
    if len(positive) == 0:
        raise ValueError("No positive delta pixels — cannot auto-select ROI.")

    n_seed    = min(n_pixels, len(positive))
    threshold = float(np.sort(positive)[-n_seed])
    seed_mask = delta_map >= threshold

    labeled, _ = scipy.ndimage.label(seed_mask)
    sizes      = np.bincount(labeled.ravel())
    sizes[0]   = 0
    mask       = (labeled == int(sizes.argmax()))

    H, W      = delta_map.shape
    candidates = np.argsort(flat)[::-1]
    for idx in candidates:
        if mask.sum() >= n_pixels:
            break
        r, c = divmod(int(idx), W)
        if any(
            0 <= nr < H and 0 <= nc < W and mask[nr, nc]
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        ):
            mask[r, c] = True

    return mask


# ---------------------------------------------------------------------------
# ROI signal and baseline statistics
# ---------------------------------------------------------------------------

def roi_signal(frames_z: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Per-frame mean z-score over ROI pixels.

    Parameters
    ----------
    frames_z : np.ndarray, shape (T, H, W)
    roi_mask : np.ndarray, shape (H, W), bool

    Returns
    -------
    np.ndarray, shape (T,), float32
    """
    flat = frames_z.reshape(frames_z.shape[0], -1)
    return flat[:, roi_mask.ravel()].mean(axis=1).astype(np.float32)


def baseline_roi_stats(
    baseline_mean_map: np.ndarray,
    baseline_std_map: np.ndarray,
    roi_mask: np.ndarray,
    baseline_frames_z: np.ndarray | None = None,
) -> tuple[float, float]:
    """Empirical mean and std of the ROI-averaged z-score signal over baseline frames.

    Averaging over ROI pixels reduces variance by ~1/sqrt(N_roi), so the
    effective std of the ROI-mean signal is much less than 1.0. This function
    computes it empirically from the actual baseline z-scored frames so that
    sigma_crossings thresholds reflect real baseline variability.

    Parameters
    ----------
    baseline_mean_map : np.ndarray, shape (H, W)
        Unused; kept for API compatibility.
    baseline_std_map : np.ndarray, shape (H, W)
        Unused; kept for API compatibility.
    roi_mask : np.ndarray, shape (H, W), bool
    baseline_frames_z : np.ndarray, shape (T, H, W), optional
        Z-scored baseline frames. If provided, mean and std are computed
        empirically from the ROI-averaged signal over these frames.
        If None, falls back to (0.0, 1/sqrt(roi_mask.sum())).

    Returns
    -------
    (mean, std) : (float, float)
        Empirical baseline mean and std of the ROI-averaged z-score signal.
    """
    if baseline_frames_z is not None and baseline_frames_z.shape[0] > 1:
        bl_signal = roi_signal(baseline_frames_z, roi_mask)
        return float(bl_signal.mean()), float(bl_signal.std())
    # Theoretical fallback: ROI averaging reduces std by 1/sqrt(N)
    n_roi = max(int(roi_mask.sum()), 1)
    return 0.0, 1.0 / float(np.sqrt(n_roi))


# ---------------------------------------------------------------------------
# σ-crossing detection
# ---------------------------------------------------------------------------

def _first_sustained_crossing(signal: np.ndarray, threshold: float, n_consec: int) -> int | None:
    above = signal > threshold
    for i in range(len(above) - n_consec + 1):
        if above[i : i + n_consec].all():
            return i
    return None


def sigma_crossings(
    signal: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    fps: float,
    n_consec: int = 2,
) -> dict[int, float | None]:
    """First sustained crossing of +1σ, +2σ, +3σ above baseline mean.

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        ROI-averaged % CBV time series starting at onset (frame 0 = t0).
    baseline_mean : float
    baseline_std : float
    fps : float
    n_consec : int
        Minimum number of consecutive frames above threshold to count.

    Returns
    -------
    dict mapping n -> seconds after onset, or None if never crossed.
    """
    results: dict[int, float | None] = {}
    for n in (1, 2, 3):
        idx = _first_sustained_crossing(signal, baseline_mean + n * baseline_std, n_consec)
        results[n] = idx / fps if idx is not None else None
    return results
