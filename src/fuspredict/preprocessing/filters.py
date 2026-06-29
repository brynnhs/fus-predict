from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from .io import STAGE_FILTERED, derive_session_id_from_path, sanitize_attrs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_frames_thw(frames: np.ndarray, ctx: str) -> tuple[int, int, int]:
    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError(f"{ctx}: expected shape (T, H, W), got {arr.shape}")
    t, h, w = arr.shape
    return int(t), int(h), int(w)


# ---------------------------------------------------------------------------
# Filter primitives
# ---------------------------------------------------------------------------

def low_pass_filter(
    frames: np.ndarray,
    fps: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Temporal low-pass filter over axis=0 for (T, H, W) frames."""
    from scipy import signal

    t, _, _ = _validate_frames_thw(frames, ctx="low_pass_filter")
    arr = np.asarray(frames, dtype=np.float32)

    if float(fps) <= 0.0:
        raise ValueError(f"fps must be > 0, got {fps!r}")
    if float(cutoff_hz) <= 0.0:
        raise ValueError(f"cutoff_hz must be > 0, got {cutoff_hz!r}")
    if int(order) < 1:
        raise ValueError(f"order must be >= 1, got {order!r}")

    nyquist = float(fps) / 2.0
    wn = min(float(cutoff_hz) / nyquist, 0.9999)
    if not (0.0 < wn < 1.0):
        raise ValueError(
            f"cutoff_hz must satisfy 0 < cutoff_hz < nyquist ({nyquist}), got {cutoff_hz!r}"
        )

    b, a = signal.butter(int(order), wn, btype="low", analog=False)
    padlen = 3 * max(len(a), len(b))

    if t <= padlen:
        warnings.warn(
            f"low_pass_filter: T={t} too short for filtfilt (padlen={padlen}); "
            f"falling back to lfilter.",
            stacklevel=2,
        )
        return np.asarray(signal.lfilter(b, a, arr, axis=0), dtype=np.float32)

    try:
        out = signal.filtfilt(b, a, arr, axis=0)
    except ValueError as exc:
        warnings.warn(
            f"low_pass_filter: filtfilt failed ({exc}); falling back to lfilter.",
            stacklevel=2,
        )
        out = signal.lfilter(b, a, arr, axis=0)

    return np.asarray(out, dtype=np.float32)


def high_pass_filter(
    frames: np.ndarray,
    fps: float,
    cutoff_hz: float,
    order: int = 3,
) -> np.ndarray:
    """Temporal high-pass filter over axis=0 for (T, H, W) frames."""
    from scipy import signal

    t, _, _ = _validate_frames_thw(frames, ctx="high_pass_filter")
    arr = np.asarray(frames, dtype=np.float32)

    if float(fps) <= 0.0:
        raise ValueError(f"fps must be > 0, got {fps!r}")
    if float(cutoff_hz) <= 0.0:
        raise ValueError(f"cutoff_hz must be > 0, got {cutoff_hz!r}")
    if int(order) < 1:
        raise ValueError(f"order must be >= 1, got {order!r}")

    nyquist = float(fps) / 2.0
    wn = float(cutoff_hz) / nyquist
    if not (0.0 < wn < 1.0):
        raise ValueError(
            f"cutoff_hz must satisfy 0 < cutoff_hz < nyquist ({nyquist}), got {cutoff_hz!r}"
        )

    b, a = signal.butter(int(order), wn, btype="high", analog=False)
    padlen = 3 * max(len(a), len(b))

    if t <= padlen:
        warnings.warn(
            f"high_pass_filter: T={t} too short for filtfilt (padlen={padlen}); "
            f"falling back to lfilter.",
            stacklevel=2,
        )
        return np.asarray(signal.lfilter(b, a, arr, axis=0), dtype=np.float32)

    try:
        out = signal.filtfilt(b, a, arr, axis=0)
    except ValueError as exc:
        warnings.warn(
            f"high_pass_filter: filtfilt failed ({exc}); falling back to lfilter.",
            stacklevel=2,
        )
        out = signal.lfilter(b, a, arr, axis=0)

    return np.asarray(out, dtype=np.float32)


def percentile_clip(
    frames: np.ndarray,
    bottom: float = 1.0,
    top: float = 99.0,
) -> np.ndarray:
    """Global percentile clipping over all values in (T, H, W)."""
    _validate_frames_thw(frames, ctx="percentile_clip")
    arr = np.asarray(frames, dtype=np.float32)

    if not (0.0 <= bottom < top <= 100.0):
        raise ValueError(f"Expected 0 <= bottom < top <= 100, got {bottom}, {top}")

    if arr.size == 0:
        return arr

    lo = float(np.percentile(arr, bottom))
    hi = float(np.percentile(arr, top))
    return np.asarray(np.clip(arr, lo, hi), dtype=np.float32)


# ---------------------------------------------------------------------------
# Composed filter stage
# ---------------------------------------------------------------------------

def apply_optional_filters(
    da: xr.DataArray,
    *,
    enable_lowpass: bool = False,
    lowpass_cutoff_hz: float = 0.5,
    lowpass_order: int = 4,
    enable_highpass: bool = False,
    highpass_cutoff_hz: float = 0.0,
    highpass_order: int = 3,
    enable_clip: bool = False,
    clip_bottom: float = 1.0,
    clip_top: float = 99.0,
    fps_fallback: float | None = None,
) -> xr.DataArray:
    """
    Apply enabled filters in fixed order: lowpass → highpass → clip.

    Parameters
    ----------
    da : xr.DataArray
        Input session DataArray with dims (time, x, y).
    enable_lowpass : bool
        Apply a Butterworth low-pass filter.
    lowpass_cutoff_hz : float
        Low-pass cutoff frequency in Hz.
    lowpass_order : int
        Low-pass filter order.
    enable_highpass : bool
        Apply a Butterworth high-pass filter.
    highpass_cutoff_hz : float
        High-pass cutoff frequency in Hz.
    highpass_order : int
        High-pass filter order.
    enable_clip : bool
        Apply global percentile clipping.
    clip_bottom : float
        Lower percentile for clipping.
    clip_top : float
        Upper percentile for clipping.
    fps_fallback : float, optional
        Frame rate to use if not present in da.attrs['frame_rate'].

    Returns
    -------
    xr.DataArray
        Filtered DataArray. Same dims and coords as input.
        Filter provenance is recorded in attrs.
    """
    frames = da.values.astype(np.float32)
    attrs = da.attrs.copy()
    session_id = attrs.get("session_id", "<unknown>")

    # Resolve fps from attrs or fallback
    if enable_lowpass or enable_highpass:
        fps = attrs.get("frame_rate", fps_fallback)
        if fps is None or float(fps) <= 0.0:
            raise ValueError(
                f"[{session_id}] Filter enabled but no valid frame_rate in attrs "
                f"and no fps_fallback provided."
            )
        fps = float(fps)

    if enable_lowpass:
        frames = low_pass_filter(frames, fps=fps, cutoff_hz=lowpass_cutoff_hz, order=lowpass_order)
        attrs["did_lowpass"] = True
        attrs["lowpass_cutoff_hz"] = float(lowpass_cutoff_hz)
        attrs["lowpass_order"] = int(lowpass_order)
        attrs["lowpass_fps"] = fps
    else:
        attrs["did_lowpass"] = False

    if enable_highpass:
        frames = high_pass_filter(frames, fps=fps, cutoff_hz=highpass_cutoff_hz, order=highpass_order)
        attrs["did_highpass"] = True
        attrs["highpass_cutoff_hz"] = float(highpass_cutoff_hz)
        attrs["highpass_order"] = int(highpass_order)
        attrs["highpass_fps"] = fps
    else:
        attrs["did_highpass"] = False

    if enable_clip:
        frames = percentile_clip(frames, bottom=clip_bottom, top=clip_top)
        attrs["did_clip"] = True
        attrs["clip_bottom"] = float(clip_bottom)
        attrs["clip_top"] = float(clip_top)
    else:
        attrs["did_clip"] = False

    attrs["stage"] = STAGE_FILTERED

    return xr.DataArray(
        data=frames,
        dims=da.dims,
        coords=da.coords,
        attrs=sanitize_attrs(attrs),
        name=da.name,
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def filter_reoriented_sessions(
    in_nc_paths: list[str],
    out_dir: str | os.PathLike[str],
    *,
    enable_lowpass: bool = False,
    lowpass_cutoff_hz: float = 0.5,
    lowpass_order: int = 4,
    enable_highpass: bool = False,
    highpass_cutoff_hz: float = 0.0,
    highpass_order: int = 3,
    enable_clip: bool = False,
    clip_bottom: float = 1.0,
    clip_top: float = 99.0,
    fps_fallback: float | None = None,
    overwrite: bool = False,
) -> list[str]:
    """
    Run the filtering stage over a list of session .nc files.

    Saves one filtered .nc file per session to out_dir.
    Skips sessions where the output already exists unless overwrite=True.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []
    for in_path_str in in_nc_paths:
        in_path = Path(in_path_str)
        da = xr.open_dataarray(in_path)

        session_id = da.attrs.get("session_id") or derive_session_id_from_path(in_path)
        out_path = out_root / f"baseline_{session_id}_{STAGE_FILTERED}.nc"

        if out_path.exists() and not overwrite:
            outputs.append(str(out_path))
            continue

        # Already filtered upstream — carry attrs forward without re-filtering
        if str(da.attrs.get("stage", "")) == STAGE_FILTERED:
            da.attrs["session_id"] = session_id
            da.to_netcdf(out_path)
            outputs.append(str(out_path))
            continue

        da_filtered = apply_optional_filters(
            da,
            enable_lowpass=enable_lowpass,
            lowpass_cutoff_hz=lowpass_cutoff_hz,
            lowpass_order=lowpass_order,
            enable_highpass=enable_highpass,
            highpass_cutoff_hz=highpass_cutoff_hz,
            highpass_order=highpass_order,
            enable_clip=enable_clip,
            clip_bottom=clip_bottom,
            clip_top=clip_top,
            fps_fallback=fps_fallback,
        )
        da_filtered.to_netcdf(out_path)
        outputs.append(str(out_path))
        print(f"  Filtered {in_path.name} → {out_path.name}")

    return outputs