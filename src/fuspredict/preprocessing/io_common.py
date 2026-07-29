"""
io_common.py
------------
Shared I/O helpers used by both the primate (.mat) and mouse (.source.scan)
extraction pipelines: NetCDF4 attr sanitization, session-id parsing, frame
alignment, spatial smoothing, and the label sidecar writer.

Author: Brynn Harris-Shanks, 2026
With adaptations from code by Leo Sperber, 2025
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Stage name constants
# ---------------------------------------------------------------------------

BASELINE_STAGE_EXTRACTED = "baseline_extracted"
TASK_STAGE_EXTRACTED     = "task_extracted"
STAGE_REORIENTED_RESIZED = "reoriented_resized"
STAGE_FILTERED           = "filtered"
STAGE_STANDARDIZED       = "standardized_zscore"

KNOWN_STAGE_SUFFIXES = (
    BASELINE_STAGE_EXTRACTED,
    TASK_STAGE_EXTRACTED,
    STAGE_REORIENTED_RESIZED,
    STAGE_FILTERED,
    STAGE_STANDARDIZED,
)


# ---------------------------------------------------------------------------
# NetCDF4 attr sanitization
# ---------------------------------------------------------------------------

def sanitize_attrs(attrs: dict) -> dict:
    """
    Convert Python types that NetCDF4 cannot store as attributes.

    NetCDF4 only supports numeric scalars and strings as attributes.
    Bools, None, and lists must be converted before calling da.to_netcdf().

    Conversions applied:
      bool  → "True" / "False"
      None  → "none"
      list  → comma-separated string  e.g. [1, 2, 3] → "1,2,3"
      other → unchanged (int, float, str, np scalar all fine)
    """
    out = {}
    for k, v in attrs.items():
        if isinstance(v, bool):
            out[k] = "True" if v else "False"
        elif v is None:
            out[k] = "none"
        elif isinstance(v, list):
            out[k] = ",".join(str(x) for x in v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def derive_session_id_from_path(path: str | os.PathLike[str]) -> str:
    """
    Derive session_id from a stage filename.

    Handles:
      - baseline_<session>_<known_stage>.nc
      - baseline_<session>.nc
    """
    stem = Path(path).stem
    if stem.startswith("baseline_"):
        stem = stem[len("baseline_"):]
    for stage_suffix in KNOWN_STAGE_SUFFIXES:
        if stem.endswith(f"_{stage_suffix}"):
            stem = stem[: -len(f"_{stage_suffix}")]
            break
    return stem


def mismatch(images: np.ndarray, labels_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align image and label sequences by trimming both to the shortest length."""
    if images.shape[0] != len(labels_arr):
        min_len = min(images.shape[0], len(labels_arr))
        print(
            f"  MISMATCH: images={images.shape[0]}, labels={len(labels_arr)}. "
            f"Trimming to {min_len} frames."
        )
        return images[:min_len], labels_arr[:min_len]
    print(f"  Match confirmed: {images.shape[0]} frames and labels.")
    return images, labels_arr


def spatial_mean_filter_frames(
    frames: np.ndarray,
    kernel_size: int,
    mode: str = "nearest",
) -> np.ndarray:
    """
    Spatially smooth each frame with a square mean kernel.

    Parameters
    ----------
    frames : np.ndarray, shape (T, H, W)
    kernel_size : int
        Square neighbourhood width. 1 returns a copy unchanged.
    mode : str
        Boundary handling mode passed to scipy.ndimage.convolve.

    Returns
    -------
    np.ndarray, shape (T, H, W), float32
    """
    from scipy.ndimage import convolve

    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"frames must have shape (T, H, W), got {arr.shape}")
    k = int(kernel_size)
    if k < 1:
        raise ValueError(f"kernel_size must be a positive integer, got {kernel_size}")
    if k == 1:
        return arr.copy()
    kernel = np.ones((k, k), dtype=np.float32) / (k * k)
    out = np.empty_like(arr)
    for t in range(arr.shape[0]):
        out[t] = convolve(arr[t], kernel, mode=mode)
    return out


# ---------------------------------------------------------------------------
# Label sidecar
# ---------------------------------------------------------------------------

LABEL_SIDECAR_SUFFIX = "labels"


def save_label_sidecar(
    labels_arr: np.ndarray,
    session_id: str,
    output_dir: str | os.PathLike[str],
    fps: float,
    source_file: str,
    *,
    has_real_timing: bool = True,
    overwrite: bool = False,
) -> str | None:
    """
    Write the full (trimmed) per-frame label sequence as a sidecar .nc file.

    The sidecar records the raw integer label code for every frame in the
    trimmed acquisition timeline, co-indexed with the baseline/task .nc files
    produced from the same mismatch()-trimmed arrays.  Downstream loaders
    recover period indices by grouping on label value without touching the
    source .mat files.

    Label convention (monkey .mat):
      -1  baseline
       0  pause
      >0  task/stimulus

    Mouse sessions use a synthetic label array (-1 = baseline, 1 = task)
    derived from the timing mask; the same convention applies.

    Parameters
    ----------
    labels_arr : np.ndarray, shape (T,), integer dtype
        Per-frame label codes over the full trimmed timeline.
    session_id : str
    output_dir : str or Path
    fps : float
    source_file : str
        Basename of the source .mat or .source.scan file (for provenance).
    overwrite : bool

    Returns
    -------
    str or None
        Path to the saved sidecar, or None if it already exists and
        overwrite is False.
    """
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{LABEL_SIDECAR_SUFFIX}_{session_id}.nc"

    if out_path.exists() and not overwrite:
        return str(out_path)

    T = len(labels_arr)
    da = xr.DataArray(
        data=labels_arr.astype(np.int8),
        dims=["time"],
        coords={"time": np.arange(T) / fps},
        attrs=sanitize_attrs({
            "session_id":   session_id,
            "frame_rate":   fps,
            "n_frames":     int(T),
            "source_file":  source_file,
            "label_codes":  "-1=baseline, 0=pause, >0=task",
            "has_real_timing": has_real_timing,
        }),
        name="labels",
    )
    ds = da.to_dataset(name="labels")
    ds.to_netcdf(out_path)
    return str(out_path)
