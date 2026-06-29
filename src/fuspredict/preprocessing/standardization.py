"""
standardization.py
------------------
Per-pixel z-score standardization for fUS session DataArrays.

Each output is an xr.Dataset containing:
  - frames   : standardized frames (T, x, y)
  - mean_map : per-pixel temporal mean (x, y)
  - std_map  : per-pixel temporal std  (x, y)

The mean_map and std_map allow inverse transformation back to the original
signal space: original ≈ frames * std_map + mean_map.

Typical usage
-------------
    from standardization import standardize_stage_sessions

    standardize_stage_sessions(
        reoriented_paths + filtered_paths,
        out_dir="derivatives/baseline_only_standardized",
        eps=1e-8,
        floor_percentile=10.0,
        clip_abs=3.0,
    )
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

from .io import (
    STAGE_FILTERED,
    STAGE_REORIENTED_RESIZED,
    STAGE_STANDARDIZED,
    derive_session_id_from_path,
    sanitize_attrs,
    spatial_mean_filter_frames,
)

CONDITION_UNFILTERED = "unfiltered"
CONDITION_FILTERED   = "filtered"

_SUPPORTED_INPUT_STAGES = (STAGE_REORIENTED_RESIZED, STAGE_FILTERED)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_frames_thw(frames: np.ndarray, ctx: str) -> tuple[int, int, int]:
    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError(f"{ctx}: expected shape (T, H, W), got {arr.shape}")
    t, h, w = arr.shape
    return int(t), int(h), int(w)


def _condition_for_stage(stage: str) -> str:
    if stage == STAGE_REORIENTED_RESIZED:
        return CONDITION_UNFILTERED
    if stage == STAGE_FILTERED:
        return CONDITION_FILTERED
    raise ValueError(
        f"Unsupported input stage {stage!r}; expected one of {_SUPPORTED_INPUT_STAGES!r}."
    )


# ---------------------------------------------------------------------------
# Core standardization
# ---------------------------------------------------------------------------

def standardize_frames_pixelwise(
    frames: np.ndarray,
    *,
    eps: float = 1e-8,
    floor_percentile: float = 10.0,
    clip_abs: float | None = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score standardize (T, H, W) frames per pixel over time.

    A robust floor is applied to the std map to avoid division by near-zero
    values in low-signal pixels: any std below the floor_percentile of the
    std distribution is raised to that floor.

    Parameters
    ----------
    frames : np.ndarray, shape (T, H, W)
    eps : float
        Minimum value for the std floor. Prevents division by zero.
    floor_percentile : float
        Percentile of the std map used as a robust floor.
    clip_abs : float or None
        If provided, clip standardized output to [-clip_abs, +clip_abs].

    Returns
    -------
    frames_z : np.ndarray, shape (T, H, W), float32
        Standardized frames.
    mean_map : np.ndarray, shape (H, W), float32
        Per-pixel temporal mean used for standardization.
    std_map : np.ndarray, shape (H, W), float32
        Per-pixel temporal std used for standardization (after floor).
    """
    _validate_frames_thw(frames, ctx="standardize_frames_pixelwise")
    arr = np.asarray(frames, dtype=np.float32)

    mean_map = arr.mean(axis=0).astype(np.float32)
    std_map  = arr.std(axis=0).astype(np.float32)

    std_floor = max(float(np.percentile(std_map, floor_percentile)), float(eps))
    std_map   = np.maximum(std_map, std_floor)

    frames_z = (arr - mean_map[np.newaxis]) / std_map[np.newaxis]

    if clip_abs is not None and float(clip_abs) > 0:
        frames_z = np.clip(frames_z, -float(clip_abs), float(clip_abs))

    return frames_z.astype(np.float32), mean_map, std_map


def inverse_standardize(
    frames_z: np.ndarray,
    mean_map: np.ndarray,
    std_map: np.ndarray,
) -> np.ndarray:
    """
    Invert z-score standardization: original ≈ frames_z * std_map + mean_map.

    Parameters
    ----------
    frames_z : np.ndarray, shape (T, H, W)
        Standardized frames.
    mean_map : np.ndarray, shape (H, W)
        Per-pixel mean from standardize_frames_pixelwise.
    std_map : np.ndarray, shape (H, W)
        Per-pixel std from standardize_frames_pixelwise.

    Returns
    -------
    np.ndarray, shape (T, H, W), float32
    """
    arr  = np.asarray(frames_z, dtype=np.float32)
    mean = np.asarray(mean_map, dtype=np.float32)
    std  = np.asarray(std_map,  dtype=np.float32)
    return (arr * std[np.newaxis] + mean[np.newaxis]).astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def standardize_stage_sessions(
    in_nc_paths: list[str],
    out_dir: str | os.PathLike[str],
    *,
    eps: float = 1e-8,
    floor_percentile: float = 10.0,
    clip_abs: float | None = 3.0,
    smooth_kernel_sizes: tuple[int, ...] = (),
    overwrite: bool = False,
) -> list[str]:
    """
    Z-score standardize a list of session .nc files and save as xr.Datasets.

    Accepts both reoriented and filtered sessions in the same call. The
    condition (unfiltered/filtered) is inferred from each file's stage attr
    and recorded in the output attrs.

    For each input session, one base output is produced. If smooth_kernel_sizes
    is provided, additional spatially-smoothed variants are also saved.

    Output files are xr.Datasets containing:
      - frames   : standardized frames (time, x, y)
      - mean_map : per-pixel temporal mean (x, y)
      - std_map  : per-pixel temporal std  (x, y)

    Parameters
    ----------
    in_nc_paths : list of str
        Paths to input .nc session files (reoriented or filtered stage).
    out_dir : str or Path
        Directory where standardized .nc files will be saved.
    eps : float
        Minimum std floor for numerical stability.
    floor_percentile : float
        Percentile of the std map used as a robust floor.
    clip_abs : float or None
        Clip standardized values to [-clip_abs, +clip_abs]. None to disable.
    smooth_kernel_sizes : tuple of int
        If provided, also save spatially-smoothed standardized variants for
        each kernel size (e.g. (2, 3) saves 2x2 and 3x3 smoothed outputs).
    overwrite : bool
        Overwrite existing output files.

    Returns
    -------
    list of str
        Paths to all saved .nc files (base + smooth variants).
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []

    for in_path_str in in_nc_paths:
        in_path = Path(in_path_str)
        da      = xr.open_dataarray(in_path)

        stage_in   = str(da.attrs.get("stage", ""))
        if stage_in not in _SUPPORTED_INPUT_STAGES:
            raise ValueError(
                f"{in_path.name}: expected input stage in {_SUPPORTED_INPUT_STAGES!r}, "
                f"got {stage_in!r}."
            )

        session_id = da.attrs.get("session_id") or derive_session_id_from_path(in_path)
        condition  = _condition_for_stage(stage_in)
        frames_in  = da.values  # (T, H, W)

        def _make_dataset(frames: np.ndarray, kernel_size: int | None = None) -> xr.Dataset:
            frames_z, mean_map, std_map = standardize_frames_pixelwise(
                frames,
                eps=eps,
                floor_percentile=floor_percentile,
                clip_abs=clip_abs,
            )
            T, H, W = frames_z.shape
            attrs = sanitize_attrs({
                **da.attrs,
                "stage":                STAGE_STANDARDIZED,
                "session_id":           session_id,
                "condition":            condition,
                "input_stage":          stage_in,
                "standardize_method":   "zscore",
                "standardize_eps":      eps,
                "floor_percentile":     floor_percentile,
                "clip_abs":             clip_abs if clip_abs is not None else "none",
                "smooth_kernel_size":   kernel_size if kernel_size is not None else "none",
                "zscored":              True,
            })
            return xr.Dataset(
                {
                    "frames":   xr.DataArray(frames_z,  dims=["time", "x", "y"]),
                    "mean_map": xr.DataArray(mean_map,  dims=["x", "y"]),
                    "std_map":  xr.DataArray(std_map,   dims=["x", "y"]),
                },
                coords=da.coords,
                attrs=attrs,
            )

        # Base output (no smoothing)
        base_name = f"baseline_{session_id}_{condition}_{STAGE_STANDARDIZED}.nc"
        base_path = out_root / base_name
        if overwrite or not base_path.exists():
            ds = _make_dataset(frames_in)
            ds.to_netcdf(base_path)
            print(f"  Standardized {in_path.name} → {base_path.name}")
        saved.append(str(base_path))

        # Smoothed variants
        for ks in smooth_kernel_sizes:
            ks        = int(ks)
            sm_name   = f"baseline_{session_id}_{condition}_{STAGE_STANDARDIZED}_smooth{ks}x{ks}.nc"
            sm_path   = out_root / sm_name
            if overwrite or not sm_path.exists():
                frames_smoothed = spatial_mean_filter_frames(frames_in, ks)
                ds = _make_dataset(frames_smoothed, kernel_size=ks)
                ds.to_netcdf(sm_path)
                print(f"  Standardized (smooth {ks}x{ks}) {in_path.name} → {sm_path.name}")
            saved.append(str(sm_path))

    return saved