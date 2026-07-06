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


def standardize_frames_pixelwise_causal(
    frames: np.ndarray,
    *,
    eps: float = 1e-8,
    floor_percentile: float = 10.0,
    clip_abs: float | None = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Causal (expanding-window) per-pixel z-score standardization.

    At each timepoint t, mean and std are estimated from frames 0..t (inclusive),
    so no future information is used. The std floor is applied per-frame using
    the spatial distribution of the running std map at that timepoint.

    Frame 0 is standardized with mean=frame[0] and std=floor (no variance yet),
    yielding zeros for t=0.

    Parameters
    ----------
    frames : np.ndarray, shape (T, H, W)
    eps : float
        Minimum value for the std floor.
    floor_percentile : float
        Percentile of the per-frame std map used as a robust floor.
    clip_abs : float or None
        If provided, clip standardized output to [-clip_abs, +clip_abs].

    Returns
    -------
    frames_z : np.ndarray, shape (T, H, W), float32
        Standardized frames.
    mean_map : np.ndarray, shape (H, W), float32
        Per-pixel mean estimated from the full session (for reference/inversion).
    std_map : np.ndarray, shape (H, W), float32
        Per-pixel std estimated from the full session (for reference/inversion).
    """
    _validate_frames_thw(frames, ctx="standardize_frames_pixelwise_causal")
    arr = np.asarray(frames, dtype=np.float64)
    T, H, W = arr.shape

    frames_z = np.empty((T, H, W), dtype=np.float32)

    # Welford online mean/variance accumulators
    running_mean = np.zeros((H, W), dtype=np.float64)
    running_M2   = np.zeros((H, W), dtype=np.float64)

    for t in range(T):
        n = t + 1
        delta = arr[t] - running_mean
        running_mean += delta / n
        delta2 = arr[t] - running_mean
        running_M2 += delta * delta2

        running_var = running_M2 / n  # biased (population) variance
        running_std = np.sqrt(running_var).astype(np.float32)

        std_floor = max(float(np.percentile(running_std, floor_percentile)), float(eps))
        std_floored = np.maximum(running_std, std_floor)

        z = (arr[t] - running_mean).astype(np.float32) / std_floored
        if clip_abs is not None and float(clip_abs) > 0:
            z = np.clip(z, -float(clip_abs), float(clip_abs))
        frames_z[t] = z

    mean_map = running_mean.astype(np.float32)
    std_map  = running_std.astype(np.float32)

    return frames_z, mean_map, std_map


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
    causal: bool = False,
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
    causal : bool
        If True, use expanding-window (causal) standardization instead of
        whole-session statistics. The mean_map and std_map saved in the output
        reflect the final running estimates (i.e. the full-session values).
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

        _std_fn = standardize_frames_pixelwise_causal if causal else standardize_frames_pixelwise

        # Z-score the raw frames once; smoothed variants are derived from this.
        frames_z, mean_map, std_map = _std_fn(
            frames_in,
            eps=eps,
            floor_percentile=floor_percentile,
            clip_abs=clip_abs,
        )

        def _make_dataset(frames: np.ndarray, kernel_size: int | None = None) -> xr.Dataset:
            T, H, W = frames.shape
            attrs = sanitize_attrs({
                **da.attrs,
                "stage":                STAGE_STANDARDIZED,
                "session_id":           session_id,
                "condition":            condition,
                "input_stage":          stage_in,
                "standardize_method":   "zscore_causal" if causal else "zscore",
                "standardize_eps":      eps,
                "floor_percentile":     floor_percentile,
                "clip_abs":             clip_abs if clip_abs is not None else "none",
                "smooth_kernel_size":   kernel_size if kernel_size is not None else "none",
                "zscored":              True,
            })
            return xr.Dataset(
                {
                    "frames":   xr.DataArray(frames,    dims=["time", "x", "y"]),
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
            ds = _make_dataset(frames_z)
            ds.to_netcdf(base_path)
            print(f"  Standardized {in_path.name} → {base_path.name}")
        saved.append(str(base_path))

        # Smoothed variants (applied to z-scored frames)
        for ks in smooth_kernel_sizes:
            ks      = int(ks)
            sm_name = f"baseline_{session_id}_{condition}_{STAGE_STANDARDIZED}_smooth{ks}x{ks}.nc"
            sm_path = out_root / sm_name
            if overwrite or not sm_path.exists():
                frames_smoothed = spatial_mean_filter_frames(frames_z, ks)
                ds = _make_dataset(frames_smoothed, kernel_size=ks)
                ds.to_netcdf(sm_path)
                print(f"  Standardized (smooth {ks}x{ks}) {in_path.name} → {sm_path.name}")
            saved.append(str(sm_path))

    return saved


def standardize_task_sessions_with_baseline_stats(
    task_nc_paths: list[str],
    baseline_std_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    clip_abs: float | None = 3.0,
    smooth_kernel_sizes: tuple[int, ...] = (),
    overwrite: bool = False,
) -> list[str]:
    """
    Z-score active (task) frames using per-pixel stats from matched baseline sessions.

    For each task session, loads the corresponding baseline standardized .nc to
    get mean_map and std_map, then applies those to the task frames. This puts
    task frames on the same scale as baseline z-scores.

    Parameters
    ----------
    task_nc_paths : list of str
        Paths to reoriented task .nc files.
    baseline_std_dir : str or Path
        Directory containing baseline standardized .nc files (must contain
        mean_map and std_map variables).
    out_dir : str or Path
        Directory where task standardized .nc files will be saved.
    clip_abs : float or None
        Clip standardized values to [-clip_abs, +clip_abs]. None to disable.
    overwrite : bool

    Returns
    -------
    list of str
        Paths to saved .nc files.
    """
    out_root     = Path(out_dir)
    baseline_dir = Path(baseline_std_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []

    for task_path_str in task_nc_paths:
        task_path  = Path(task_path_str)
        task_da    = xr.open_dataarray(task_path)
        session_id = task_da.attrs.get("session_id") or derive_session_id_from_path(task_path)

        # Find matching baseline standardized file (unfiltered variant)
        baseline_nc = baseline_dir / f"baseline_{session_id}_unfiltered_{STAGE_STANDARDIZED}.nc"
        if not baseline_nc.exists():
            # Fall back to any matching file for this session
            candidates = sorted(baseline_dir.glob(f"baseline_{session_id}_*_{STAGE_STANDARDIZED}.nc"))
            if not candidates:
                import warnings
                warnings.warn(
                    f"{session_id}: no baseline standardized file found in {baseline_dir} — skipping.",
                    stacklevel=2,
                )
                continue
            baseline_nc = candidates[0]

        baseline_ds = xr.open_dataset(baseline_nc)
        mean_map    = baseline_ds["mean_map"].values.astype(np.float32)
        std_map     = baseline_ds["std_map"].values.astype(np.float32)
        baseline_ds.close()

        frames = task_da.values.astype(np.float32)  # (T, H, W)

        frames_z = (frames - mean_map[np.newaxis]) / std_map[np.newaxis]
        if clip_abs is not None and float(clip_abs) > 0:
            frames_z = np.clip(frames_z, -float(clip_abs), float(clip_abs))

        attrs = sanitize_attrs({
            **task_da.attrs,
            "stage":              STAGE_STANDARDIZED,
            "session_id":         session_id,
            "condition":          "unfiltered",
            "standardize_method": "zscore_baseline_stats",
            "clip_abs":           clip_abs if clip_abs is not None else "none",
            "baseline_stats_src": baseline_nc.name,
            "zscored":            True,
        })

        ds = xr.Dataset(
            {
                "frames":   xr.DataArray(frames_z, dims=["time", "x", "y"]),
                "mean_map": xr.DataArray(mean_map,  dims=["x", "y"]),
                "std_map":  xr.DataArray(std_map,   dims=["x", "y"]),
            },
            coords=task_da.coords,
            attrs=attrs,
        )

        out_name = f"task_{session_id}_unfiltered_{STAGE_STANDARDIZED}.nc"
        out_path = out_root / out_name
        if overwrite or not out_path.exists():
            ds.to_netcdf(out_path)
            print(f"  Standardized (baseline stats) {task_path.name} → {out_path.name}")
        saved.append(str(out_path))

        # Smoothed variants (applied to z-scored frames)
        for ks in smooth_kernel_sizes:
            ks      = int(ks)
            sm_name = f"task_{session_id}_unfiltered_{STAGE_STANDARDIZED}_smooth{ks}x{ks}.nc"
            sm_path = out_root / sm_name
            if overwrite or not sm_path.exists():
                frames_smoothed = spatial_mean_filter_frames(frames_z, ks)
                sm_attrs = sanitize_attrs({**attrs, "smooth_kernel_size": ks})
                sm_ds = xr.Dataset(
                    {
                        "frames":   xr.DataArray(frames_smoothed, dims=["time", "x", "y"]),
                        "mean_map": xr.DataArray(mean_map,         dims=["x", "y"]),
                        "std_map":  xr.DataArray(std_map,          dims=["x", "y"]),
                    },
                    coords=task_da.coords,
                    attrs=sm_attrs,
                )
                sm_ds.to_netcdf(sm_path)
                print(f"  Standardized (baseline stats, smooth {ks}x{ks}) {task_path.name} → {sm_path.name}")
            saved.append(str(sm_path))

    return saved