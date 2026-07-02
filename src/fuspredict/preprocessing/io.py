"""
io.py
-----
Source data extraction and stage I/O for fUS preprocessing.

Reads raw .mat files (Datas_*.mat + Label_pauses_*.mat), extracts baseline
and task frames, and saves stage 1 outputs as xarray DataArrays in .nc format.

Author: Brynn Harris-Shanks, 2026
With adaptations from code by Leo Sperber, 2025
"""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

import numpy as np
import scipy.io
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
# .mat extraction primitives
# ---------------------------------------------------------------------------

def extract_baseline_frames_from_mat(mat_dict: dict) -> np.ndarray:
    """Extract frames from a .mat structure and return shape (T, H, W)."""
    key = "Data" if "Data" in mat_dict else ("Datas" if "Datas" in mat_dict else None)
    if key is None:
        raise KeyError("Neither 'Data' nor 'Datas' found in .mat file.")

    try:
        fus_struct = mat_dict[key]["fus"][0, 0]
        frames     = fus_struct["frame"][0, 0]
    except Exception as exc:
        raise KeyError("Failed to access mat[key]['fus'][0,0]['frame'][0,0].") from exc

    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"Expected 3D frames, got shape {frames.shape}.")

    # MATLAB stores as (H, W, T) — transpose to (T, H, W)
    if frames.shape[2] > frames.shape[0] and frames.shape[2] > frames.shape[1]:
        frames = np.transpose(frames, (2, 0, 1))

    return np.asarray(frames, dtype=np.float32)


def extract_fps_from_mat(mat_dict: dict) -> float | None:
    """Best-effort extraction of acquisition fps from a .mat structure."""
    key = "Data" if "Data" in mat_dict else ("Datas" if "Datas" in mat_dict else None)
    if key is None:
        return None

    try:
        fus_struct = mat_dict[key]["fus"][0, 0]
    except Exception:
        return None

    for cand in ("fps", "frame_rate", "framerate", "sampling_rate", "acq_fps"):
        if cand not in (fus_struct.dtype.names or ()):
            continue
        try:
            val = float(np.asarray(fus_struct[cand][0, 0]).squeeze())
            if val > 0:
                return val
        except Exception:
            continue

    return None


def load_label_file(label_path: str) -> np.ndarray:
    """Load labels from a Label_pauses_*.mat file."""
    lab = scipy.io.loadmat(label_path)
    if "Datas" not in lab:
        raise KeyError(f"{label_path}: missing top-level key 'Datas'.")
    try:
        labels = lab["Datas"]["Label"][0, 0]
    except Exception as exc:
        raise KeyError(f"{label_path}: failed to access Datas['Label'][0,0].") from exc
    return np.asarray(labels).squeeze()


# ---------------------------------------------------------------------------
# Stage 1 — Baseline extraction
# ---------------------------------------------------------------------------

def extract_and_save_baseline(
    fus_path: str,
    label_path: str,
    output_dir: str,
    *,
    baseline_value: int = -1,
    apply_log10: bool = True,
    log10_eps: float = 1e-6,
    overwrite: bool = False,
) -> str | None:
    """
    Extract baseline frames from a .mat session and save as a .nc DataArray.

    Parameters
    ----------
    fus_path : str
        Path to Datas_*.mat file.
    label_path : str
        Path to the matching Label_pauses_*.mat file.
    output_dir : str
        Directory where the output .nc file will be saved.
    baseline_value : int
        Label value that identifies baseline frames. Default: -1.
    apply_log10 : bool
        Apply log10(frames + eps) before saving.
    log10_eps : float
        Epsilon added before log10 to avoid log(0).
    overwrite : bool
        Overwrite existing output files.

    Returns
    -------
    str or None
        Path to the saved .nc file, or None if extraction failed.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_code = Path(fus_path).stem.replace("Datas_", "")
    out_path  = out_dir / f"baseline_{date_code}_{BASELINE_STAGE_EXTRACTED}.nc"

    if out_path.exists() and not overwrite:
        print(f"  Skipping {date_code} (already exists)")
        return str(out_path)

    try:
        mat        = scipy.io.loadmat(fus_path)
        frames     = extract_baseline_frames_from_mat(mat)
        source_fps = extract_fps_from_mat(mat)

        labels_arr         = load_label_file(label_path)
        frames, labels_arr = mismatch(frames, labels_arr)

        baseline_mask    = labels_arr == baseline_value
        baseline_indices = np.where(baseline_mask)[0]
        baseline_frames  = frames[baseline_mask]

        if baseline_frames.shape[0] == 0:
            warnings.warn(
                f"{date_code}: no baseline frames found (label == {baseline_value}); skipping.",
                stacklevel=2,
            )
            return None

        if apply_log10:
            eps             = float(np.float32(log10_eps))
            baseline_frames = np.log10(baseline_frames + eps).astype(np.float32)
            did_log10       = True
        else:
            baseline_frames = baseline_frames.astype(np.float32)
            did_log10       = False
            eps             = None

        T, H, W = baseline_frames.shape
        fps     = source_fps or 2.5

        attrs = sanitize_attrs({
            "stage":             BASELINE_STAGE_EXTRACTED,
            "session_id":        date_code,
            "frame_rate":        fps,
            "source_fus_file":   os.path.basename(fus_path),
            "source_label_file": os.path.basename(label_path),
            "n_total_frames":    int(frames.shape[0]),
            "n_baseline_frames": int(T),
            "baseline_value":    int(baseline_value),
            "baseline_indices":  baseline_indices.tolist(),
            "did_log10":         did_log10,
            "log10_eps":         eps,
            "zscored":           False,
        })

        da = xr.DataArray(
            data=baseline_frames,
            dims=["time", "x", "y"],
            coords={
                "time": np.arange(T) / fps,
                "x":    np.arange(H),
                "y":    np.arange(W),
            },
            attrs=attrs,
            name=date_code,
        )
        da.to_netcdf(out_path)
        print(
            f"  Saved {out_path.name}: {T}/{frames.shape[0]} baseline frames "
            f"({100.0 * T / max(1, frames.shape[0]):.1f}%)"
        )
        return str(out_path)

    except Exception as exc:
        warnings.warn(
            f"Skipping {os.path.basename(fus_path)}: {exc}",
            stacklevel=2,
        )
        return None


def process_all_baseline_files(
    data_directory: str,
    output_dir: str,
    *,
    overwrite: bool = False,
    apply_log10: bool = True,
    log10_eps: float = 1e-6,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """Extract baseline stage files for all sessions in a source directory."""
    fus_files = sorted(glob.glob(os.path.join(data_directory, "Datas_*.mat")))
    if not fus_files:
        print(f"No Datas_*.mat files found in {data_directory}")
        return []

    print(f"Found {len(fus_files)} fUS files")
    saved_paths: list[str] = []

    for fus_path in fus_files:
        date_code  = Path(fus_path).stem.replace("Datas_", "")
        if exclude_ids and date_code in exclude_ids:
            print(f"  Skipping excluded session {date_code}")
            continue
        label_path = os.path.join(data_directory, f"Label_pauses_{date_code}.mat")

        if not os.path.exists(label_path):
            fallback = sorted(glob.glob(os.path.join(data_directory, f"Label*{date_code}.mat")))
            if fallback:
                label_path = fallback[0]
            else:
                warnings.warn(
                    f"No label file for {os.path.basename(fus_path)}; skipping.",
                    stacklevel=2,
                )
                continue

        out = extract_and_save_baseline(
            fus_path=fus_path,
            label_path=label_path,
            output_dir=output_dir,
            apply_log10=apply_log10,
            log10_eps=log10_eps,
            overwrite=overwrite,
        )
        if out is not None:
            saved_paths.append(out)

    print(f"Extracted baseline from {len(saved_paths)}/{len(fus_files)} sessions")
    return saved_paths


# ---------------------------------------------------------------------------
# Stage 1b — Task extraction
# ---------------------------------------------------------------------------

def extract_and_save_task(
    fus_path: str,
    label_path: str,
    output_dir: str,
    *,
    baseline_value: int = -1,
    apply_log10: bool = True,
    log10_eps: float = 1e-6,
    overwrite: bool = False,
) -> str | None:
    """
    Extract non-baseline (task/stimulus) frames and save as a .nc DataArray.

    Parameters
    ----------
    fus_path : str
        Path to Datas_*.mat file.
    label_path : str
        Path to the matching Label_pauses_*.mat file.
    output_dir : str
        Directory where the output .nc file will be saved.
    baseline_value : int
        Label value identifying baseline frames (excluded here). Default: -1.
    apply_log10 : bool
        Apply log10(frames + eps) before saving.
    log10_eps : float
        Epsilon added before log10 to avoid log(0).
    overwrite : bool
        Overwrite existing output files.

    Returns
    -------
    str or None
        Path to the saved .nc file, or None if extraction failed.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_code = Path(fus_path).stem.replace("Datas_", "")
    out_path  = out_dir / f"task_{date_code}_{TASK_STAGE_EXTRACTED}.nc"

    if out_path.exists() and not overwrite:
        print(f"  Skipping {date_code} (already exists)")
        return str(out_path)

    try:
        mat        = scipy.io.loadmat(fus_path)
        frames     = extract_baseline_frames_from_mat(mat)
        source_fps = extract_fps_from_mat(mat)

        labels_arr         = load_label_file(label_path)
        frames, labels_arr = mismatch(frames, labels_arr)

        task_mask    = labels_arr != baseline_value
        task_indices = np.where(task_mask)[0]
        task_frames  = frames[task_mask]
        task_labels  = labels_arr[task_mask]

        if task_frames.shape[0] == 0:
            warnings.warn(
                f"{date_code}: no task frames found (all labels == {baseline_value}); skipping.",
                stacklevel=2,
            )
            return None

        if apply_log10:
            eps         = float(np.float32(log10_eps))
            task_frames = np.log10(task_frames + eps).astype(np.float32)
            did_log10   = True
        else:
            task_frames = task_frames.astype(np.float32)
            did_log10   = False
            eps         = None

        T, H, W = task_frames.shape
        fps     = source_fps or 2.5

        attrs = sanitize_attrs({
            "stage":              TASK_STAGE_EXTRACTED,
            "session_id":         date_code,
            "frame_rate":         fps,
            "source_fus_file":    os.path.basename(fus_path),
            "source_label_file":  os.path.basename(label_path),
            "n_total_frames":     int(frames.shape[0]),
            "n_task_frames":      int(T),
            "baseline_value":     int(baseline_value),
            "unique_task_labels": sorted(int(v) for v in np.unique(task_labels)),
            "task_indices":       task_indices.tolist(),
            "task_labels":        task_labels.astype(np.int64).tolist(),
            "did_log10":          did_log10,
            "log10_eps":          eps,
            "zscored":            False,
        })

        da = xr.DataArray(
            data=task_frames,
            dims=["time", "x", "y"],
            coords={
                "time": np.arange(T) / fps,
                "x":    np.arange(H),
                "y":    np.arange(W),
            },
            attrs=attrs,
            name=date_code,
        )
        da.to_netcdf(out_path)
        print(
            f"  Saved {out_path.name}: {T}/{frames.shape[0]} task frames "
            f"({100.0 * T / max(1, frames.shape[0]):.1f}%)  "
            f"labels={sorted(int(v) for v in np.unique(task_labels))}"
        )
        return str(out_path)

    except Exception as exc:
        warnings.warn(
            f"Skipping {os.path.basename(fus_path)}: {exc}",
            stacklevel=2,
        )
        return None


def process_all_task_files(
    data_directory: str,
    output_dir: str,
    *,
    overwrite: bool = False,
    apply_log10: bool = True,
    log10_eps: float = 1e-6,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """Extract task stage files for all sessions in a source directory."""
    fus_files = sorted(glob.glob(os.path.join(data_directory, "Datas_*.mat")))
    if not fus_files:
        print(f"No Datas_*.mat files found in {data_directory}")
        return []

    print(f"Found {len(fus_files)} fUS files")
    saved_paths: list[str] = []

    for fus_path in fus_files:
        date_code  = Path(fus_path).stem.replace("Datas_", "")
        if exclude_ids and date_code in exclude_ids:
            print(f"  Skipping excluded session {date_code}")
            continue
        label_path = os.path.join(data_directory, f"Label_pauses_{date_code}.mat")

        if not os.path.exists(label_path):
            fallback = sorted(glob.glob(os.path.join(data_directory, f"Label*{date_code}.mat")))
            if fallback:
                label_path = fallback[0]
            else:
                warnings.warn(
                    f"No label file for {os.path.basename(fus_path)}; skipping.",
                    stacklevel=2,
                )
                continue

        out = extract_and_save_task(
            fus_path=fus_path,
            label_path=label_path,
            output_dir=output_dir,
            apply_log10=apply_log10,
            log10_eps=log10_eps,
            overwrite=overwrite,
        )
        if out is not None:
            saved_paths.append(out)

    print(f"Extracted task frames from {len(saved_paths)}/{len(fus_files)} sessions")
    return saved_paths