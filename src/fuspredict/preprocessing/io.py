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
        }),
        name="labels",
    )
    ds = da.to_dataset(name="labels")
    ds.to_netcdf(out_path)
    return str(out_path)


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

        fps = source_fps or 2.5
        save_label_sidecar(
            labels_arr=labels_arr,
            session_id=date_code,
            output_dir=output_dir,
            fps=fps,
            source_file=os.path.basename(label_path),
            overwrite=overwrite,
        )

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


# ---------------------------------------------------------------------------
# Mouse .source.scan support
# ---------------------------------------------------------------------------

def load_source_scan(path: str | os.PathLike[str]) -> tuple[np.ndarray, float, np.ndarray, dict]:
    """
    Load fUS frames and metadata from a Verasonics .source.scan HDF5 file.

    Parameters
    ----------
    path : str or Path
        Path to a ``*.source.scan`` file.

    Returns
    -------
    frames : np.ndarray, shape (T, H, W), float32
        Power-Doppler frames.
    fps : float
        Acquisition frame rate in Hz.
    frame_times : np.ndarray, shape (T,), float64
        Timestamp of each frame in seconds (from ``acqMetaData/time``).
    scan_meta : dict
        Key scanMetaData fields: ``subject_tag``, ``date``, ``session_tag``,
        ``scan_tag``.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to read .source.scan files.") from exc

    with h5py.File(str(path), "r") as f:
        raw = np.asarray(f["Data"], dtype=np.float32)
        dt = float(np.asarray(f["acqMetaData/voxDim/dt"]).squeeze())
        frame_times = np.asarray(f["acqMetaData/time"]).squeeze().astype(np.float64)

        def _read_str(key: str) -> str:
            try:
                val = f[key][:]
                raw_val = val.flat[0]
                if isinstance(raw_val, (bytes, np.bytes_)):
                    return raw_val.decode("utf-8", errors="replace").strip()
                return str(raw_val).strip()
            except Exception:
                return ""

        scan_meta = {
            "subject_tag":  _read_str("scanMetaData/Subject_tag"),
            "date":         _read_str("scanMetaData/Date"),
            "session_tag":  _read_str("scanMetaData/Session_tag"),
            "scan_tag":     _read_str("scanMetaData/Scan_tag"),
        }

    # Data is (T, H, 1, W) — squeeze the singleton Z dim
    if raw.ndim == 4 and raw.shape[2] == 1:
        raw = raw[:, :, 0, :]

    _DT_MIN, _DT_MAX = 0.01, 10.0  # plausible fUS range: 0.1–100 Hz
    if dt <= 0:
        warnings.warn(
            f"acqMetaData/voxDim/dt={dt} is non-positive; using fps fallback of 2.5 Hz. "
            "Check HDF5 file integrity.",
            stacklevel=2,
        )
        fps = 2.5
    elif not (_DT_MIN <= dt <= _DT_MAX):
        warnings.warn(
            f"acqMetaData/voxDim/dt={dt:.6f} s is outside expected range "
            f"[{_DT_MIN}, {_DT_MAX}] s — possible unit mismatch (ms vs s?). "
            f"Computed fps={1.0/dt:.4f} Hz. Verify HDF5 units.",
            stacklevel=2,
        )
        fps = 1.0 / dt
    else:
        fps = 1.0 / dt
    return raw, fps, frame_times, scan_meta


def parse_perifus_excel(excel_path: str | os.PathLike[str]) -> dict[str, dict]:
    """
    Parse stimulus timing from the PerifUS summary Excel spreadsheet.

    Reads the ``Pattern`` column (e.g. "60s baseline/ 10s ON/ 30s OFF/ 40 trials")
    and returns per-subject stimulus timing parameters.

    Parameters
    ----------
    excel_path : str or Path

    Returns
    -------
    dict mapping subject_id (str, e.g. "S2C") -> timing dict with keys:
        ``baseline_s``, ``stim_on_s``, ``stim_off_s``, ``n_trials``
    """
    import re
    import pandas as pd

    df = pd.read_excel(str(excel_path))
    timing: dict[str, dict] = {}

    pattern_re = re.compile(
        r"(?P<base>[\d.]+)\s*s\s+baseline"
        r".*?(?P<on>[\d.]+)\s*s\s+ON"
        r".*?(?P<off>[\d.]+)\s*s\s+OFF"
        r".*?(?P<trials>\d+)\s+trials",
        re.IGNORECASE,
    )

    for _, row in df.iterrows():
        subject = str(row.get("subject", "")).strip()
        pattern_str = str(row.get("Pattern", "")).strip()
        if not subject or subject.lower() == "nan":
            continue
        m = pattern_re.search(pattern_str)
        if m:
            timing[subject] = {
                "baseline_s":  float(m.group("base")),
                "stim_on_s":   float(m.group("on")),
                "stim_off_s":  float(m.group("off")),
                "n_trials":    int(m.group("trials")),
            }
        else:
            # Store with None so callers know the subject exists but pattern unparsed
            timing[subject] = None

    return timing


def extract_baseline_mask_from_timing(
    frame_times: np.ndarray,
    timing: dict,
) -> np.ndarray:
    """
    Return a boolean mask selecting resting-state (non-stimulus) frames.

    Frames are baseline if they occur before the first stimulus onset, or
    during inter-stimulus intervals (after stim offset, before next onset).

    Parameters
    ----------
    frame_times : np.ndarray, shape (T,)
        Absolute timestamp of each frame in seconds.
    timing : dict
        Keys: ``baseline_s``, ``stim_on_s``, ``stim_off_s``, ``n_trials``.

    Returns
    -------
    np.ndarray of bool, shape (T,)
    """
    mask = np.ones(len(frame_times), dtype=bool)
    base  = timing["baseline_s"]
    on_s  = timing["stim_on_s"]
    off_s = timing["stim_off_s"]
    cycle = on_s + off_s

    for i in range(timing["n_trials"]):
        onset  = base + i * cycle
        offset = onset + on_s
        mask &= ~((frame_times >= onset) & (frame_times < offset))

    return mask


def extract_and_save_baseline_mouse(
    scan_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    timing: dict | None = None,
    *,
    apply_log10: bool = True,
    log10_eps: float = 1e-6,
    overwrite: bool = False,
    _preloaded: tuple | None = None,
) -> str | None:
    """
    Extract resting-state frames from a .source.scan file and save as .nc.

    Parameters
    ----------
    scan_path : str or Path
        Path to a ``*.source.scan`` HDF5 file.
    output_dir : str or Path
        Directory for the output ``.nc`` file.
    timing : dict or None
        Stimulus timing from :func:`parse_perifus_excel`. If None, all frames
        are treated as baseline.
    apply_log10 : bool
        Apply log10(frames + eps) before saving.
    log10_eps : float
    overwrite : bool

    Returns
    -------
    str or None
        Path to the saved ``.nc`` file, or None if extraction failed.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id = Path(scan_path).with_suffix("").stem  # strips both .source and .scan
    out_path = out_dir / f"baseline_{session_id}_{BASELINE_STAGE_EXTRACTED}.nc"

    if out_path.exists() and not overwrite:
        print(f"  Skipping {session_id} (already exists)")
        return str(out_path)

    try:
        if _preloaded is not None:
            frames, fps, frame_times, scan_meta = _preloaded
        else:
            frames, fps, frame_times, scan_meta = load_source_scan(scan_path)
        T_total = frames.shape[0]

        if timing is not None:
            baseline_mask = extract_baseline_mask_from_timing(frame_times, timing)
        else:
            baseline_mask = np.ones(T_total, dtype=bool)

        # Synthesize label array: -1 = baseline, 1 = task (no pause concept for mouse)
        synth_labels = np.where(baseline_mask, np.int8(-1), np.int8(1))
        save_label_sidecar(
            labels_arr=synth_labels,
            session_id=session_id,
            output_dir=output_dir,
            fps=fps,
            source_file=os.path.basename(str(scan_path)),
            overwrite=overwrite,
        )

        baseline_frames = frames[baseline_mask]
        T = baseline_frames.shape[0]

        if T == 0:
            warnings.warn(
                f"{session_id}: no baseline frames after stimulus exclusion; skipping.",
                stacklevel=2,
            )
            return None

        if apply_log10:
            eps = float(np.float32(log10_eps))
            baseline_frames = np.log10(baseline_frames + eps).astype(np.float32)
            did_log10 = True
        else:
            baseline_frames = baseline_frames.astype(np.float32)
            did_log10 = False
            eps = None

        _, H, W = baseline_frames.shape

        attrs = sanitize_attrs({
            "stage":             BASELINE_STAGE_EXTRACTED,
            "session_id":        session_id,
            "frame_rate":        fps,
            "source_scan_file":  os.path.basename(str(scan_path)),
            "subject_tag":       scan_meta["subject_tag"],
            "scan_date":         scan_meta["date"],
            "n_total_frames":    int(T_total),
            "n_baseline_frames": int(T),
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
            name=session_id,
        )
        da.to_netcdf(out_path)
        pct = 100.0 * T / max(1, T_total)
        print(f"  Saved {out_path.name}: {T}/{T_total} baseline frames ({pct:.1f}%)")
        return str(out_path)

    except Exception as exc:
        warnings.warn(
            f"Skipping {os.path.basename(str(scan_path))}: {exc}",
            stacklevel=2,
        )
        return None


def process_all_baseline_files_mouse(
    data_directory: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    excel_path: str | os.PathLike[str] | None = None,
    *,
    overwrite: bool = False,
    apply_log10: bool = True,
    log10_eps: float = 1e-6,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """
    Extract baseline stage files for all .source.scan sessions in a directory.

    Stimulus frames are excluded using timing parsed from the PerifUS Excel
    spreadsheet (matched per session via the HDF5 ``scanMetaData/Subject_tag``).
    If no Excel path is given, or if no matching row is found, all frames are
    treated as baseline with a warning.

    Parameters
    ----------
    data_directory : str or Path
        Directory containing ``*.source.scan`` files.
    output_dir : str or Path
        Directory for output ``.nc`` files.
    excel_path : str or Path or None
        Path to ``Summary PeriFus experiments.xlsx``. Optional.
    overwrite : bool
    apply_log10 : bool
    log10_eps : float
    exclude_ids : set of str or None
        Session IDs (stem of .source.scan file) to skip.

    Returns
    -------
    list of str
        Paths to successfully written ``.nc`` files.
    """
    scan_files = sorted(glob.glob(os.path.join(str(data_directory), "*.source.scan")))
    if not scan_files:
        print(f"No .source.scan files found in {data_directory}")
        return []

    print(f"Found {len(scan_files)} .source.scan files")

    timing_lookup: dict[str, dict | None] = {}
    if excel_path is not None and Path(str(excel_path)).exists():
        try:
            timing_lookup = parse_perifus_excel(excel_path)
            print(f"  Loaded stimulus timing for {len(timing_lookup)} subjects from Excel")
        except Exception as exc:
            warnings.warn(f"Could not parse Excel metadata: {exc}", stacklevel=2)
    else:
        if excel_path is not None:
            warnings.warn(f"Excel file not found: {excel_path} — treating all frames as baseline", stacklevel=2)

    saved_paths: list[str] = []

    for scan_path in scan_files:
        session_id = Path(scan_path).stem
        if exclude_ids and session_id in exclude_ids:
            print(f"  Skipping excluded session {session_id}")
            continue

        # Load the scan once; reuse the result for both subject-tag lookup and extraction
        timing: dict | None = None
        preloaded: tuple | None = None
        try:
            preloaded = load_source_scan(scan_path)
            _, _, _, meta = preloaded
            subject_tag = meta["subject_tag"]
            if timing_lookup:
                if subject_tag in timing_lookup:
                    timing = timing_lookup[subject_tag]
                    if timing is None:
                        warnings.warn(
                            f"{session_id}: subject {subject_tag!r} found in Excel but pattern "
                            f"could not be parsed — treating all frames as baseline.",
                            stacklevel=2,
                        )
                else:
                    warnings.warn(
                        f"{session_id}: subject {subject_tag!r} not in Excel — "
                        f"treating all frames as baseline.",
                        stacklevel=2,
                    )
        except Exception as exc:
            warnings.warn(
                f"{session_id}: could not read HDF5 file ({exc}) — skipping.",
                stacklevel=2,
            )
            continue

        out = extract_and_save_baseline_mouse(
            scan_path=scan_path,
            output_dir=output_dir,
            timing=timing,
            apply_log10=apply_log10,
            log10_eps=log10_eps,
            overwrite=overwrite,
            _preloaded=preloaded,
        )
        if out is not None:
            saved_paths.append(out)

    print(f"Extracted baseline from {len(saved_paths)}/{len(scan_files)} sessions")
    return saved_paths