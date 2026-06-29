"""
data_loader.py
--------------
Load functional ultrasound (fUS) sessions from .mat files into xarray DataArrays.

Supports both legacy .mat (v5/v6, via scipy.io) and modern .mat (v7.3 HDF5, via h5py).
The returned DataArray has labelled dimensions (time, x, y) and carries session
metadata as attributes. Data is always returned raw — z-scoring is handled
downstream by preprocessing.py.

Usage
-----
    from data_loader import load_session, load_sessions

    # Single session
    da = load_session("path/to/session_01.mat", session_id="session_01")

    # All sessions from a directory
    sessions = load_sessions("path/to/data/", frame_rate=2.5)
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io
import h5py
import xarray as xr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_mat_v73(path: str) -> bool:
    """Return True if the .mat file is HDF5-based (MATLAB v7.3+)."""
    with open(path, "rb") as f:
        magic = f.read(8)
    return magic == b"\x89HDF\r\n\x1a\n"


def _load_mat_legacy(path: str) -> dict:
    """Load a legacy .mat file (v5/v6) with scipy.io."""
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def _load_mat_v73(path: str) -> dict:
    """Load a v7.3 .mat file (HDF5-based) with h5py."""
    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            # HDF5 .mat stores arrays transposed relative to MATLAB convention
            data[key] = np.array(f[key]).T
    return data


def _extract_frames(mat_data: dict, frames_key: Optional[str] = None) -> np.ndarray:
    """
    Pull the frame array out of the loaded .mat dict.

    Tries common key names if frames_key is not specified.
    Raises ValueError with a helpful message if nothing is found.

    Output shape: (T, H, W) — time first.
    """
    candidates = [frames_key] if frames_key else [
        "PDdata", "pddata", "frames", "data", "fUS", "fus", "signal"
    ]

    for key in candidates:
        if key and key in mat_data:
            arr = np.array(mat_data[key], dtype=np.float32)
            if arr.ndim != 3:
                raise ValueError(
                    f"Array under key '{key}' has shape {arr.shape}. "
                    f"Expected 3D (H, W, T) or (T, H, W)."
                )
            # MATLAB stores as (H, W, T) — detect by time being the smallest axis
            if arr.shape[2] < arr.shape[0] and arr.shape[2] < arr.shape[1]:
                arr = arr.transpose(2, 0, 1)
            return arr

    raise ValueError(
        f"Could not find frame data in .mat file. "
        f"Keys present: {list(mat_data.keys())}. "
        f"Pass frames_key='<your_key>' explicitly."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_session(
    path: str,
    session_id: Optional[str] = None,
    frame_rate: float = 2.5,
    frames_key: Optional[str] = None,
) -> xr.DataArray:
    """
    Load a single fUS session from a .mat file into an xarray DataArray.

    Data is returned raw (no normalization). Pass the result to
    preprocessing.zscore_session() to normalize.

    Parameters
    ----------
    path : str
        Path to the .mat file.
    session_id : str, optional
        Human-readable session identifier stored as metadata.
        Defaults to the filename stem.
    frame_rate : float
        Acquisition frame rate in Hz. Used to build the time coordinate.
        Default: 2.5 Hz.
    frames_key : str, optional
        Key in the .mat file that holds the frame array.
        If None, common names are tried automatically.

    Returns
    -------
    xr.DataArray
        Shape: (time, x, y)
        Coordinates:
            time  — seconds from session start (float)
            x     — pixel index along first spatial axis
            y     — pixel index along second spatial axis
        Attributes:
            session_id  — session identifier string
            frame_rate  — acquisition rate in Hz
            n_frames    — number of frames
            height      — spatial height in pixels
            width       — spatial width in pixels
            zscored     — always False; set to True by preprocessing.zscore_session()
            source_file — absolute path to the source .mat file
    """
    path = str(path)

    if session_id is None:
        session_id = Path(path).stem

    # Load raw arrays from disk
    if _is_mat_v73(path):
        mat_data = _load_mat_v73(path)
    else:
        mat_data = _load_mat_legacy(path)

    frames = _extract_frames(mat_data, frames_key=frames_key)  # (T, H, W)
    T, H, W = frames.shape

    # Build coordinates
    time_coords = np.arange(T) / frame_rate   # seconds
    x_coords    = np.arange(H)
    y_coords    = np.arange(W)

    return xr.DataArray(
        data=frames,
        dims=["time", "x", "y"],
        coords={"time": time_coords, "x": x_coords, "y": y_coords},
        attrs={
            "session_id":  session_id,
            "frame_rate":  frame_rate,
            "n_frames":    T,
            "height":      H,
            "width":       W,
            "zscored":     False,
            "source_file": os.path.abspath(path),
        },
        name=session_id,
    )


def load_sessions(
    data_dir: str,
    frame_rate: float = 2.5,
    frames_key: Optional[str] = None,
    pattern: str = "*.mat",
) -> list[xr.DataArray]:
    """
    Load all .mat sessions from a directory.

    Parameters
    ----------
    data_dir : str
        Directory containing .mat session files.
    frame_rate : float
        Acquisition frame rate in Hz. Applied to all sessions.
    frames_key : str, optional
        Key in .mat files holding the frame array (same key assumed for all).
    pattern : str
        Glob pattern for matching files. Default: '*.mat'.

    Returns
    -------
    list of xr.DataArray
        One DataArray per session, sorted by filename.
    """
    data_dir = Path(data_dir)
    mat_files = sorted(data_dir.glob(pattern))

    if not mat_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir}."
        )

    sessions = []
    for mat_file in mat_files:
        try:
            da = load_session(
                path=str(mat_file),
                frame_rate=frame_rate,
                frames_key=frames_key,
            )
            sessions.append(da)
            print(f"Loaded {mat_file.name}: {da.sizes}")
        except Exception as e:
            warnings.warn(f"Failed to load {mat_file.name}: {e}")

    return sessions