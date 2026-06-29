"""
loading.py
----------
I/O functions for constructing Session objects from .nc files on disk.

All downstream code operates on Session objects; this module is the only
place that touches the filesystem or xarray.

Expected file layout
--------------------
Standardized sessions::

    <standardized_dir>/
        baseline_<session_id>_unfiltered_standardized_zscore.nc
        ...

Vessel masks::

    <mask_dir>/
        tissue_mask_<session_id>.nc
        ...

Each standardized .nc file is an xr.Dataset with:
  - ``frames``   : DataArray (time, x, y), float32, z-scored
  - ``mean_map`` : DataArray (x, y)
  - ``std_map``  : DataArray (x, y)
  - attrs including ``session_id``, ``frame_rate``, ``zscored``

Each tissue mask .nc file is an xr.Dataset with:
  - ``vessel_mask`` : DataArray (x, y), bool
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from fuspredict.preprocessing.io import STAGE_STANDARDIZED, derive_session_id_from_path
from fuspredict.data.session import Session

# Glob pattern for standardized baseline files
_STANDARDIZED_GLOB = f"baseline_*_unfiltered_{STAGE_STANDARDIZED}.nc"


# ---------------------------------------------------------------------------
# Single-session loader
# ---------------------------------------------------------------------------

def load_session(
    nc_path: str | Path,
    mask_dir: str | Path | None = None,
) -> Session:
    """
    Load one standardized session from a ``.nc`` file.

    Parameters
    ----------
    nc_path : str or Path
        Path to a standardized baseline ``.nc`` file (xr.Dataset).
    mask_dir : str or Path or None
        Directory containing tissue mask files named
        ``tissue_mask_<session_id>.nc``. Pass None to skip mask loading;
        ``Session.vessel_mask`` will be None.

    Returns
    -------
    Session
        Populated Session object with frames, fps, optional vessel_mask,
        and metadata from the dataset attrs.

    Raises
    ------
    FileNotFoundError
        If ``nc_path`` does not exist.
    KeyError
        If the dataset does not contain a ``frames`` variable.
    ValueError
        If ``frames`` is not 3-D or is not z-scored.
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(f"Session file not found: {nc_path}")

    ds = xr.open_dataset(nc_path)

    if "frames" not in ds:
        raise KeyError(f"{nc_path.name}: dataset has no 'frames' variable")

    frames_da = ds["frames"]
    if frames_da.ndim != 3:
        raise ValueError(
            f"{nc_path.name}: 'frames' must be 3-D (time, x, y), "
            f"got shape {frames_da.shape}"
        )

    # Verify z-scored flag (stored as "True"/"False" string by sanitize_attrs)
    zscored_raw = ds.attrs.get("zscored", "False")
    if str(zscored_raw).strip().lower() != "true":
        raise ValueError(
            f"{nc_path.name}: dataset is not z-scored "
            f"(zscored attr = {zscored_raw!r})"
        )

    frames = frames_da.values.astype(np.float32)

    session_id = str(
        ds.attrs.get("session_id") or derive_session_id_from_path(nc_path)
    )
    fps = float(ds.attrs.get("frame_rate", 2.5))

    # Collect metadata: all attrs except fields already on Session
    _skip = {"session_id", "frame_rate", "zscored"}
    metadata = {k: v for k, v in ds.attrs.items() if k not in _skip}

    vessel_mask = _load_vessel_mask(session_id, mask_dir)

    return Session(
        id=session_id,
        frames=frames,
        fps=fps,
        vessel_mask=vessel_mask,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Bulk loader
# ---------------------------------------------------------------------------

def load_sessions(
    standardized_dir: str | Path,
    mask_dir: str | Path | None = None,
    exclude_ids: list[str] | None = None,
) -> list[Session]:
    """
    Load all standardized sessions from a directory.

    Discovers files matching ``baseline_*_unfiltered_standardized_zscore.nc``,
    loads each one, and returns the successful results. Sessions that fail to
    load emit a warning and are skipped rather than raising an exception.

    Parameters
    ----------
    standardized_dir : str or Path
        Directory containing standardized ``.nc`` session files.
    mask_dir : str or Path or None
        Directory containing tissue mask files. Passed to :func:`load_session`.
    exclude_ids : list of str or None
        Session IDs to skip entirely (e.g. sessions with known quality issues).

    Returns
    -------
    list of Session
        Sessions sorted by ``Session.id``. Empty list if none are found.
    """
    standardized_dir = Path(standardized_dir)
    if not standardized_dir.is_dir():
        raise FileNotFoundError(
            f"Standardized directory not found: {standardized_dir}"
        )

    nc_paths = sorted(standardized_dir.glob(_STANDARDIZED_GLOB))
    if not nc_paths:
        warnings.warn(
            f"No files matching '{_STANDARDIZED_GLOB}' found in {standardized_dir}",
            stacklevel=2,
        )
        return []

    excluded = set(exclude_ids or [])
    sessions: list[Session] = []

    for nc_path in nc_paths:
        # Quick ID check before paying the cost of opening the file
        candidate_id = derive_session_id_from_path(nc_path)
        if candidate_id in excluded:
            continue

        try:
            session = load_session(nc_path, mask_dir=mask_dir)
        except Exception as exc:
            warnings.warn(
                f"Skipping {nc_path.name}: {exc}",
                stacklevel=2,
            )
            continue

        if session.id in excluded:
            continue

        sessions.append(session)

    sessions.sort(key=lambda s: s.id)
    return sessions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_vessel_mask(
    session_id: str,
    mask_dir: str | Path | None,
) -> np.ndarray | None:
    """
    Load the vessel mask for ``session_id`` from ``mask_dir``.

    Returns None (without raising) if mask_dir is None or the mask file
    does not exist.
    """
    if mask_dir is None:
        return None

    mask_path = Path(mask_dir) / f"tissue_mask_{session_id}.nc"
    if not mask_path.exists():
        return None

    try:
        ds = xr.open_dataset(mask_path)
        if "vessel_mask" not in ds:
            warnings.warn(
                f"Mask file {mask_path.name} has no 'vessel_mask' variable; "
                "ignoring.",
                stacklevel=3,
            )
            return None
        return ds["vessel_mask"].values.astype(bool)
    except Exception as exc:
        warnings.warn(
            f"Could not load mask {mask_path.name}: {exc}; ignoring.",
            stacklevel=3,
        )
        return None
