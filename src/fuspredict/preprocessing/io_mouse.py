"""
io_mouse.py
-----------
Raw .source.scan extraction for the mouse pipeline: HDF5 loading, PerifUS
Excel stimulus-timing parsing, and baseline/task extraction mirroring the
primate pipeline in :mod:`fuspredict.preprocessing.io_primate`.

Author: Brynn Harris-Shanks, 2026
With adaptations from code by Leo Sperber, 2025
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from fuspredict.preprocessing.io_common import (
    BASELINE_STAGE_EXTRACTED,
    TASK_STAGE_EXTRACTED,
    sanitize_attrs,
    save_label_sidecar,
)


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

        has_real_timing = timing is not None
        if has_real_timing:
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
            has_real_timing=has_real_timing,
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
            "has_real_timing":   has_real_timing,
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

    Also writes ``<output_dir>/sessions_with_real_baseline_timing.json``: a
    list of session IDs whose baseline mask came from real parsed Excel
    timing, as opposed to the "no timing found" fallback that treats the
    whole recording as baseline.

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
    real_timing_ids: list[str] = []

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
            if timing is not None:
                real_timing_ids.append(session_id)

    print(f"Extracted baseline from {len(saved_paths)}/{len(scan_files)} sessions")
    print(f"  {len(real_timing_ids)}/{len(saved_paths)} sessions have real (Excel-derived) baseline timing")

    manifest_path = Path(output_dir) / "sessions_with_real_baseline_timing.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(sorted(real_timing_ids), f, indent=2)
    print(f"  Wrote {manifest_path}")

    return saved_paths


def extract_and_save_task_mouse(
    scan_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    timing: dict,
    *,
    apply_log10: bool = True,
    log10_eps: float = 1e-6,
    overwrite: bool = False,
    _preloaded: tuple | None = None,
) -> str | None:
    """
    Extract non-baseline (task/stimulus) frames from a .source.scan file and save as .nc.

    Mirrors :func:`extract_and_save_baseline_mouse`, saving the complement of
    the resting-state mask. Requires real timing (unlike the baseline
    extractor, there is no "treat everything as task" fallback).

    Parameters
    ----------
    scan_path : str or Path
        Path to a ``*.source.scan`` HDF5 file.
    output_dir : str or Path
        Directory for the output ``.nc`` file.
    timing : dict
        Stimulus timing from :func:`parse_perifus_excel`.
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
    out_path = out_dir / f"task_{session_id}_{TASK_STAGE_EXTRACTED}.nc"

    if out_path.exists() and not overwrite:
        print(f"  Skipping {session_id} (already exists)")
        return str(out_path)

    try:
        if _preloaded is not None:
            frames, fps, frame_times, scan_meta = _preloaded
        else:
            frames, fps, frame_times, scan_meta = load_source_scan(scan_path)
        T_total = frames.shape[0]

        baseline_mask = extract_baseline_mask_from_timing(frame_times, timing)
        task_mask     = ~baseline_mask
        task_frames   = frames[task_mask]
        T             = task_frames.shape[0]

        if T == 0:
            warnings.warn(
                f"{session_id}: no task frames found; skipping.",
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

        _, H, W = task_frames.shape

        attrs = sanitize_attrs({
            "stage":             TASK_STAGE_EXTRACTED,
            "session_id":        session_id,
            "frame_rate":        fps,
            "source_scan_file":  os.path.basename(str(scan_path)),
            "subject_tag":       scan_meta["subject_tag"],
            "scan_date":         scan_meta["date"],
            "n_total_frames":    int(T_total),
            "n_task_frames":     int(T),
            "did_log10":         did_log10,
            "log10_eps":         eps,
            "zscored":           False,
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
            name=session_id,
        )
        da.to_netcdf(out_path)
        pct = 100.0 * T / max(1, T_total)
        print(f"  Saved {out_path.name}: {T}/{T_total} task frames ({pct:.1f}%)")
        return str(out_path)

    except Exception as exc:
        warnings.warn(
            f"Skipping {os.path.basename(str(scan_path))}: {exc}",
            stacklevel=2,
        )
        return None


def process_all_task_files_mouse(
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
    Extract task (non-baseline) stage files for all .source.scan sessions in a directory.

    Sessions with no real Excel-derived timing are skipped (there is no
    "everything is task" fallback, since without timing baseline and task
    frames can't be distinguished).

    Parameters
    ----------
    data_directory : str or Path
        Directory containing ``*.source.scan`` files.
    output_dir : str or Path
        Directory for output ``.nc`` files.
    excel_path : str or Path or None
        Path to ``Summary PeriFus experiments.xlsx``.
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
            warnings.warn(f"Excel file not found: {excel_path}", stacklevel=2)

    saved_paths: list[str] = []

    for scan_path in scan_files:
        session_id = Path(scan_path).stem
        if exclude_ids and session_id in exclude_ids:
            print(f"  Skipping excluded session {session_id}")
            continue

        try:
            preloaded = load_source_scan(scan_path)
            _, _, _, meta = preloaded
            subject_tag = meta["subject_tag"]
        except Exception as exc:
            warnings.warn(
                f"{session_id}: could not read HDF5 file ({exc}) — skipping.",
                stacklevel=2,
            )
            continue

        timing = timing_lookup.get(subject_tag) if timing_lookup else None
        if timing is None:
            warnings.warn(
                f"{session_id}: subject {subject_tag!r} has no parsed Excel timing — "
                f"skipping task extraction (cannot distinguish task frames without it).",
                stacklevel=2,
            )
            continue

        out = extract_and_save_task_mouse(
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

    print(f"Extracted task frames from {len(saved_paths)}/{len(scan_files)} sessions")
    return saved_paths
