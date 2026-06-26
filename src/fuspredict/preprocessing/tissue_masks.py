"""
tissue_masks.py
---------------
Vessel vs parenchyma segmentation for fUS baseline sessions.

Each output is an xr.Dataset containing:
  - vessel_mask     : bool (x, y) — True where vessel signal detected
  - parenchyma_mask : bool (x, y) — True where parenchyma (not vessel)
  - mean_map        : float32 (x, y) — temporal mean of input frames
  - cv_map          : float32 (x, y) — coefficient of variation (std / |mean|)

Loading a saved mask:
    import xarray as xr
    ds = xr.open_dataset("tissue_masks/tissue_mask_<session_id>.nc")
    vessel_mask = ds["vessel_mask"].values.astype(bool)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

from .io import (
    STAGE_REORIENTED_RESIZED,
    derive_session_id_from_path,
)

STAGE_TISSUE_MASK = "tissue_mask"


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def compute_tissue_masks(
    frames: np.ndarray,
    *,
    vessel_intensity_percentile: float = 60.0,
    vessel_cv_percentile: float = 40.0,
    min_vessel_pixels: int = 10,
) -> dict:
    """
    Segment baseline fUS frames into vessel and parenchyma masks.

    Power Doppler signal is high in vessels (high mean intensity, high
    coefficient of variation from cardiac pulsatility) and low in parenchyma.
    A pixel is classified as a vessel if it exceeds both the intensity and CV
    thresholds; everything else is parenchyma.

    Parameters
    ----------
    frames : np.ndarray, shape (T, H, W)
        Reoriented/resized baseline frames (log10 space).
    vessel_intensity_percentile : float
        Pixels with temporal mean above this percentile are vessel candidates.
    vessel_cv_percentile : float
        Pixels with temporal CV (std / |mean|) above this percentile are vessel
        candidates. Combined with the intensity threshold via AND.
    min_vessel_pixels : int
        If fewer vessel pixels are found after AND thresholding, fall back to
        intensity-only threshold.

    Returns
    -------
    dict with keys:
        vessel_mask         : bool (H, W)
        parenchyma_mask     : bool (H, W)
        mean_map            : float32 (H, W)
        cv_map              : float32 (H, W)
        intensity_threshold : float
        cv_threshold        : float
        n_vessel_pixels     : int
        n_parenchyma_pixels : int
        method              : str
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"frames must be shape (T, H, W), got {arr.shape}")

    mean_map = arr.mean(axis=0)
    std_map  = arr.std(axis=0)
    eps      = float(np.finfo(np.float32).eps)
    cv_map   = std_map / (np.abs(mean_map) + eps)

    finite_mean = mean_map[np.isfinite(mean_map)]
    finite_cv   = cv_map[np.isfinite(cv_map)]

    if finite_mean.size == 0 or finite_cv.size == 0:
        H, W  = mean_map.shape
        dummy = np.zeros((H, W), dtype=bool)
        return {
            "vessel_mask":         dummy,
            "parenchyma_mask":     ~dummy,
            "mean_map":            mean_map,
            "cv_map":              cv_map,
            "intensity_threshold": float("nan"),
            "cv_threshold":        float("nan"),
            "n_vessel_pixels":     0,
            "n_parenchyma_pixels": int((~dummy).sum()),
            "method":              "fallback_empty",
        }

    intensity_thr = float(np.percentile(finite_mean, vessel_intensity_percentile))
    cv_thr        = float(np.percentile(finite_cv,   vessel_cv_percentile))

    vessel_mask = (mean_map >= intensity_thr) & (cv_map >= cv_thr)
    method      = "intensity_and_cv"

    if int(vessel_mask.sum()) < min_vessel_pixels:
        vessel_mask = mean_map >= intensity_thr
        method      = "intensity_only_fallback"

    parenchyma_mask = ~vessel_mask

    return {
        "vessel_mask":         vessel_mask.astype(bool),
        "parenchyma_mask":     parenchyma_mask.astype(bool),
        "mean_map":            mean_map.astype(np.float32),
        "cv_map":              cv_map.astype(np.float32),
        "intensity_threshold": intensity_thr,
        "cv_threshold":        cv_thr,
        "n_vessel_pixels":     int(vessel_mask.sum()),
        "n_parenchyma_pixels": int(parenchyma_mask.sum()),
        "method":              method,
    }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def segment_all_sessions(
    in_nc_paths: list[str],
    out_dir: str | os.PathLike[str],
    *,
    vessel_intensity_percentile: float = 60.0,
    vessel_cv_percentile: float = 40.0,
    min_vessel_pixels: int = 10,
    overwrite: bool = False,
) -> list[str]:
    """
    Compute and save tissue masks for a list of reoriented baseline .nc files.

    Parameters
    ----------
    in_nc_paths : list of str
        Paths to reoriented/resized baseline .nc session files.
    out_dir : path-like
        Directory to write tissue mask .nc files.
    vessel_intensity_percentile, vessel_cv_percentile, min_vessel_pixels
        Forwarded to compute_tissue_masks.
    overwrite : bool
        If False, skip sessions whose output file already exists.

    Returns
    -------
    list of str
        Paths to written tissue mask .nc files.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []
    for in_path_str in in_nc_paths:
        in_path = Path(in_path_str)
        da      = xr.open_dataarray(in_path)

        stage_in = str(da.attrs.get("stage", ""))
        if stage_in != STAGE_REORIENTED_RESIZED:
            raise ValueError(
                f"{in_path.name}: tissue mask requires stage "
                f"'{STAGE_REORIENTED_RESIZED}', got '{stage_in}'."
            )

        session_id = da.attrs.get("session_id") or derive_session_id_from_path(in_path)
        out_path   = out_root / f"tissue_mask_{session_id}.nc"

        if out_path.exists() and not overwrite:
            print(f"  Skipping {session_id} (already exists)")
            outputs.append(str(out_path))
            continue

        result = compute_tissue_masks(
            da.values,
            vessel_intensity_percentile=vessel_intensity_percentile,
            vessel_cv_percentile=vessel_cv_percentile,
            min_vessel_pixels=min_vessel_pixels,
        )

        H, W       = result["mean_map"].shape
        pct_vessel = 100.0 * result["n_vessel_pixels"] / max(1, H * W)
        print(
            f"  {session_id}: vessel={result['n_vessel_pixels']} px "
            f"({pct_vessel:.1f}%), parenchyma={result['n_parenchyma_pixels']} px "
            f"| method={result['method']}"
        )

        x_coords = da.coords["x"].values
        y_coords = da.coords["y"].values

        ds = xr.Dataset(
            {
                "vessel_mask":     xr.DataArray(result["vessel_mask"],     dims=["x", "y"]),
                "parenchyma_mask": xr.DataArray(result["parenchyma_mask"], dims=["x", "y"]),
                "mean_map":        xr.DataArray(result["mean_map"],        dims=["x", "y"]),
                "cv_map":          xr.DataArray(result["cv_map"],          dims=["x", "y"]),
            },
            coords={"x": x_coords, "y": y_coords},
            attrs={
                "stage":                        STAGE_TISSUE_MASK,
                "session_id":                   session_id,
                "input_stage":                  stage_in,
                "vessel_intensity_percentile":  vessel_intensity_percentile,
                "vessel_cv_percentile":         vessel_cv_percentile,
                "min_vessel_pixels":            min_vessel_pixels,
                "intensity_threshold":          result["intensity_threshold"],
                "cv_threshold":                 result["cv_threshold"],
                "n_vessel_pixels":              result["n_vessel_pixels"],
                "n_parenchyma_pixels":          result["n_parenchyma_pixels"],
                "method":                       result["method"],
                "frame_rate":                   da.attrs.get("frame_rate", 2.5),
                "source_fus_file":              da.attrs.get("source_fus_file", ""),
            },
        )
        ds.to_netcdf(out_path)
        outputs.append(str(out_path))

    return outputs