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
from scipy.ndimage import label, binary_fill_holes
from .morphology import imclose, strel_disk

from .io_common import (
    STAGE_REORIENTED_RESIZED,
    derive_session_id_from_path,
    sanitize_attrs,
)

STAGE_TISSUE_MASK = "tissue_mask"


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def compute_tissue_masks(
    frames: np.ndarray,
    *,
    method: str = "imclose",
    vessel_intensity_percentile: float = 75.0,
    closing_radius: int = 3,
    min_component_pixels: int = 50,
    cv_percentile: float = 60.0,
    roi_intensity_percentile: float = 30.0,
    roi_closing_radius: int = 15,
) -> dict:
    """
    Segment baseline fUS frames into vessel and parenchyma masks.

    Parameters
    ----------
    frames : np.ndarray, shape (T, H, W)
    method : "imclose" | "cv_threshold" | "roi"
        imclose      — threshold temporal mean, then morphological closing.
        cv_threshold — require both high mean AND high CV (std/|mean|);
                       vessels are bright *and* temporally variable,
                       bone/skull is bright but static.
        roi          — cohesive filled region: low-percentile threshold to
                       capture all brain tissue, binary_fill_holes to remove
                       internal holes, large morphological closing to bridge
                       gaps and smooth the boundary, then keep only the largest
                       connected component.
    vessel_intensity_percentile : float
        Percentile of the mean map used as intensity threshold (imclose/cv_threshold).
    closing_radius : int
        Disk radius for morphological closing (imclose method only).
    min_component_pixels : int
        Drop connected components smaller than this (imclose/cv_threshold).
    cv_percentile : float
        Percentile of the CV map used as the CV threshold (cv_threshold method).
    roi_intensity_percentile : float
        Percentile threshold for initial tissue detection in roi method.
    roi_closing_radius : int
        Disk radius for the large closing pass in roi method.
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"frames must be shape (T, H, W), got {arr.shape}")

    mean_map = arr.mean(axis=0)

    finite_mean = mean_map[np.isfinite(mean_map)]
    if finite_mean.size == 0:
        H, W  = mean_map.shape
        dummy = np.zeros((H, W), dtype=bool)
        return {
            "vessel_mask":         dummy,
            "parenchyma_mask":     ~dummy,
            "mean_map":            mean_map.astype(np.float32),
            "intensity_threshold": float("nan"),
            "n_vessel_pixels":     0,
            "n_parenchyma_pixels": int((~dummy).sum()),
            "method":              "fallback_empty",
        }

    intensity_thr = float(np.percentile(finite_mean, vessel_intensity_percentile))

    if method == "cv_threshold":
        std_map  = arr.std(axis=0)
        cv_map   = std_map / (np.abs(mean_map) + 1e-8)
        finite_cv = cv_map[np.isfinite(cv_map)]
        cv_thr   = float(np.percentile(finite_cv, cv_percentile))
        vessel_mask = (mean_map > intensity_thr) & (cv_map > cv_thr)
    elif method == "roi":
        roi_thr = float(np.percentile(finite_mean, roi_intensity_percentile))
        vessel_mask = mean_map > roi_thr
        vessel_mask = binary_fill_holes(vessel_mask)
        if roi_closing_radius > 0:
            vessel_mask = imclose(vessel_mask, strel_disk(roi_closing_radius))
        vessel_mask = binary_fill_holes(vessel_mask)
        # keep only the largest connected component
        labeled, n_components = label(vessel_mask)
        if n_components > 1:
            sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
            largest = int(np.argmax(sizes)) + 1
            vessel_mask = labeled == largest
        vessel_mask = vessel_mask.astype(bool)
    else:
        vessel_mask = mean_map > intensity_thr
        if closing_radius > 0:
            vessel_mask = imclose(vessel_mask, strel_disk(closing_radius))

    if method not in ("roi",) and min_component_pixels > 0:
        labeled, n_components = label(vessel_mask)
        for comp_id in range(1, n_components + 1):
            if (labeled == comp_id).sum() < min_component_pixels:
                vessel_mask[labeled == comp_id] = False

    vessel_mask     = vessel_mask.astype(bool)
    parenchyma_mask = ~vessel_mask

    return {
        "vessel_mask":         vessel_mask,
        "parenchyma_mask":     parenchyma_mask,
        "mean_map":            mean_map.astype(np.float32),
        "intensity_threshold": intensity_thr,
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
    method: str = "imclose",
    vessel_intensity_percentile: float = 75.0,
    closing_radius: int = 3,
    min_component_pixels: int = 50,
    cv_percentile: float = 60.0,
    roi_intensity_percentile: float = 30.0,
    roi_closing_radius: int = 15,
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
    vessel_intensity_percentile, closing_radius, min_component_pixels
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
            method=method,
            vessel_intensity_percentile=vessel_intensity_percentile,
            closing_radius=closing_radius,
            min_component_pixels=min_component_pixels,
            cv_percentile=cv_percentile,
            roi_intensity_percentile=roi_intensity_percentile,
            roi_closing_radius=roi_closing_radius,
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
            },
            coords={"x": x_coords, "y": y_coords},
            attrs=sanitize_attrs({
                "stage":                        STAGE_TISSUE_MASK,
                "session_id":                   session_id,
                "input_stage":                  stage_in,
                "vessel_intensity_percentile":  vessel_intensity_percentile,
                "closing_radius":               closing_radius,
                "min_component_pixels":         min_component_pixels,
                "roi_intensity_percentile":     roi_intensity_percentile,
                "roi_closing_radius":           roi_closing_radius,
                "intensity_threshold":          result["intensity_threshold"],
                "n_vessel_pixels":              result["n_vessel_pixels"],
                "n_parenchyma_pixels":          result["n_parenchyma_pixels"],
                "method":                       result["method"],
                "frame_rate":                   da.attrs.get("frame_rate", 2.5),
                "source_fus_file":              da.attrs.get("source_fus_file", ""),
            }),
        )
        ds.to_netcdf(out_path)
        outputs.append(str(out_path))

    return outputs