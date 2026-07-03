"""
active_period_analysis.py
-------------------------
Orchestrator for the active-period ROI and % CBV analysis.

For each session:
  1. Load baseline standardized .nc → mean_map, std_map (log10 baseline stats)
  2. Load task reoriented .nc → log10 task frames (112×112, geometry-corrected)
  3. Compute activation delta map and auto-select ROI
  4. Convert task frames to % CBV using baseline mean_map as reference
  5. Compute ROI-averaged signal and σ-crossing times
  6. Save figures to derivatives/active_period_analysis/<subject>/

Usage
-----
    python scripts/active_period_analysis.py
    python scripts/active_period_analysis.py --config config.yml
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from fuspredict.analysis.active_period import (
    activation_delta,
    auto_roi,
    baseline_roi_stats,
    roi_signal,
    sigma_crossings,
    zscore_frames,
)
from fuspredict.analysis.active_period_plots import (
    fig_activation_delta,
    fig_roi_timeseries,
    fig_transition_analysis,
)
from fuspredict.data.loading import load_session
from fuspredict.project import find_repo_root, load_project_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AUTO_ROI_N_PIXELS = 125
DELTA_WINDOW      = 20   # task frames averaged for activation delta
TIMESERIES_WINDOW = 15.0 # seconds shown before and after onset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active-period ROI and % CBV analysis.")
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Config filename inside config/ (default: config.yml).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Task frame loader
# ---------------------------------------------------------------------------

def load_task_frames(task_nc_path: Path, session_id: str) -> np.ndarray | None:
    """Load log10 task frames from a reoriented task .nc file.

    The data variable is named after the session_id (not 'frames') because
    the reorientation step reuses the baseline pipeline which preserves the
    original variable name.

    Returns shape (T, H, W) float32, or None if loading fails.
    """
    if not task_nc_path.exists():
        return None
    try:
        ds = xr.open_dataset(task_nc_path)
        if session_id in ds:
            return ds[session_id].values.astype(np.float32)
        # Fall back to first data variable if name differs
        data_vars = list(ds.data_vars)
        if data_vars:
            warnings.warn(
                f"{task_nc_path.name}: expected variable '{session_id}', "
                f"using '{data_vars[0]}' instead.",
                stacklevel=2,
            )
            return ds[data_vars[0]].values.astype(np.float32)
        warnings.warn(f"{task_nc_path.name}: no data variables found.", stacklevel=2)
        return None
    except Exception as exc:
        warnings.warn(f"Could not load {task_nc_path.name}: {exc}", stacklevel=2)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args      = parse_args()
    repo_root = find_repo_root()
    config    = load_project_config(repo_root, config_name=args.config)

    deriv_root = repo_root / config["paths"]["preprocessing"]
    out_root   = repo_root / "derivatives" / "active_period_analysis"
    subjects   = config["subjects"]["all"]

    for subject in subjects:
        exclude_ids = set(
            config["subjects"].get("sessions_to_exclude", {}).get(subject, [])
        )
        std_dir  = deriv_root / subject / "baseline_only_standardized"
        task_dir = deriv_root / subject / "task_only_reoriented_resized"

        if not std_dir.exists():
            warnings.warn(f"No baseline standardized dir for {subject}: {std_dir}")
            continue
        if not task_dir.exists():
            warnings.warn(f"No task reoriented dir for {subject}: {task_dir}")
            continue

        nc_paths = sorted(std_dir.glob("baseline_*_unfiltered_standardized_zscore.nc"))
        print(f"\n=== {subject}: {len(nc_paths)} sessions ===")

        for nc_path in nc_paths:
            session = load_session(nc_path)
            if session.id in exclude_ids:
                continue

            task_nc = task_dir / f"baseline_{session.id}_reoriented_resized.nc"
            task_frames = load_task_frames(task_nc, session.id)
            if task_frames is None:
                print(f"  {session.id}: no task frames found — skipping")
                continue

            print(f"  {session.id}: baseline={session.n_frames} frames, "
                  f"task={task_frames.shape[0]} frames")

            # Load mean_map and std_map from the standardized .nc
            ds          = xr.open_dataset(nc_path)
            mean_map    = ds["mean_map"].values.astype(np.float32)   # (H, W) log10
            std_map     = ds["std_map"].values.astype(np.float32)    # (H, W) log10
            ds.close()

            # ROI selection
            delta_map = activation_delta(mean_map, task_frames, window=DELTA_WINDOW)
            try:
                roi_mask = auto_roi(delta_map, n_pixels=AUTO_ROI_N_PIXELS)
            except ValueError as exc:
                print(f"  {session.id}: ROI selection failed ({exc}) — skipping")
                continue

            # Z-score task frames using baseline stats and compute ROI signal
            task_z = zscore_frames(task_frames, mean_map, std_map)
            signal = roi_signal(task_z, roi_mask)

            # Pre-onset signal from the end of the z-scored baseline frames
            pre_signal = roi_signal(session.frames, roi_mask)

            # Baseline σ stats and crossing times (empirical from baseline z-scored frames)
            bl_mean, bl_std = baseline_roi_stats(mean_map, std_map, roi_mask, session.frames)
            crossings       = sigma_crossings(signal, bl_mean, bl_std, session.fps)

            print(f"    ROI: {roi_mask.sum()} px | "
                  f"baseline σ={bl_std:.2f}σ")
            for n, t in crossings.items():
                t_str = f"{t:.2f} s" if t is not None else "—"
                print(f"    +{n}σ crossing: {t_str}")

            # Save outputs
            out_dir = out_root / subject / session.id
            out_dir.mkdir(parents=True, exist_ok=True)

            np.save(out_dir / "roi_mask.npy",     roi_mask)
            np.save(out_dir / "activation_delta.npy", delta_map)

            fig_activation_delta(
                delta_map, roi_mask, session.id,
                out_path=out_dir / "activation_delta.png",
            )
            fig_roi_timeseries(
                signal, bl_mean, bl_std, session.fps, session.id,
                out_path=out_dir / "roi_timeseries.png",
                window_s=TIMESERIES_WINDOW,
                pre_signal=pre_signal,
            )

            # Dizeux-style transition analysis
            ckpt_path = (
                repo_root / "derivatives" / "modeling" / "convlstm" / "unweighted"
                / session.id / "h1" / "model.pt"
            )
            baseline_log10 = session.frames * std_map[None] + mean_map[None]
            fig_transition_analysis(
                baseline_frames_log10=baseline_log10,
                task_frames_log10=task_frames,
                roi_mask=roi_mask,
                mean_map=mean_map,
                std_map=std_map,
                fps=session.fps,
                session_id=session.id,
                out_path=out_dir / "transition_analysis.png",
                checkpoint_path=ckpt_path if ckpt_path.exists() else None,
            )
            print(f"    Saved -> {out_dir.relative_to(repo_root)}/")


if __name__ == "__main__":
    main()
