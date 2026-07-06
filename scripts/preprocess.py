"""
scripts/preprocess.py
----------------------
Preprocessing pipeline entrypoint.

For each session, splits raw frames into baseline and active (task) periods,
then runs the same preprocessing steps on each independently:
  1. Frame extraction with optional log10 transform (baseline and active separately)
  2. Geometry — reorientation and resizing
  3. Standardization (z-scoring)
  4. Tissue segmentation (vessel vs parenchyma masks, from baseline reoriented frames)

Outputs per session:
  - baseline_only_standardized/<session>.nc
  - task_only_standardized/<session>.nc

Outputs are written under ``derivatives/preprocessing/<subject>/``.

Usage:
  python scripts/preprocess.py
  python scripts/preprocess.py --config config_mouse.yml
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from fuspredict.preprocessing.geometry import reorient_baseline_sessions
from fuspredict.preprocessing.io import (
    process_all_baseline_files,
    process_all_baseline_files_mouse,
    process_all_task_files,
)
from fuspredict.preprocessing.standardization import (
    standardize_stage_sessions,
    standardize_task_sessions_with_baseline_stats,
)
from fuspredict.preprocessing.tissue_masks import segment_all_sessions
from fuspredict.project import find_repo_root, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fUS preprocessing pipeline.")
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Config filename inside config/ (default: config.yml).",
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline extraction, reorientation, standardization, and tissue masks.",
    )
    stage.add_argument(
        "--task-only",
        action="store_true",
        help="Run only task extraction, reorientation, and standardization "
             "(requires baseline_only_standardized to already exist for baseline stats).",
    )
    return parser.parse_args()


def list_nc(directory: Path, exclude_ids: set[str] | None = None) -> list[str]:
    """
    List ``.nc`` files in ``directory``, optionally excluding session IDs.

    Glob all ``.nc`` outputs rather than using each stage's return value,
    which only includes files written in the current run (empty when
    overwrite=False).
    """
    paths = sorted(
        str(p) for p in Path(directory).glob("*.nc")
        if not p.stem.startswith("labels_")
    )
    if exclude_ids:
        paths = [p for p in paths if not any(sid in Path(p).stem for sid in exclude_ids)]
    return paths


def plot_vessel_masks(tissue_dir: Path, subject: str, out_dir: Path) -> None:
    mask_paths = sorted(tissue_dir.glob("tissue_mask_*.nc"))
    if not mask_paths:
        return

    ncols = 6
    nrows = math.ceil(len(mask_paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.array(axes).reshape(-1)

    for ax, path in zip(axes, mask_paths):
        ds = xr.open_dataset(path)
        mask = ds["vessel_mask"].values.astype(bool)
        session_id = ds.attrs.get("session_id", path.stem.replace("tissue_mask_", ""))
        pct = 100.0 * mask.sum() / mask.size
        ax.imshow(mask, origin="upper", cmap="gray", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{session_id}\n{pct:.0f}%", fontsize=6)
        ax.axis("off")
        ds.close()

    for ax in axes[len(mask_paths):]:
        ax.axis("off")

    fig.suptitle(f"Vessel masks — {subject} (% coverage)", fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vessel_masks_{subject}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Vessel mask figure saved: {out_path}")


def main() -> None:
    args = parse_args()
    run_baseline = not args.task_only
    run_task     = not args.baseline_only

    repo_root = find_repo_root()
    config = load_project_config(repo_root, config_name=args.config)

    deriv_root  = repo_root / config["paths"]["preprocessing"]
    source_root = repo_root / config["paths"]["sourcedata"]
    subjects    = config["subjects"]["all"]
    species     = config["subjects"].get("species", "monkey")

    base_cfg   = config["preprocessing"]["baseline"]
    geo_cfg    = config["preprocessing"]["geometry"]
    std_cfg    = config["preprocessing"]["standardization"]
    tissue_cfg = config["preprocessing"]["tissue_segmentation"]

    APPLY_LOG10 = base_cfg["apply_log10"]
    dir_suffix  = "" if APPLY_LOG10 else "_nolog"

    for subject in subjects:
        subj_deriv  = deriv_root / subject
        exclude_ids = set(config["subjects"].get("sessions_to_exclude", {}).get(subject, []))

        print(f"\n=== Processing subject {subject} (species={species}) ===")

        flip_ids    = set() if species == "mouse" else set(
            geo_cfg["flip_session_ids_by_subject"].get(subject, [])
        )
        subj_source = source_root / subject

        # ---------------------------------------------------------------
        # Stage 1 — Extract baseline frames
        # ---------------------------------------------------------------
        baseline_raw_dir = subj_deriv / f"baseline_only{dir_suffix}"
        if run_baseline:
            if species == "mouse":
                excel_path = repo_root / config["subjects"].get(
                    "excel_metadata",
                    "data/sourcedata/mouse/Summary PeriFus experiments.xlsx",
                )
                process_all_baseline_files_mouse(
                    data_directory=str(subj_source),
                    output_dir=str(baseline_raw_dir),
                    excel_path=str(excel_path),
                    overwrite=base_cfg["overwrite"],
                    apply_log10=APPLY_LOG10,
                    log10_eps=base_cfg["log10_eps"],
                    exclude_ids=exclude_ids,
                )
            else:
                process_all_baseline_files(
                    str(subj_source),
                    str(baseline_raw_dir),
                    overwrite=base_cfg["overwrite"],
                    apply_log10=APPLY_LOG10,
                    log10_eps=base_cfg["log10_eps"],
                    exclude_ids=exclude_ids,
                )
        baseline_raw_paths = list_nc(baseline_raw_dir, exclude_ids)
        print(f"  Baseline extracted: {len(baseline_raw_paths)} sessions")

        # ---------------------------------------------------------------
        # Stage 1b — Extract active (task) frames (monkey only)
        # ---------------------------------------------------------------
        task_raw_paths: list[str] = []
        if run_task and species != "mouse":
            task_raw_dir = subj_deriv / f"task_only{dir_suffix}"
            process_all_task_files(
                str(subj_source),
                str(task_raw_dir),
                overwrite=base_cfg["overwrite"],
                apply_log10=APPLY_LOG10,
                log10_eps=base_cfg["log10_eps"],
                exclude_ids=exclude_ids,
            )
            task_raw_paths = list_nc(task_raw_dir, exclude_ids)
            print(f"  Active extracted: {len(task_raw_paths)} sessions")

        # ---------------------------------------------------------------
        # Stage 2 — Reorient and resize (baseline)
        # ---------------------------------------------------------------
        baseline_reoriented_dir = subj_deriv / f"baseline_only_reoriented_resized{dir_suffix}"
        if run_baseline:
            reorient_baseline_sessions(
                baseline_raw_paths,
                baseline_reoriented_dir,
                rotate_k=geo_cfg["rotate_k"],
                flip_session_ids=flip_ids,
                target_size=geo_cfg["target_size"],
                save_previews=geo_cfg["save_previews"],
                overwrite=geo_cfg["overwrite"],
            )
        baseline_reoriented_paths = list_nc(baseline_reoriented_dir)
        print(f"  Baseline reoriented: {len(baseline_reoriented_paths)} sessions")

        # ---------------------------------------------------------------
        # Stage 2b — Reorient and resize (active)
        # ---------------------------------------------------------------
        task_reoriented_paths: list[str] = []
        if task_raw_paths:
            task_reoriented_dir = subj_deriv / f"task_only_reoriented_resized{dir_suffix}"
            reorient_baseline_sessions(
                task_raw_paths,
                task_reoriented_dir,
                rotate_k=geo_cfg["rotate_k"],
                flip_session_ids=flip_ids,
                target_size=geo_cfg["target_size"],
                save_previews=False,
                overwrite=geo_cfg["overwrite"],
            )
            task_reoriented_paths = list_nc(task_reoriented_dir)
            print(f"  Active reoriented: {len(task_reoriented_paths)} sessions")

        # ---------------------------------------------------------------
        # Stage 3 — Standardize (baseline)
        # ---------------------------------------------------------------
        baseline_std_dir = subj_deriv / f"baseline_only_standardized{dir_suffix}"
        if run_baseline:
            standardize_stage_sessions(
                baseline_reoriented_paths,
                baseline_std_dir,
                eps=std_cfg["eps"],
                floor_percentile=std_cfg["floor_percentile"],
                clip_abs=std_cfg["clip_abs"],
                smooth_kernel_sizes=std_cfg["smooth_kernel_sizes"],
                causal=std_cfg.get("causal", False),
                overwrite=std_cfg["overwrite"],
            )
        print(f"  Baseline standardized: {len(list_nc(baseline_std_dir))} sessions")

        # ---------------------------------------------------------------
        # Stage 3b — Standardize (active) using baseline stats
        # ---------------------------------------------------------------
        if task_reoriented_paths:
            task_std_dir = subj_deriv / f"task_only_standardized{dir_suffix}"
            standardize_task_sessions_with_baseline_stats(
                task_reoriented_paths,
                baseline_std_dir=baseline_std_dir,
                out_dir=task_std_dir,
                clip_abs=std_cfg["clip_abs"],
                smooth_kernel_sizes=std_cfg["smooth_kernel_sizes"],
                overwrite=std_cfg["overwrite"],
            )
            print(f"  Active standardized: {len(list_nc(task_std_dir))} sessions")

        # ---------------------------------------------------------------
        # Stage 4 — Tissue segmentation (from baseline reoriented frames)
        # ---------------------------------------------------------------
        if run_baseline:
            print(f"\n=== Tissue segmentation for {subject} ===")
            tissue_dir = subj_deriv / f"tissue_masks{dir_suffix}"
            segment_all_sessions(
                baseline_reoriented_paths,
                tissue_dir,
                vessel_intensity_percentile=tissue_cfg["vessel_intensity_percentile"],
                vessel_cv_percentile=tissue_cfg["vessel_cv_percentile"],
                min_vessel_pixels=tissue_cfg["min_vessel_pixels"],
                closing_radius=tissue_cfg.get("closing_radius", 1),
                overwrite=tissue_cfg["overwrite"],
            )
            print(f"  Tissue masks: {len(list_nc(tissue_dir))}")
            plot_vessel_masks(tissue_dir, subject, subj_deriv / f"figures{dir_suffix}")


if __name__ == "__main__":
    main()
