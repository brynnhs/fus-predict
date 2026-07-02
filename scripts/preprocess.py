"""
scripts/preprocess.py
----------------------
Preprocessing pipeline entrypoint.

Runs, per subject, the full preprocessing stage sequence:
  1. Baseline (and optionally task) frame extraction with optional log10 transform
  2. Geometry — reorientation and resizing
  3. Temporal filtering (lowpass / highpass / amplitude clip)
  4. Standardization (z-scoring)
  5. Tissue segmentation (vessel vs parenchyma masks)

Outputs are written under ``derivatives/preprocessing/<subject>/`` as
described by each stage module; see fuspredict.preprocessing.* for details.

Usage:
  python scripts/preprocess.py
  python scripts/preprocess.py --config config_mouse.yml
"""

import argparse
from pathlib import Path

from fuspredict.preprocessing.filters import filter_reoriented_sessions
from fuspredict.preprocessing.geometry import reorient_baseline_sessions
from fuspredict.preprocessing.io import (
    process_all_baseline_files,
    process_all_baseline_files_mouse,
    process_all_task_files,
)
from fuspredict.preprocessing.standardization import standardize_stage_sessions
from fuspredict.preprocessing.tissue_masks import segment_all_sessions
from fuspredict.project import find_repo_root, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fUS preprocessing pipeline.")
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Config filename inside config/ (default: config.yml).",
    )
    return parser.parse_args()


def list_nc(directory: Path, exclude_ids: set[str] | None = None) -> list[str]:
    """
    List ``.nc`` files in ``directory``, optionally excluding session IDs.

    Glob all ``.nc`` outputs rather than using each stage's return value,
    which only includes files written in the current run (empty when
    overwrite=False).
    """
    paths = sorted(str(p) for p in Path(directory).glob("*.nc"))
    if exclude_ids:
        paths = [p for p in paths if not any(sid in Path(p).stem for sid in exclude_ids)]
    return paths


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    config = load_project_config(repo_root, config_name=args.config)

    deriv_root  = repo_root / config["paths"]["preprocessing"]
    source_root = repo_root / config["paths"]["sourcedata"]
    subjects    = config["subjects"]["all"]
    species     = config["subjects"].get("species", "monkey")

    base_cfg   = config["preprocessing"]["baseline"]
    geo_cfg    = config["preprocessing"]["geometry"]
    filt_cfg   = config["preprocessing"]["filtering"]
    std_cfg    = config["preprocessing"]["standardization"]
    tissue_cfg = config["preprocessing"]["tissue_segmentation"]

    APPLY_LOG10 = base_cfg["apply_log10"]
    dir_suffix  = "" if APPLY_LOG10 else "_nolog"

    for subject in subjects:
        subj_deriv  = deriv_root / subject
        exclude_ids = set(config["subjects"].get("sessions_to_exclude", {}).get(subject, []))

        print(f"\n=== Processing subject {subject} (species={species}) ===")

        # flip_ids applies to monkey geometry; always computed so it's available
        # for both the task-reorientation block and the main geometry stage
        flip_ids    = set() if species == "mouse" else set(
            geo_cfg["flip_session_ids_by_subject"].get(subject, [])
        )
        subj_source = source_root / subject

        period       = config["preprocessing"].get("period", "baseline")
        run_baseline = period in ("baseline", "both")
        run_task     = period in ("task", "both")

        # Stage 1 — Baseline extraction
        baseline_dir   = subj_deriv / f"baseline_only{dir_suffix}"
        baseline_paths = []
        if run_baseline:
            if species == "mouse":
                # Mouse: read .source.scan HDF5 files; stimulus timing from Excel
                excel_path  = repo_root / config["subjects"].get(
                    "excel_metadata",
                    "data/sourcedata/mouse/Summary PeriFus experiments.xlsx",
                )
                process_all_baseline_files_mouse(
                    data_directory=str(subj_source),
                    output_dir=str(baseline_dir),
                    excel_path=str(excel_path),
                    overwrite=base_cfg["overwrite"],
                    apply_log10=APPLY_LOG10,
                    log10_eps=base_cfg["log10_eps"],
                    exclude_ids=exclude_ids,
                )
            else:
                # Monkey: read Datas_*.mat + Label_pauses_*.mat files
                process_all_baseline_files(
                    str(subj_source),
                    str(baseline_dir),
                    overwrite=base_cfg["overwrite"],
                    apply_log10=APPLY_LOG10,
                    log10_eps=base_cfg["log10_eps"],
                    exclude_ids=exclude_ids,
                )
            baseline_paths = list_nc(baseline_dir, exclude_ids)
            print(f"  Baseline sessions: {len(baseline_paths)}")

        # Stage 1b — Task extraction (monkey only; mouse has no separate task frames)
        if run_task and species != "mouse":
            task_dir = subj_deriv / f"task_only{dir_suffix}"
            process_all_task_files(
                str(subj_source),
                str(task_dir),
                overwrite=base_cfg["overwrite"],
                apply_log10=APPLY_LOG10,
                log10_eps=base_cfg["log10_eps"],
                exclude_ids=exclude_ids,
            )
            task_paths = list_nc(task_dir, exclude_ids)
            print(f"  Task sessions: {len(task_paths)}")

            task_reoriented_dir = subj_deriv / f"task_only_reoriented_resized{dir_suffix}"
            reorient_baseline_sessions(
                task_paths,
                task_reoriented_dir,
                rotate_k=geo_cfg["rotate_k"],
                flip_session_ids=flip_ids,
                target_size=geo_cfg["target_size"],
                save_previews=False,
                overwrite=geo_cfg["overwrite"],
            )
            task_reoriented_paths = list_nc(task_reoriented_dir)

            task_std_dir = subj_deriv / f"task_only_standardized{dir_suffix}"
            standardize_stage_sessions(
                task_reoriented_paths,
                task_std_dir,
                eps=std_cfg["eps"],
                floor_percentile=std_cfg["floor_percentile"],
                clip_abs=std_cfg["clip_abs"],
                smooth_kernel_sizes=std_cfg["smooth_kernel_sizes"],
                causal=std_cfg.get("causal", False),
                overwrite=std_cfg["overwrite"],
            )
            print(f"  Task standardized: {len(list_nc(task_std_dir))}")

        # Stages 2–5 — Baseline geometry, filtering, standardization, tissue masks
        reoriented_paths = []
        filtered_paths   = []
        if run_baseline:
            # Stage 2 — Reorient and resize
            reoriented_dir = subj_deriv / f"baseline_only_reoriented_resized{dir_suffix}"
            reorient_baseline_sessions(
                baseline_paths,
                reoriented_dir,
                rotate_k=geo_cfg["rotate_k"],
                flip_session_ids=flip_ids,
                target_size=geo_cfg["target_size"],
                save_previews=geo_cfg["save_previews"],
                overwrite=geo_cfg["overwrite"],
            )
            reoriented_paths = list_nc(reoriented_dir)
            print(f"  Reoriented sessions: {len(reoriented_paths)}")

            # Stage 3 — Temporal filtering
            filtered_dir = subj_deriv / f"baseline_only_filtered{dir_suffix}"
            filter_reoriented_sessions(
                reoriented_paths,
                filtered_dir,
                enable_lowpass=filt_cfg.get("enable_lowpass", False),
                lowpass_cutoff_hz=filt_cfg.get("lowpass_cutoff_hz", 0.5),
                lowpass_order=filt_cfg.get("lowpass_order", 4),
                enable_highpass=filt_cfg["enable_highpass"],
                highpass_cutoff_hz=filt_cfg["highpass_cutoff_hz"],
                highpass_order=filt_cfg["highpass_order"],
                enable_clip=filt_cfg["enable_clip"],
                clip_bottom=filt_cfg["clip_bottom"],
                clip_top=filt_cfg["clip_top"],
                fps_fallback=filt_cfg["fps_fallback"],
                overwrite=filt_cfg["overwrite"],
            )
            filtered_paths = list_nc(filtered_dir)
            print(f"  Filtered sessions: {len(filtered_paths)}")

            # Stage 4 — Standardization
            std_dir = subj_deriv / f"baseline_only_standardized{dir_suffix}"
            standardize_stage_sessions(
                reoriented_paths + filtered_paths,
                std_dir,
                eps=std_cfg["eps"],
                floor_percentile=std_cfg["floor_percentile"],
                clip_abs=std_cfg["clip_abs"],
                smooth_kernel_sizes=std_cfg["smooth_kernel_sizes"],
                causal=std_cfg.get("causal", False),
                overwrite=std_cfg["overwrite"],
            )
            print(f"  Standardized sessions: {len(list_nc(std_dir))}")

        # Stage 5 — Tissue segmentation (vessel vs parenchyma masks)
        print(f"\n=== Tissue segmentation for {subject} ===")
        tissue_dir = subj_deriv / f"tissue_masks{dir_suffix}"
        segment_all_sessions(
            reoriented_paths,
            tissue_dir,
            vessel_intensity_percentile=tissue_cfg["vessel_intensity_percentile"],
            vessel_cv_percentile=tissue_cfg["vessel_cv_percentile"],
            min_vessel_pixels=tissue_cfg["min_vessel_pixels"],
            overwrite=tissue_cfg["overwrite"],
        )
        print(f"  Tissue masks: {len(list_nc(tissue_dir))}")


if __name__ == "__main__":
    main()
