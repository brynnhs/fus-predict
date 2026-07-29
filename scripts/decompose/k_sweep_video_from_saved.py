"""
scripts/decompose/k_sweep_video_from_saved.py
-----------------------------------------------
Rebuild the combined k-sweep reconstruction video (one panel per k) from an
already-completed decompose_sweep.py run, without refitting anything. Reads
the saved <k_dir>/<session_id>.npz files under
derivatives/decompose/sweep/<method>/kNNN/<session>/ for one session, and
calls the same render_k_sweep_reconstruction_video used by decompose_sweep.py.

Usage
-----
    python scripts/decompose/k_sweep_video_from_saved.py --config config/decompose.yaml --video-session Se01092020
    python scripts/decompose/k_sweep_video_from_saved.py --config config/decompose.yaml --video-session Se01092020 --component-sizes 2 10 40 100
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from decompose import DecomposeConfig, DecompositionResult
from decompose_sweep import render_k_sweep_reconstruction_video

from fuspredict.data.loading import load_sessions
from fuspredict.project import (
    find_repo_root,
    get_excluded_sessions,
    load_project_config,
    resolve_standardized_and_mask_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='Path to decompose YAML config (used for method/paths)')
    parser.add_argument('--project-config', default='config.yml')
    parser.add_argument('--sweep-dir', default=None,
                         help='Root of the sweep output (defaults to derivatives/decompose/sweep)')
    parser.add_argument('--video-session', required=True, help='Session ID to render the combined video for')
    parser.add_argument('--component-sizes', type=int, nargs='*', default=None,
                         help='Restrict to these k values instead of every kNNN folder found')
    parser.add_argument('--video-speed', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--top-n', type=int, default=None,
                         help='If set, each panel reconstructs from only its top-N ranked components '
                              'instead of all k — sensitive to PCA vs ICA component ordering')
    args = parser.parse_args()

    cfg = DecomposeConfig.from_yaml(args.config)

    repo_root = find_repo_root()
    project_cfg = load_project_config(repo_root, config_name=args.project_config)
    subject = project_cfg['subjects']['all'][0]
    exclude_ids = get_excluded_sessions(project_cfg, subject, cfg.exclude_sessions)

    standardized_dir, mask_dir = resolve_standardized_and_mask_dirs(
        repo_root, project_cfg, subject, cfg.standardized_dir, cfg.mask_dir
    )

    if args.sweep_dir:
        sweep_root = repo_root / args.sweep_dir / cfg.method
    else:
        sweep_root = repo_root / 'derivatives/decompose/sweep' / cfg.method

    if not sweep_root.exists():
        raise SystemExit(f"Sweep output not found: {sweep_root}")

    k_dirs = sorted(p for p in sweep_root.iterdir() if p.is_dir() and re.fullmatch(r'k\d+', p.name))
    if args.component_sizes is not None:
        wanted_ks = set(args.component_sizes)
        k_dirs = [p for p in k_dirs if int(p.name[1:]) in wanted_ks]
    if not k_dirs:
        raise SystemExit(f"No kNNN subdirectories found under {sweep_root}"
                          + (f" matching {sorted(wanted_ks)}" if args.component_sizes is not None else ""))

    sessions = load_sessions(standardized_dir, mask_dir=mask_dir, exclude_ids=exclude_ids)
    session = next((s for s in sessions if s.id == args.video_session), None)
    if session is None:
        raise SystemExit(f"Session {args.video_session} not found among loaded sessions.")

    results_by_k: dict[int, DecompositionResult] = {}
    for k_dir in k_dirs:
        k = int(k_dir.name[1:])
        npz_path = k_dir / session.id / f"{session.id}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        results_by_k[k] = DecompositionResult(
            session_id=str(data['session_id']),
            method=str(data['method']),
            spatial=data['spatial'],
            recon_basis=data['recon_basis'],
            timecourse=data['timecourse'],
            ranking_metric=data['ranking_metric'],
            ranking_name=str(data['ranking_name']),
            order=data['order'],
            fps=float(data['fps']),
            calibration_frames=int(data['calibration_frames']),
            mask_kind=str(data['mask_kind']),
            valid_mask=data['valid_mask'],
        )
        print(f"  loaded k={k}: n_components={results_by_k[k].spatial.shape[0]}")

    if len(results_by_k) < 2:
        raise SystemExit(f"Need at least 2 k's with saved results for {session.id}, found {len(results_by_k)}.")

    suffix = '' if args.top_n is None else f'_top{args.top_n}'
    combined_out = sweep_root / 'k_sweep_reconstruction' / f'{session.id}{suffix}.mp4'
    saved_path = render_k_sweep_reconstruction_video(session, results_by_k, combined_out,
                                                       speed=args.video_speed, top_n=args.top_n)
    print(f"\nSaved combined k-sweep video: {saved_path}")


if __name__ == '__main__':
    main()
