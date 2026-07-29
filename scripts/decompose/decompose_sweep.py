"""
scripts/decompose/decompose_sweep.py
-------------------------------------
Same as decompose.py, but instead of choosing n_components automatically
(auto_cv / auto_pa), sweeps a fixed list of candidate component counts and
saves one decomposition per (session, k) to derivatives/decompose/sweep.

Pipeline
--------
1. Load session, apply vessel mask if configured, flatten to (T, n_pixels).
2. For each k in --component-sizes: fit PCA or ICA with n_components=k on
   ``calibration_frames`` only (frozen-basis, same causal constraint as
   decompose.py), apply to the full session, rank components, save one .npz
   per (session, k).
3. Optionally render the same figure set as decompose.py, per (session, k).
4. Optionally render component-activation videos for one example session,
   across all swept k values.

Usage
-----
    python scripts/decompose/decompose_sweep.py --config config/decompose.yaml
    python scripts/decompose/decompose_sweep.py --config config/decompose.yaml --component-sizes 2 5 10 20 40
    python scripts/decompose/decompose_sweep.py --config config/decompose.yaml --sessions Se01092020
    python scripts/decompose/decompose_sweep.py --config config/decompose.yaml --no-figures
    python scripts/decompose/decompose_sweep.py --config config/decompose.yaml --video-session Se01092020
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from decompose import (
    DecomposeConfig,
    DecompositionResult,
    decompose_session,
    save_result,
    fig_spatial_component_grid,
    fig_timecourses,
    fig_ranking,
    fig_psd_per_component,
    fig_morans_vs_peakiness,
    fig_pooled_ranking,
    fig_pooled_stability,
    fig_spatial_temporal_tradeoff,
    render_component_activation_video,
    _write_frames_to_video,
    _compute_acf,
    _METHOD_COLOR,
    _DOUBLE_COL,
)

from fuspredict.data.loading import load_sessions
from fuspredict.data.session import Session
from fuspredict.plot_utils import savefig
from fuspredict.project import find_repo_root, get_excluded_sessions, load_project_config

matplotlib.use('Agg')

_DEFAULT_COMPONENT_SIZES = [2, 6, 10, 20, 40, 50, 80, 100]


# ---------------------------------------------------------------------------
# Reconstruction RMSE vs k
# ---------------------------------------------------------------------------

def _reconstruct(session: Session, result: DecompositionResult, component_indices) -> np.ndarray:
    """Additive reconstruction using only the given component indices."""
    recon = np.zeros((session.n_frames, session.height, session.width), dtype=np.float64)
    for comp_i in component_indices:
        smap = np.nan_to_num(result.recon_basis[comp_i])
        recon += np.outer(result.timecourse[comp_i], smap.ravel()).reshape(recon.shape)
    return recon


def _full_frame_reconstruction_rmse(session: Session, result: DecompositionResult) -> float:
    """Full-session, full-frame RMSE between original and the additive
    reconstruction using all of result's components (i.e. all k components
    at this sweep size) — same recon_basis @ timecourse construction used
    for the reconstruction video.
    """
    recon = _reconstruct(session, result, range(result.spatial.shape[0]))
    original = session.frames.astype(np.float64)
    diff = (original - recon)[:, result.valid_mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def _top_n_reconstruction_rmse(session: Session, result: DecompositionResult, top_n: int) -> float:
    """Same as _full_frame_reconstruction_rmse, but reconstructing from only
    the top-N ranked components (result.order — variance-ranked for PCA,
    |kurtosis|-ranked for ICA). Unlike the full-k RMSE, this is sensitive to
    *which* components each method picks as most important, since PCA and
    ICA order the same subspace differently.
    """
    top_idx = result.order[:top_n]
    recon = _reconstruct(session, result, top_idx)
    original = session.frames.astype(np.float64)
    diff = (original - recon)[:, result.valid_mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def fig_rmse_vs_k(rmse_by_k: dict[int, list[float]], method: str, out: Path,
                   title: str | None = None, fname: str = 'fig_rmse_vs_k') -> None:
    """Jittered per-session reconstruction RMSE scatter at each k, with a
    median bar — same visual convention as plot_rmse_strip in
    fuspredict.evaluation.visualization.
    """
    ks = sorted(rmse_by_k)
    rng = np.random.default_rng(0)
    color = _METHOD_COLOR[method]

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.5), constrained_layout=True)

    for xi, k in enumerate(ks):
        vals = np.array(rmse_by_k[k], dtype=float)
        if not len(vals):
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=18, alpha=0.75,
                   linewidths=0.3, edgecolors='white', zorder=3)
        med = float(np.median(vals))
        ax.plot([xi - 0.25, xi + 0.25], [med, med], color='k', lw=2.0,
                zorder=4, solid_capstyle='round')

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel('Number of components (k)')
    ax.set_ylabel('Full-frame reconstruction RMSE')
    ax.set_title(title or f'{method.upper()} reconstruction RMSE vs k (dots=sessions, bar=median)', fontsize=9)
    ax.grid(axis='y', alpha=0.25, zorder=0)

    savefig(fig, out / fname)
    plt.close(fig)


def fig_acf1_vs_k(acf1_by_k: dict[int, list[float]], method: str, out: Path,
                   fname: str = 'fig_acf1_vs_k') -> None:
    """Jittered per-component lag-1 ACF scatter at each k, with a median bar
    — same visual convention as fig_rmse_vs_k. One point per (session,
    top-N component) at each swept k.
    """
    ks = sorted(acf1_by_k)
    rng = np.random.default_rng(0)
    color = _METHOD_COLOR[method]

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.5), constrained_layout=True)

    for xi, k in enumerate(ks):
        vals = np.array(acf1_by_k[k], dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=18, alpha=0.75,
                   linewidths=0.3, edgecolors='white', zorder=3)
        med = float(np.median(vals))
        ax.plot([xi - 0.25, xi + 0.25], [med, med], color='k', lw=2.0,
                zorder=4, solid_capstyle='round')

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel('Number of components (k)')
    ax.set_ylabel('Lag-1 ACF')
    ax.set_title(f'{method.upper()} lag-1 ACF vs k (dots=top-N components/session, bar=median)', fontsize=9)
    ax.grid(axis='y', alpha=0.25, zorder=0)

    savefig(fig, out / fname)
    plt.close(fig)


def render_k_sweep_reconstruction_video(session: Session, results_by_k: dict[int, DecompositionResult],
                                         out_path: Path, speed: float = 1.0, top_n: int | None = None) -> Path:
    """Grid video, one panel per swept k, each panel = reconstruction from
    that k's components — lets you watch how reconstruction quality changes
    across the sweep, frame by frame, for one example session.

    If top_n is None, each panel uses all k components (full reconstruction).
    If top_n is set, each panel uses only its top-N ranked components
    (result.order) — this is the version that can actually visually
    distinguish PCA (variance-ranked) from ICA (|kurtosis|-ranked), since a
    full-k reconstruction spans the same subspace regardless of ranking.
    """
    ks = sorted(results_by_k)
    recons = {}
    for k in ks:
        result = results_by_k[k]
        comp_indices = range(result.spatial.shape[0]) if top_n is None else result.order[:top_n]
        recons[k] = _reconstruct(session, result, comp_indices)

    original = session.frames.astype(np.float64)
    vmax = float(np.nanpercentile(np.abs(original), 98))

    ncols = min(len(ks), 4)
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows), constrained_layout=True, dpi=120)
    axes = np.array(axes).reshape(nrows, ncols)

    ims = {}
    for i, k in enumerate(ks):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ims[k] = ax.imshow(recons[k][0], cmap='gray', vmin=-vmax, vmax=vmax)
        ax.set_title(f'k={k}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for i in range(len(ks), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    label = 'full' if top_n is None else f'top-{top_n}'

    def update(frame_i):
        for k in ks:
            ims[k].set_data(recons[k][frame_i])
        fig.suptitle(f'{session.id} — component sweep reconstruction ({label}) — frame {frame_i}/{session.n_frames}',
                     fontsize=9)

    saved_path = _write_frames_to_video(fig, update, session.n_frames, out_path, fps=session.fps * speed)
    plt.close(fig)
    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='Path to decompose YAML config')
    parser.add_argument('--project-config', default='config.yml',
                         help='Project config filename in config/ (selects subject, e.g. config_gus.yml)')
    parser.add_argument('--sessions', nargs='*', default=None, help='Override session ID subset')
    parser.add_argument('--component-sizes', type=int, nargs='*', default=_DEFAULT_COMPONENT_SIZES,
                         help='Fixed n_components values to sweep (replaces auto_cv/auto_pa selection)')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('--top-n', type=int, default=8, help='Top-N components shown in figures')
    parser.add_argument('--rmse-top-n', type=int, nargs='*', default=None,
                         help='If set, also plot fig_rmse_vs_k_top{N} using only the top-N ranked '
                              'components per (session, k) instead of all k — unlike the full-k RMSE, '
                              'this is sensitive to PCA vs ICA component ordering')
    parser.add_argument('--acf-lags', type=int, nargs='*', default=[1, 5, 10, 20, 30, 40, 50],
                         help='Lags (frames) shown in fig_spatial_temporal_tradeoff')
    parser.add_argument('--video-session', default=None,
                         help='Session ID to render per-component activation videos for, one per swept k')
    parser.add_argument('--video-top-k', type=int, default=None,
                         help='If set, only render videos for the top-k components of the largest sweep '
                              'size instead of all components at every k')
    parser.add_argument('--video-speed', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--no-combined-video', action='store_true',
                         help='Skip the combined grid video (one panel per k) for --video-session')
    args = parser.parse_args()

    cfg = DecomposeConfig.from_yaml(args.config)
    if args.sessions:
        cfg.sessions = args.sessions
    # sweep overrides any auto_cv/auto_pa selection in the config
    cfg.output_dir = 'derivatives/decompose/sweep'

    repo_root = find_repo_root()
    project_cfg = load_project_config(repo_root, config_name=args.project_config)
    subject = project_cfg['subjects']['all'][0]
    exclude_ids = get_excluded_sessions(project_cfg, subject, cfg.exclude_sessions)

    preproc_root = repo_root / project_cfg['paths']['preprocessing'] / subject
    standardized_dir = Path(cfg.standardized_dir) if cfg.standardized_dir else \
        preproc_root / 'baseline_only_standardized'
    mask_dir = Path(cfg.mask_dir) if cfg.mask_dir else \
        preproc_root / 'tissue_masks'
    out_root = repo_root / cfg.output_dir / cfg.method

    sessions = load_sessions(standardized_dir, mask_dir=mask_dir, exclude_ids=exclude_ids)
    if cfg.sessions:
        wanted = set(cfg.sessions)
        sessions = [s for s in sessions if s.id in wanted]
    if not sessions:
        raise SystemExit("No sessions to process — check config paths / session filters.")

    print(f"Sweeping {len(sessions)} sessions with method={cfg.method}, "
          f"component_sizes={args.component_sizes}, mask={cfg.mask}, "
          f"calibration_frames={cfg.calibration_frames}")

    results_by_k: dict[int, list[DecompositionResult]] = {k: [] for k in args.component_sizes}
    rmse_by_k: dict[int, list[float]] = {k: [] for k in args.component_sizes}
    top_n_rmse_by_n: dict[int, dict[int, list[float]]] = {n: {k: [] for k in args.component_sizes}
                                                            for n in (args.rmse_top_n or [])}
    acf1_by_k: dict[int, list[float]] = {k: [] for k in args.component_sizes}

    for session in sessions:
        for k in args.component_sizes:
            k_cfg = DecomposeConfig(**{**cfg.__dict__, 'n_components': k})
            try:
                result = decompose_session(session, k_cfg)
            except Exception as exc:
                warnings.warn(f"Skipping {session.id} k={k}: {exc}", stacklevel=2)
                continue

            k_dir = out_root / f'k{k:03d}' / session.id
            out_path = save_result(result, k_dir)
            rmse = _full_frame_reconstruction_rmse(session, result)
            rmse_by_k[k].append(rmse)
            for n in (args.rmse_top_n or []):
                n_eff = min(n, result.spatial.shape[0])
                top_n_rmse_by_n[n][k].append(_top_n_reconstruction_rmse(session, result, n_eff))
            top_n_acf = min(args.top_n, result.spatial.shape[0])
            for comp_i in result.order[:top_n_acf]:
                acf1_by_k[k].append(_compute_acf(result.timecourse[comp_i], 1))
            print(f"  {session.id} k={k}: saved {out_path.name}  "
                  f"(n_components={result.spatial.shape[0]}, T={result.timecourse.shape[1]}, "
                  f"recon_rmse={rmse:.4f})")
            results_by_k[k].append(result)

            if not args.no_figures:
                top_n = min(args.top_n, result.spatial.shape[0])
                fig_spatial_component_grid(result, top_n, k_dir)
                fig_timecourses(result, top_n, k_dir)
                fig_ranking(result, k_dir)
                fig_psd_per_component(result, top_n, cfg.freq_gate_hz, 0.448, k_dir)
                fig_morans_vs_peakiness(result, k_dir)

    if not args.no_figures:
        for k, results in results_by_k.items():
            if len(results) < 2:
                continue
            min_n_comp = min(r.spatial.shape[0] for r in results)
            k_out = out_root / f'k{k:03d}'
            fig_pooled_ranking(results, out=k_out)
            fig_pooled_stability(results, top_n=min(args.top_n, min_n_comp), out=k_out)
            fig_spatial_temporal_tradeoff(results, top_n=min(args.top_n, min_n_comp),
                                           lags=args.acf_lags, out=k_out)

        if any(len(v) for v in rmse_by_k.values()):
            fig_rmse_vs_k(rmse_by_k, cfg.method, out=out_root)

        if any(len(v) for v in acf1_by_k.values()):
            fig_acf1_vs_k(acf1_by_k, cfg.method, out=out_root)

        for n, rmse_by_k_n in top_n_rmse_by_n.items():
            if any(len(v) for v in rmse_by_k_n.values()):
                fig_rmse_vs_k(rmse_by_k_n, cfg.method, out=out_root,
                              title=f'{cfg.method.upper()} reconstruction RMSE vs k, top-{n} components '
                                    f'(dots=sessions, bar=median)',
                              fname=f'fig_rmse_vs_k_top{n}')

    if args.video_session:
        session = next((s for s in sessions if s.id == args.video_session), None)
        if session is None:
            raise SystemExit(f"Session {args.video_session} not found among processed sessions.")

        session_results_by_k: dict[int, DecompositionResult] = {}
        for k in args.component_sizes:
            result = next((r for r in results_by_k[k] if r.session_id == args.video_session), None)
            if result is None:
                continue
            session_results_by_k[k] = result

            video_dir = out_root / f'k{k:03d}' / session.id
            n_comp = result.spatial.shape[0]
            comp_indices = range(n_comp) if args.video_top_k is None else result.order[:args.video_top_k]
            for comp_i in comp_indices:
                out_path = video_dir / f'component_{comp_i}.mp4'
                saved_path = render_component_activation_video(session, result, int(comp_i), out_path,
                                                                 speed=args.video_speed)
                print(f"Saved video: {saved_path}")

        if not args.no_combined_video and len(session_results_by_k) > 1:
            combined_out = out_root / 'k_sweep_reconstruction' / f'{session.id}.mp4'
            saved_path = render_k_sweep_reconstruction_video(session, session_results_by_k, combined_out,
                                                               speed=args.video_speed)
            print(f"Saved combined k-sweep video: {saved_path}")

            if args.video_top_k is not None:
                combined_top_n_out = out_root / 'k_sweep_reconstruction' / f'{session.id}_top{args.video_top_k}.mp4'
                saved_path = render_k_sweep_reconstruction_video(session, session_results_by_k, combined_top_n_out,
                                                                   speed=args.video_speed, top_n=args.video_top_k)
                print(f"Saved combined top-{args.video_top_k} k-sweep video: {saved_path}")

    n_done = sum(len(v) for v in results_by_k.values())
    n_total = len(sessions) * len(args.component_sizes)
    print(f"\nDone. {n_done}/{n_total} (session, k) decompositions. Output: {out_root}")


if __name__ == '__main__':
    main()
