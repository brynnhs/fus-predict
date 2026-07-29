"""
scripts/decompose.py
---------------------
Fit a frozen spatial basis (PCA or ICA) per session on a causal calibration
window, then apply it to the full recording to get component time-courses.

Pipeline
--------
1. Load session, apply vessel mask if configured, flatten to (T, n_pixels).
2. Fit PCA or ICA on ``calibration_frames`` only — never on the full session.
   This mirrors the frozen-basis constraint used elsewhere in the pipeline
   (same causal-leakage avoidance as ICA-AROMA / aCompCor): the basis must
   not see frames it will later be used to explain.
3. Apply the frozen basis to the *entire* session (not just the calibration
   block) via matrix multiply, producing time-courses for the full recording.
4. Rank components: explained-variance ratio for PCA, |excess kurtosis| for
   ICA (ICA components have no natural variance ordering).
5. Save spatial maps, time-courses, ranking metric, and method metadata to
   one .npz per session.
6. Optionally render the figure set (spatial grid, time-courses, ranking,
   per-component PSD, Moran's I vs spectral peakiness, PCA-vs-ICA compare,
   pooled/per-session stability) and a reconstruction or single-component
   activation video.

Usage
-----
    python scripts/decompose/decompose.py --config config/decompose.yaml
    python scripts/decompose/decompose.py --config config/decompose.yaml --sessions Se01092020
    python scripts/decompose/decompose.py --config config/decompose.yaml --no-figures
    python scripts/decompose/decompose.py --config config/decompose.yaml --video reconstruction --video-session Se01092020
    python scripts/decompose/decompose.py --config config/decompose.yaml --video component --video-session Se01092020 --video-component 2
    python scripts/decompose/decompose.py --config config/decompose.yaml --video all-components --video-session Se01092020
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from fuspredict.autocorrelation import safe_temporal_corr_map
from fuspredict.data.loading import load_sessions
from fuspredict.data.session import Session
from fuspredict.plot_utils import savefig
from fuspredict.project import find_repo_root, get_excluded_sessions, load_project_config

matplotlib.use('Agg')

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Palatino Linotype', 'Palatino', 'Georgia', 'DejaVu Serif'],
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'axes.grid':          False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.major.size':   0,
    'ytick.major.size':   3,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'legend.frameon':     False,
})

_DOUBLE_COL = 7.0
_METHOD_COLOR = {'pca': '#3B82C4', 'ica': '#E8872A'}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DecomposeConfig:
    method: str                    # "pca" | "ica"
    n_components: int | str        # int, "auto_cv" (cross-validated reconstruction error), or
                                    # "auto_pa" (parallel analysis) — "auto" is an alias for "auto_pa"
    fit_mode: str                  # "per_session" | "pooled"  (open question — see NOTE below)
    calibration_frames: int        # causal frozen-basis fit window
    mask: str                      # "full_frame" | "vessel_only"
    freq_gate_hz: float            # QPP respiration-band exclusion (matches 0.448 Hz artifact gate)
    ica_algorithm: dict = field(default_factory=dict)
    standardized_dir: str | None = None
    mask_dir: str | None = None
    output_dir: str = "derivatives/decompose"
    sessions: list[str] | None = None
    exclude_sessions: list[str] | None = None
    seed: int = 0
    n_components_candidates: list[int] = field(
        default_factory=lambda: [2, 5, 10, 15, 20, 30, 40, 50, 75, 100])  # used when n_components == "auto_cv"
    pa_percentile: float = 95.0    # null percentile threshold for n_components == "auto_pa"
    pa_n_permutations: int = 100   # number of permutations for n_components == "auto_pa"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DecomposeConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def __post_init__(self) -> None:
        if self.method not in ("pca", "ica"):
            raise ValueError(f"method must be 'pca' or 'ica', got {self.method!r}")
        if isinstance(self.n_components, str):
            if self.n_components == "auto":
                self.n_components = "auto_pa"
            if self.n_components not in ("auto_cv", "auto_pa"):
                raise ValueError(
                    f"n_components must be an int, 'auto_cv', or 'auto_pa', got {self.n_components!r}")
        if self.fit_mode not in ("per_session", "pooled"):
            raise ValueError(f"fit_mode must be 'per_session' or 'pooled', got {self.fit_mode!r}")
        if self.mask not in ("full_frame", "vessel_only"):
            raise ValueError(f"mask must be 'full_frame' or 'vessel_only', got {self.mask!r}")
        if self.fit_mode == "pooled":
            # NOTE: pooled fit_mode is an open question (cross-session basis, unclear
            # calibration-window semantics across sessions of different length/timing).
            # Flagged rather than silently handled — falls back to per_session behavior.
            warnings.warn(
                "fit_mode='pooled' is not yet implemented (open design question — "
                "how to define a shared calibration window across sessions of "
                "different length/timing). Falling back to per_session fitting.",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Fit result container
# ---------------------------------------------------------------------------

@dataclass
class DecompositionResult:
    session_id: str
    method: str
    spatial: np.ndarray          # (n_components, H, W), NaN outside mask — projection/unmixing weights
    recon_basis: np.ndarray      # (n_components, H, W), NaN outside mask — additive reconstruction basis
    timecourse: np.ndarray       # (n_components, T) — full-session time-courses
    ranking_metric: np.ndarray   # (n_components,) — explained-var-ratio (pca) or |kurtosis| (ica)
    ranking_name: str            # "explained_variance_ratio" | "abs_kurtosis"
    order: np.ndarray            # (n_components,) indices sorted best-first by ranking_metric
    fps: float
    calibration_frames: int
    mask_kind: str
    valid_mask: np.ndarray       # (H, W) bool — pixels included in the decomposition
    n_components_selection: tuple | None = None
    # if n_components="auto_cv": (candidates, mean_rmse)
    # if n_components="auto_pa": (real_eigenvalues, null_threshold, null_eigenvalues)
    n_components_method: str | None = None  # "auto_cv" | "auto_pa" | None (fixed int)


# ---------------------------------------------------------------------------
# Shared fit(X) -> (spatial, timecourse, ranking_metric) interface
# ---------------------------------------------------------------------------

def _flatten(frames: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """(T, H, W) -> (T, n_valid_pixels) float64, mean-centered per pixel."""
    px = frames[:, valid_mask].astype(np.float64)
    px -= px.mean(axis=0, keepdims=True)
    return px


def select_n_components(X_calib: np.ndarray, candidates: list[int], seed: int,
                         n_splits: int = 5, holdout_frac: float = 0.2) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Choose k via repeated frame-holdout cross-validated PCA reconstruction error.

    For each split, a random subset of calibration frames is held out; PCA is
    fit on the remaining frames and used to project + reconstruct the held-out
    frames. Held-out RMSE is averaged across splits for each candidate k, and
    the k minimizing mean held-out RMSE is returned. Used to pick n_components
    for both PCA and ICA (FastICA whitens to n_components via an internal PCA
    step, so the same reconstruction-error criterion applies).

    Returns (best_k, candidates_array, mean_rmse_per_candidate).
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    n_frames = X_calib.shape[0]
    n_holdout = max(1, int(round(n_frames * holdout_frac)))

    candidates = [k for k in candidates if k < n_frames - n_holdout]
    if not candidates:
        raise ValueError("select_n_components: no valid candidate k below (n_calib_frames - n_holdout)")

    rmse = np.zeros((n_splits, len(candidates)), dtype=np.float64)
    for split_i in range(n_splits):
        perm = rng.permutation(n_frames)
        holdout_idx = perm[:n_holdout]
        train_idx = perm[n_holdout:]
        X_train, X_holdout = X_calib[train_idx], X_calib[holdout_idx]

        max_k = max(candidates)
        pca = PCA(n_components=min(max_k, X_train.shape[0] - 1, X_train.shape[1]), random_state=seed)
        pca.fit(X_train)

        for cand_i, k in enumerate(candidates):
            k_eff = min(k, pca.components_.shape[0])
            basis = pca.components_[:k_eff]           # (k_eff, n_pixels)
            scores = (X_holdout - pca.mean_) @ basis.T
            recon = scores @ basis + pca.mean_
            rmse[split_i, cand_i] = float(np.sqrt(np.mean((X_holdout - recon) ** 2)))

    mean_rmse = rmse.mean(axis=0)
    best_k = int(candidates[int(np.argmin(mean_rmse))])
    return best_k, np.array(candidates), mean_rmse


def fig_component_selection(candidates: np.ndarray, mean_rmse: np.ndarray, best_k: int,
                             session_id: str, out: Path) -> None:
    """Held-out reconstruction RMSE vs number of PCA components, with chosen k marked."""
    fig, ax = plt.subplots(figsize=(_DOUBLE_COL * 0.6, _DOUBLE_COL * 0.4), constrained_layout=True)
    ax.plot(candidates, mean_rmse, color=_METHOD_COLOR['pca'], marker='o', ms=3, lw=1.0)
    ax.axvline(best_k, color='k', lw=0.8, ls='--', label=f'selected k={best_k}')
    ax.set_xlabel('Number of components (k)')
    ax.set_ylabel('Held-out reconstruction RMSE')
    ax.set_title(f'Cross-validated component selection — {session_id}', fontsize=9)
    ax.legend(fontsize=7)
    savefig(fig, out / 'fig_component_selection')
    plt.close(fig)


def select_n_components_parallel_analysis(X_calib: np.ndarray, seed: int, n_permutations: int = 100,
                                           percentile: float = 95.0, max_k: int | None = None
                                           ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Choose k via parallel analysis (Horn, 1965): keep components whose real
    eigenvalue exceeds the given percentile of the null eigenvalue distribution
    at that rank, where the null is built by circularly shifting each pixel's
    time series by an independent random offset. This preserves each pixel's
    own autocorrelation/spectral content and keeps cross-pixel spatial
    correlation largely intact, while randomizing the phase alignment between
    pixels — so genuinely shared timing (evoked structure, drift common to
    many pixels) is destroyed but spurious "signal" from spatial correlation
    alone is not. A plain frame-order permutation (same reorder applied to
    every pixel) does NOT work here: covariance is invariant to row order, so
    it leaves eigenvalues unchanged and cannot serve as a null. Independently
    shuffling each pixel (breaking temporal structure per-pixel) also
    destroys spatial correlation, which under-nulls it and can inflate the
    apparent "signal" tail (this was tried first — see git history).

    k is the number of leading eigenvalues before the first one that fails
    to exceed its null threshold (first crossing from below), which is more
    conservative than counting scattered later crossings due to noise.

    Returns (best_k, real_eigenvalues, null_threshold_per_rank, null_eigenvalues_matrix).
    null_eigenvalues_matrix has shape (n_permutations, max_k) for diagnostic plotting.
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    n_frames, n_pixels = X_calib.shape
    k_cap = min(max_k or n_frames - 1, n_frames - 1, n_pixels)

    real_pca = PCA(n_components=k_cap, random_state=seed)
    real_pca.fit(X_calib)
    real_eigenvalues = real_pca.explained_variance_

    null_eigenvalues = np.zeros((n_permutations, k_cap), dtype=np.float64)
    for perm_i in range(n_permutations):
        shifts = rng.integers(1, n_frames, size=n_pixels)  # exclude shift=0 (identity)
        col_idx = np.arange(n_pixels)
        row_idx = (np.arange(n_frames)[:, None] + shifts[None, :]) % n_frames
        X_null = X_calib[row_idx, col_idx]
        null_pca = PCA(n_components=k_cap, random_state=seed)
        null_pca.fit(X_null)
        null_eigenvalues[perm_i] = null_pca.explained_variance_

    null_threshold = np.percentile(null_eigenvalues, percentile, axis=0)

    above = real_eigenvalues > null_threshold
    best_k = int(np.argmin(above)) if not above.all() else k_cap
    best_k = max(best_k, 1)

    return best_k, real_eigenvalues, null_threshold, null_eigenvalues


def fig_parallel_analysis(real_eigenvalues: np.ndarray, null_threshold: np.ndarray,
                           null_eigenvalues: np.ndarray, best_k: int, session_id: str, out: Path) -> None:
    """Real eigenvalue spectrum vs permutation-null threshold, with chosen k marked."""
    fig, ax = plt.subplots(figsize=(_DOUBLE_COL * 0.6, _DOUBLE_COL * 0.4), constrained_layout=True)
    x = np.arange(1, len(real_eigenvalues) + 1)
    null_lo = np.percentile(null_eigenvalues, 5, axis=0)
    null_hi = np.percentile(null_eigenvalues, 95, axis=0)
    ax.fill_between(x, null_lo, null_hi, color='0.7', alpha=0.4, label='null 5th-95th pct')
    ax.plot(x, null_threshold, color='k', lw=0.8, ls='--', label='null threshold')
    ax.plot(x, real_eigenvalues, color=_METHOD_COLOR['pca'], marker='o', ms=2.5, lw=1.0, label='real eigenvalues')
    ax.axvline(best_k, color='#D62728', lw=0.8, ls=':', label=f'selected k={best_k}')
    ax.set_xlabel('Component rank')
    ax.set_ylabel('Eigenvalue (explained variance)')
    ax.set_yscale('log')
    ax.set_title(f'Parallel analysis — {session_id}', fontsize=9)
    ax.legend(fontsize=6)
    savefig(fig, out / 'fig_parallel_analysis')
    plt.close(fig)


def _fit_pca(X_calib: np.ndarray, n_components: int, seed: int) -> tuple:
    """PCA fit(X) -> (spatial_weights, recon_basis, project_fn, ranking_metric, ranking_name)."""
    from sklearn.decomposition import PCA

    n_comp = min(n_components, X_calib.shape[0] - 1, X_calib.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(X_calib)

    spatial_weights = pca.components_               # (n_comp, n_pixels)
    mean_ = pca.mean_

    def project(X: np.ndarray) -> np.ndarray:
        return (X - mean_) @ spatial_weights.T       # (T, n_comp)

    # PCA's components_ is an orthonormal basis, so it is its own inverse —
    # valid for both projection and additive reconstruction.
    recon_basis = spatial_weights
    return spatial_weights, recon_basis, project, pca.explained_variance_ratio_, "explained_variance_ratio"


def _fit_ica(X_calib: np.ndarray, n_components: int, seed: int, algorithm_kwargs: dict) -> tuple:
    """ICA fit(X) -> (spatial_weights, recon_basis, project_fn, ranking_metric, ranking_name)."""
    from sklearn.decomposition import FastICA

    n_comp = min(n_components, X_calib.shape[0] - 1, X_calib.shape[1])
    kwargs = dict(whiten='unit-variance', max_iter=500, random_state=seed)
    kwargs.update(algorithm_kwargs or {})
    ica = FastICA(n_components=n_comp, **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sources_calib = ica.fit_transform(X_calib)   # (T_calib, n_comp)

    spatial_weights = ica.components_                # (n_comp, n_pixels) — unmixing rows (whitened space)
    mean_ = ica.mean_

    def project(X: np.ndarray) -> np.ndarray:
        return (X - mean_) @ spatial_weights.T        # (T, n_comp)

    # ica.components_ operates in the whitened space and is not the inverse of
    # itself the way PCA's orthonormal basis is. ica.mixing_ (n_pixels, n_comp)
    # is the actual mixing matrix that maps sources back to pixel-space
    # amplitudes and must be used for additive reconstruction instead.
    recon_basis = ica.mixing_.T                       # (n_comp, n_pixels)

    kurtosis = np.array([
        float(np.mean((sources_calib[:, i] - sources_calib[:, i].mean()) ** 4) /
              max(np.var(sources_calib[:, i]) ** 2, 1e-12)) - 3
        for i in range(n_comp)
    ])
    return spatial_weights, recon_basis, project, np.abs(kurtosis), "abs_kurtosis"


_FITTERS: dict[str, Callable] = {
    "pca": lambda X, n, seed, ica_kwargs: _fit_pca(X, n, seed),
    "ica": lambda X, n, seed, ica_kwargs: _fit_ica(X, n, seed, ica_kwargs),
}


def decompose_session(session: Session, cfg: DecomposeConfig) -> DecompositionResult:
    """
    Fit a frozen basis on session's calibration window and apply it to the
    full recording. Respects the causal constraint: the basis is fit only
    on frames [0, calibration_frames), then used to project all T frames.
    """
    if cfg.mask == "vessel_only":
        if session.vessel_mask is None:
            raise ValueError(f"{session.id}: mask='vessel_only' but no vessel_mask loaded")
        valid_mask = session.vessel_mask
    else:
        valid_mask = np.isfinite(session.frames).all(axis=0)

    n_calib = min(cfg.calibration_frames, session.n_frames)
    if n_calib < 2:
        raise ValueError(f"{session.id}: calibration window too short ({n_calib} frames)")

    X_calib = _flatten(session.frames[:n_calib], valid_mask)

    selection = None
    selection_method = None
    if cfg.n_components == "auto_cv":
        n_components, candidates, mean_rmse = select_n_components(
            X_calib, cfg.n_components_candidates, cfg.seed)
        selection = (candidates, mean_rmse)
        selection_method = "auto_cv"
    elif cfg.n_components == "auto_pa":
        n_components, real_eig, null_thresh, null_eig = select_n_components_parallel_analysis(
            X_calib, cfg.seed, n_permutations=cfg.pa_n_permutations, percentile=cfg.pa_percentile)
        selection = (real_eig, null_thresh, null_eig)
        selection_method = "auto_pa"
    else:
        n_components = cfg.n_components

    fitter = _FITTERS[cfg.method]
    spatial_weights, recon_weights, project, ranking_metric, ranking_name = fitter(
        X_calib, n_components, cfg.seed, cfg.ica_algorithm
    )

    # apply frozen basis to the FULL session (causal fit, full-recording apply)
    X_full = _flatten(session.frames, valid_mask)
    timecourse = project(X_full).T                    # (n_comp, T)

    order = np.argsort(ranking_metric)[::-1]

    n_comp = spatial_weights.shape[0]
    H, W = session.height, session.width
    spatial = np.full((n_comp, H, W), np.nan, dtype=np.float64)
    spatial[:, valid_mask] = spatial_weights
    recon_basis = np.full((n_comp, H, W), np.nan, dtype=np.float64)
    recon_basis[:, valid_mask] = recon_weights

    return DecompositionResult(
        session_id=session.id,
        method=cfg.method,
        spatial=spatial,
        recon_basis=recon_basis,
        timecourse=timecourse,
        ranking_metric=ranking_metric,
        ranking_name=ranking_name,
        order=order,
        fps=session.fps,
        calibration_frames=n_calib,
        mask_kind=cfg.mask,
        valid_mask=valid_mask,
        n_components_selection=selection,
        n_components_method=selection_method,
    )


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_result(result: DecompositionResult, session_dir: Path) -> Path:
    """Save one session's decomposition to <session_dir>/<session_id>.npz."""
    session_dir.mkdir(parents=True, exist_ok=True)
    out_path = session_dir / f"{result.session_id}.npz"
    np.savez_compressed(
        out_path,
        spatial=result.spatial,
        recon_basis=result.recon_basis,
        timecourse=result.timecourse,
        ranking_metric=result.ranking_metric,
        order=result.order,
        valid_mask=result.valid_mask,
        session_id=result.session_id,
        method=result.method,
        ranking_name=result.ranking_name,
        fps=result.fps,
        calibration_frames=result.calibration_frames,
        mask_kind=result.mask_kind,
    )
    return out_path


# ---------------------------------------------------------------------------
# Metrics reused for figures (Moran's I, spectral peakiness) — see hfc_pilot.py
# ---------------------------------------------------------------------------

def _morans_i(spatial_map: np.ndarray) -> float:
    """Queen-contiguity Moran's I; NaN pixels excluded. See hfc_pilot.py for derivation notes."""
    H, W = spatial_map.shape
    finite = np.isfinite(spatial_map)
    ys, xs = np.where(finite)
    idx_map = np.full((H, W), -1, dtype=np.int32)
    idx_map[finite] = np.arange(finite.sum())

    n = int(finite.sum())
    if n < 4:
        return np.nan

    vals = spatial_map[finite].astype(np.float64)
    vals -= vals.mean()

    neighbour_sum = np.zeros(n, dtype=np.float64)
    w_count = np.zeros(n, dtype=np.float64)
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]

    for i, (y, x) in enumerate(zip(ys, xs)):
        for dyi, dxi in zip(dy, dx):
            ny_, nx_ = y + dyi, x + dxi
            if 0 <= ny_ < H and 0 <= nx_ < W:
                j = idx_map[ny_, nx_]
                if j >= 0:
                    neighbour_sum[i] += vals[j]
                    w_count[i] += 1.0

    W_total = w_count.sum()
    if W_total == 0:
        return np.nan
    numerator = (vals * neighbour_sum).sum()
    denominator = (vals * vals).sum()
    if denominator == 0:
        return np.nan
    return float(n / W_total * numerator / denominator)


def _compute_acf(time_course: np.ndarray, lag: int) -> float:
    """Lag-k autocorrelation via Pearson r between x[:-lag] and x[lag:]. See hfc_pilot.py."""
    x = time_course - time_course.mean()
    if len(x) <= lag:
        return np.nan
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def _spectral_peakiness(time_course: np.ndarray, fps: float) -> tuple[float, float]:
    """max(PSD)/mean(PSD) and the dominant peak frequency. See hfc_pilot.py."""
    from scipy.signal import welch
    n = len(time_course)
    nperseg = min(n, max(8, n // 4))
    freqs, psd = welch(time_course, fs=fps, nperseg=nperseg)
    mean_psd = psd.mean()
    if mean_psd == 0:
        return np.nan, np.nan
    peak_idx = int(np.argmax(psd))
    return float(psd[peak_idx] / mean_psd), float(freqs[peak_idx])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_spatial_component_grid(result: DecompositionResult, top_n: int, out: Path) -> None:
    """Top-N spatial maps, vessel mask contour overlaid."""
    top_idx = result.order[:top_n]
    ncols = min(top_n, 4)
    nrows = int(np.ceil(top_n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.32 * nrows),
                              constrained_layout=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for plot_i, comp_i in enumerate(top_idx):
        r, c = divmod(plot_i, ncols)
        ax = axes[r, c]
        smap = result.spatial[comp_i]
        vals = smap[np.isfinite(smap)]
        vmax = float(np.percentile(np.abs(vals), 98)) if vals.size else 1.0
        ax.imshow(smap, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        ax.contour(result.valid_mask.astype(float), levels=[0.5], colors='k', linewidths=0.5)
        metric_label = 'EVR' if result.ranking_name == 'explained_variance_ratio' else '|kurt|'
        ax.set_title(f'C{comp_i + 1}  {metric_label}={result.ranking_metric[comp_i]:.3f}',
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    for plot_i in range(top_n, nrows * ncols):
        r, c = divmod(plot_i, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f'{result.method.upper()} spatial components — {result.session_id}', fontsize=9)
    savefig(fig, out / 'fig_spatial_maps')
    plt.close(fig)


def fig_timecourses(result: DecompositionResult, top_n: int, out: Path) -> None:
    """Stacked line plot of top-N component time-courses."""
    top_idx = result.order[:top_n]
    t = np.arange(result.timecourse.shape[1]) / result.fps

    fig, axes = plt.subplots(top_n, 1, figsize=(_DOUBLE_COL, 1.1 * top_n),
                              constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)

    color = _METHOD_COLOR[result.method]
    for i, comp_i in enumerate(top_idx):
        ax = axes[i]
        ax.plot(t, result.timecourse[comp_i], color=color, lw=0.7)
        ax.set_ylabel(f'C{comp_i + 1}', fontsize=7, rotation=0, labelpad=15, va='center')
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel('Time (s)')

    fig.suptitle(f'{result.method.upper()} time-courses — {result.session_id}', fontsize=9)
    savefig(fig, out / 'fig_timecourses')
    plt.close(fig)


def fig_ranking(result: DecompositionResult, out: Path) -> None:
    """Scree plot (PCA, explained variance) or kurtosis bar plot (ICA)."""
    sorted_metric = result.ranking_metric[result.order]
    n = len(sorted_metric)

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.4), constrained_layout=True)
    color = _METHOD_COLOR[result.method]
    ax.bar(np.arange(n), sorted_metric, color=color, width=0.8)
    ax.set_xlabel('Component rank')
    if result.ranking_name == 'explained_variance_ratio':
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(f'PCA scree — {result.session_id}', fontsize=9)
    else:
        ax.set_ylabel('|Excess kurtosis|')
        ax.set_title(f'ICA non-Gaussianity ranking — {result.session_id}', fontsize=9)

    savefig(fig, out / 'fig_ranking')
    plt.close(fig)


def fig_pooled_ranking(results: list[DecompositionResult], out: Path) -> None:
    """Overlay every session's scree/kurtosis ranking curve to check whether the
    same small number of components dominates across sessions or is session-specific.
    """
    method = results[0].method
    ranking_name = results[0].ranking_name
    n_max = max(len(r.ranking_metric) for r in results)

    palette = np.concatenate([
        matplotlib.colormaps['tab20'].colors,
        matplotlib.colormaps['tab20b'].colors,
        matplotlib.colormaps['tab20c'].colors,
    ])
    if len(results) > len(palette):
        extra = matplotlib.colormaps['hsv'].resampled(len(results) - len(palette))
        palette = np.concatenate([palette, extra(np.arange(len(results) - len(palette)))[:, :3]])

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.4),
                                          constrained_layout=True)
    for i, r in enumerate(results):
        sorted_metric = r.ranking_metric[r.order]
        x = np.arange(len(sorted_metric))
        ax_lin.plot(x, sorted_metric, color=palette[i % len(palette)], lw=0.8, alpha=0.8,
                    label=r.session_id)
        ax_log.plot(x, sorted_metric, color=palette[i % len(palette)], lw=0.8, alpha=0.8)

    ylabel = 'Explained variance ratio' if ranking_name == 'explained_variance_ratio' else '|Excess kurtosis|'
    for ax in (ax_lin, ax_log):
        ax.set_xlabel('Component rank')
        ax.set_xlim(0, n_max - 1)
    ax_lin.set_ylabel(ylabel)
    ax_log.set_yscale('log')
    ax_log.set_ylabel(f'{ylabel} (log)')

    title = 'PCA scree' if ranking_name == 'explained_variance_ratio' else 'ICA non-Gaussianity ranking'
    fig.suptitle(f'{title} across sessions ({method.upper()}, n={len(results)})', fontsize=9)
    fig.legend(loc='outside right upper', fontsize=5, title='Session', title_fontsize=6)

    savefig(fig, out / 'fig_pooled_ranking')
    plt.close(fig)


def fig_pooled_k_selection(results: list[DecompositionResult], candidate_ks: list[int], out: Path) -> None:
    """Histogram of selected k across sessions, plus mean cumulative explained-variance-ratio
    vs k (PCA only) for a handful of candidate k values — summarizes the auto_pa scree spread
    from fig_pooled_ranking into concrete numbers for proposing a fixed n_components.
    """
    method = results[0].method
    selected_k = np.array([r.spatial.shape[0] for r in results])

    fig, (ax_hist, ax_evr) = plt.subplots(1, 2, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.4),
                                           constrained_layout=True)

    bins = np.arange(selected_k.min(), selected_k.max() + 2) - 0.5
    ax_hist.hist(selected_k, bins=bins, color=_METHOD_COLOR[method], edgecolor='white')
    ax_hist.axvline(np.median(selected_k), color='k', lw=0.8, ls='--',
                     label=f'median k={np.median(selected_k):.0f}')
    ax_hist.set_xlabel('Selected n_components (k)')
    ax_hist.set_ylabel('Number of sessions')
    ax_hist.set_title(f'Selected k across sessions (n={len(results)})', fontsize=9)
    ax_hist.legend(fontsize=7)

    if method == 'pca':
        candidate_ks = [k for k in candidate_ks if k <= max(len(r.ranking_metric) for r in results)]
        mean_evr = []
        for k in candidate_ks:
            per_session = []
            for r in results:
                sorted_evr = r.ranking_metric[r.order]
                k_eff = min(k, len(sorted_evr))
                per_session.append(float(np.sum(sorted_evr[:k_eff])))
            mean_evr.append(np.mean(per_session))
        ax_evr.plot(candidate_ks, mean_evr, color=_METHOD_COLOR[method], marker='o', ms=3, lw=1.0)
        for k, evr in zip(candidate_ks, mean_evr):
            ax_evr.annotate(f'{evr:.2f}', (k, evr), fontsize=6, xytext=(0, 4), textcoords='offset points',
                             ha='center')
        ax_evr.set_xlabel('n_components (k)')
        ax_evr.set_ylabel('Mean cumulative explained variance ratio')
        ax_evr.set_title('Cumulative EVR vs k, averaged across sessions', fontsize=9)
    else:
        ax_evr.set_visible(False)

    fig.suptitle(f'{method.upper()} n_components summary — auto_pa selection', fontsize=9)
    savefig(fig, out / 'fig_pooled_k_selection')
    plt.close(fig)

    print(f"\n  Selected k across {len(results)} sessions: "
          f"median={np.median(selected_k):.0f}, min={selected_k.min()}, max={selected_k.max()}, "
          f"mean={selected_k.mean():.1f}")
    if method == 'pca':
        for k, evr in zip(candidate_ks, mean_evr):
            print(f"    k={k:3d}: mean cumulative EVR = {evr:.3f}")


def fig_psd_per_component(result: DecompositionResult, top_n: int, freq_gate_hz: float,
                           respiration_hz: float, out: Path) -> None:
    """PSD of each top-N component time-course; frequency gates marked."""
    from scipy.signal import welch

    top_idx = result.order[:top_n]
    ncols = min(top_n, 3)
    nrows = int(np.ceil(top_n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.35 * nrows),
                              constrained_layout=True)
    axes = np.array(axes).reshape(nrows, ncols)

    n = result.timecourse.shape[1]
    nperseg = min(n, max(8, n // 4))

    for plot_i, comp_i in enumerate(top_idx):
        r, c = divmod(plot_i, ncols)
        ax = axes[r, c]
        freqs, psd = welch(result.timecourse[comp_i], fs=result.fps, nperseg=nperseg)
        ax.plot(freqs, psd, color=_METHOD_COLOR[result.method], lw=1.0)
        ax.axvline(freq_gate_hz, color='#2CA02C', lw=0.8, ls='--', label=f'gate {freq_gate_hz} Hz')
        ax.axvline(respiration_hz, color='#D62728', lw=0.8, ls=':', label=f'resp {respiration_hz} Hz')
        ax.set_title(f'C{comp_i + 1}', fontsize=7)
        ax.set_xlabel('Frequency (Hz)', fontsize=7)
        ax.set_ylabel('PSD', fontsize=7)
        ax.tick_params(labelsize=6)
        if plot_i == 0:
            ax.legend(fontsize=6)

    for plot_i in range(top_n, nrows * ncols):
        r, c = divmod(plot_i, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f'{result.method.upper()} component PSDs — {result.session_id}', fontsize=9)
    savefig(fig, out / 'fig_psd')
    plt.close(fig)


def fig_morans_vs_peakiness(result: DecompositionResult, out: Path) -> None:
    """Moran's I x spectral peakiness scatter, all components (reused from hfc_pilot QPP figure)."""
    n_comp = result.spatial.shape[0]
    morans = np.array([_morans_i(result.spatial[i]) for i in range(n_comp)])
    peakiness = np.array([_spectral_peakiness(result.timecourse[i], result.fps)[0]
                           for i in range(n_comp)])

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.55), constrained_layout=True)
    ax.scatter(morans, peakiness, color=_METHOD_COLOR[result.method], s=16, alpha=0.75)
    for i in range(n_comp):
        ax.annotate(str(i + 1), (morans[i], peakiness[i]), fontsize=6,
                    xytext=(2, 2), textcoords='offset points')
    ax.set_xlabel("Moran's I (spatial coherence)")
    ax.set_ylabel('Spectral peakiness  [max(PSD)/mean(PSD)]')
    ax.set_title(f"{result.method.upper()} — Moran's I vs peakiness — {result.session_id}", fontsize=9)

    savefig(fig, out / 'fig_morans_vs_peakiness')
    plt.close(fig)


def fig_pca_vs_ica(result_pca: DecompositionResult, result_ica: DecompositionResult,
                    n_components: int, out: Path) -> None:
    """Matched spatial maps, same session, same n_components, PCA vs ICA side-by-side."""
    n_show = min(n_components, len(result_pca.order), len(result_ica.order))
    fig, axes = plt.subplots(2, n_show, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.32 * 2),
                              constrained_layout=True)
    axes = np.array(axes).reshape(2, n_show)

    for row, result in enumerate([result_pca, result_ica]):
        top_idx = result.order[:n_show]
        for col, comp_i in enumerate(top_idx):
            ax = axes[row, col]
            smap = result.spatial[comp_i]
            vals = smap[np.isfinite(smap)]
            vmax = float(np.percentile(np.abs(vals), 98)) if vals.size else 1.0
            ax.imshow(smap, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if col == 0:
                ax.set_ylabel(result.method.upper(), fontsize=8)
            ax.set_title(f'rank {col + 1}', fontsize=7)

    fig.suptitle(f'PCA vs ICA — {result_pca.session_id}', fontsize=9)
    savefig(fig, out / 'fig_pca_vs_ica')
    plt.close(fig)


def fig_pooled_stability(results: list[DecompositionResult], top_n: int, out: Path) -> None:
    """Cross-session correlation matrix between top-N component spatial maps.

    Tests whether components are subject-general (block-diagonal breaks down)
    or session-specific (only diagonal is high). Only meaningful when
    sessions share a common valid_mask footprint (or are compared on their
    intersection).
    """
    n_sess = len(results)
    if n_sess < 2:
        return

    common_mask = results[0].valid_mask.copy()
    for r in results[1:]:
        common_mask &= r.valid_mask
    if common_mask.sum() < 4:
        warnings.warn("fig_pooled_stability: insufficient common valid pixels across sessions; skipping.")
        return

    vecs = []
    session_ids = []
    block_sizes = []
    for r in results:
        comps = r.order[:top_n]
        for comp_i in comps:
            vecs.append(r.spatial[comp_i][common_mask])
        session_ids.append(r.session_id)
        block_sizes.append(len(comps))
    vecs = np.array(vecs)
    corr = np.corrcoef(vecs)
    n_total = corr.shape[0]

    # tick positions at the center of each session's block, for readable labels
    boundaries = np.cumsum([0] + block_sizes)
    centers = (boundaries[:-1] + boundaries[1:]) / 2 - 0.5

    fig, ax = plt.subplots(figsize=(_DOUBLE_COL, _DOUBLE_COL), constrained_layout=True)
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

    # session-boundary separators make the block structure legible
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='k', lw=0.4, alpha=0.5)
        ax.axvline(b - 0.5, color='k', lw=0.4, alpha=0.5)

    ax.set_xticks(centers); ax.set_xticklabels(session_ids, rotation=90, fontsize=5)
    ax.set_yticks(centers); ax.set_yticklabels(session_ids, fontsize=5)
    ax.set_xlim(-0.5, n_total - 0.5)
    ax.set_ylim(n_total - 0.5, -0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='Pearson r')
    ax.set_title(f'Cross-session spatial-map correlation (top-{top_n} components/session, common mask)\n'
                 'Thin lines mark session boundaries; block-diagonal-only = session-specific components',
                 fontsize=8)

    savefig(fig, out / 'fig_pooled_stability')
    plt.close(fig)


def fig_spatial_temporal_tradeoff(results: list[DecompositionResult], top_n: int, lags: list[int],
                                   out: Path) -> None:
    """Moran's I (x) vs lag-k ACF (y), one panel per lag, dots = top-N components/session.

    Colour = session, one unique colour per session (tab20+tab20b+tab20c chained,
    since tab20 alone only has 20 distinct colours and sessions can exceed that).
    Upper-right quadrant (high Moran's I AND high ACF) = QPP-compatible components.
    Mirrors hfc_pilot.py's _fig_spatial_temporal_tradeoff, but run per-method since
    decompose.py processes one method (pca or ica) per invocation.
    """
    session_ids = [r.session_id for r in results]
    palette = np.concatenate([
        matplotlib.colormaps['tab20'].colors,
        matplotlib.colormaps['tab20b'].colors,
        matplotlib.colormaps['tab20c'].colors,
    ])
    if len(session_ids) > len(palette):
        extra = matplotlib.colormaps['hsv'].resampled(len(session_ids) - len(palette))
        palette = np.concatenate([palette, extra(np.arange(len(session_ids) - len(palette)))[:, :3]])
    sess_color = {sid: palette[i % len(palette)] for i, sid in enumerate(session_ids)}
    method = results[0].method

    pool_morans, pool_acf, pool_sess = [], [], []
    for r in results:
        comps = r.order[:top_n]
        for comp_i in comps:
            pool_morans.append(_morans_i(r.spatial[comp_i]))
            pool_acf.append([_compute_acf(r.timecourse[comp_i], lag) for lag in lags])
            pool_sess.append(r.session_id)
    pool_morans = np.array(pool_morans)
    pool_acf = np.array(pool_acf)          # (N, len(lags))
    colors = [sess_color[s] for s in pool_sess]

    mi_med = float(np.nanmedian(pool_morans))

    n_panels = len(lags)
    ncols = min(n_panels, 2)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.55 * nrows),
                              constrained_layout=True, squeeze=False)

    print(f"\n  {method.upper()} spatial/temporal tradeoff — Moran's I vs lag-k ACF:")

    for panel_i, lag in enumerate(lags):
        row, col = divmod(panel_i, ncols)
        ax = axes[row, col]
        acf_k = pool_acf[:, panel_i]

        valid = np.isfinite(pool_morans) & np.isfinite(acf_k)
        ax.scatter(pool_morans[valid], acf_k[valid],
                   c=[c for c, v in zip(colors, valid) if v],
                   s=10, alpha=0.6, linewidths=0)

        acf_med = float(np.nanmedian(acf_k))
        in_qpp = (pool_morans > mi_med) & (acf_k > acf_med) & valid
        n_qpp = int(in_qpp.sum())
        n_valid = int(valid.sum())

        ax.axhline(0, color='0.6', lw=0.7, ls=':')
        ax.axvline(0, color='0.6', lw=0.7, ls=':')
        ax.axhline(acf_med, color='#A0A0A0', lw=0.7, ls='--')
        ax.axvline(mi_med, color='#A0A0A0', lw=0.7, ls='--')

        ax.set_xlabel("Moran's I (spatial coherence)", fontsize=8)
        ax.set_ylabel(f'Lag-{lag} ACF (temporal predictability)', fontsize=8)
        ax.set_title(f"Lag-{lag} ACF   (n={n_valid})\nQPP quadrant (>median both axes): {n_qpp} components",
                     fontsize=7)

        print(f"    Lag-{lag:2d}:  QPP quadrant n={n_qpp}/{n_valid}")

    for panel_i in range(n_panels, nrows * ncols):
        row, col = divmod(panel_i, ncols)
        axes[row, col].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color=sess_color[sid], lw=0, ms=4, label=sid)
               for sid in session_ids]
    fig.legend(handles=handles, loc='outside right upper', fontsize=5, title='Session', title_fontsize=6)

    fig.suptitle(
        f'{method.upper()} — QPP diagnostic: upper-right quadrant = high spatial coherence AND temporal predictability\n'
        'Populated → QPP prior may be useful; empty → QPP prior unlikely to help',
        fontsize=9,
    )
    savefig(fig, out / 'fig_spatial_temporal_tradeoff')
    plt.close(fig)


def _reconstruct_top_k(result: DecompositionResult, top_k: int) -> np.ndarray:
    """Additive top-k reconstruction from a DecompositionResult, (T, H, W)."""
    top_idx = result.order[:top_k]
    H, W = result.recon_basis.shape[1:]
    T = result.timecourse.shape[1]
    recon = np.zeros((T, H, W), dtype=np.float64)
    for comp_i in top_idx:
        smap = np.nan_to_num(result.recon_basis[comp_i])
        recon += np.outer(result.timecourse[comp_i], smap.ravel()).reshape(T, H, W)
    return recon


def fig_acf_reconstruction_across_sessions(results: list[DecompositionResult], top_k: int, out: Path,
                                            lags: list[int] = (1,)) -> None:
    """Lag-k ACF distribution of the top-k reconstruction, pooled across sessions.

    One row of [KDE, per-session median] panels per lag in ``lags``.
    Mirrors characterize.py's _figX_acf_distribution_across_sessions, but computed
    on the per-pixel top-k PCA/ICA reconstruction rather than the raw frames — this
    shows how much lag-k structure survives after keeping only the top-k components.
    """
    from scipy.stats import gaussian_kde

    method = results[0].method
    lags = list(lags)
    n_lags = len(lags)

    fig, axes = plt.subplots(n_lags, 2, figsize=(_DOUBLE_COL, _DOUBLE_COL * 0.5 * n_lags),
                              constrained_layout=True, squeeze=False)
    cmap = plt.colormaps['tab20'].resampled(len(results))
    sess_color = {r.session_id: cmap(i) for i, r in enumerate(results)}
    x_grid = np.linspace(-0.5, 1.0, 400)

    for row, lag in enumerate(lags):
        ax_kde, ax_med = axes[row]
        records = []
        for r in results:
            recon = _reconstruct_top_k(r, top_k)
            corr, _ = safe_temporal_corr_map(recon[:-lag], recon[lag:])
            vals = corr[r.valid_mask]
            vals = vals[np.isfinite(vals)]
            if vals.size < 10:
                continue
            records.append({'session_id': r.session_id, 'vals': vals, 'median': float(np.median(vals))})

        if not records:
            warnings.warn(f"fig_acf_reconstruction_across_sessions: no sessions with valid ACF values at lag {lag}; skipping row.")
            ax_kde.set_visible(False)
            ax_med.set_visible(False)
            continue

        for r in records:
            try:
                kde = gaussian_kde(r['vals'], bw_method='scott')
                ax_kde.plot(x_grid, kde(x_grid), lw=1.0, color=sess_color[r['session_id']],
                            alpha=0.75, label=r['session_id'])
            except Exception:
                pass

        ax_kde.axvline(0.5, color='#8B1E1E', ls='--', lw=0.8)
        ax_kde.set_xlabel(f'Lag-{lag} autocorrelation')
        ax_kde.set_ylabel('Density')
        ax_kde.set_xlim(-0.4, 0.9)
        ax_kde.set_title(f'Lag-{lag} ACF distribution per session', fontsize=8)
        if len(records) <= 12:
            ax_kde.legend(fontsize=6, ncol=2, loc='upper right')

        medians = [r['median'] for r in records]
        sids = [r['session_id'] for r in records]
        order = np.argsort(medians)
        ax_med.scatter([medians[i] for i in order], range(len(order)),
                        color=[sess_color[sids[i]] for i in order], s=25, zorder=3)
        ax_med.axvline(float(np.median(medians)), color='#1E3A5F', ls='--', lw=1,
                        label=f'grand median {np.median(medians):.3f}')
        if len(order) <= 30:
            ax_med.set_yticks(range(len(order)))
            ax_med.set_yticklabels([sids[i] for i in order], fontsize=7)
        else:
            ax_med.set_ylabel(f'Session rank (n={len(order)})')
            ax_med.set_yticks([])
        ax_med.set_xlabel(f'Median lag-{lag} ACF')
        ax_med.legend(fontsize=7)
        ax_med.set_title(f'Lag-{lag} per-session median', fontsize=8)

    fig.suptitle(f'{method.upper()} top-{top_k} reconstruction — ACF across sessions', fontsize=9)
    savefig(fig, out / 'fig_acf_reconstruction_across_sessions')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def _write_frames_to_video(fig, update_fn, n_frames: int, out_path: Path, fps: float) -> Path:
    """
    Render n_frames via update_fn(frame_i) and write them to an .mp4 with
    cv2.VideoWriter — same approach as make_triplet_video in
    fuspredict.evaluation.visualization, so no ffmpeg binary is required.
    """
    import cv2

    out_path = Path(out_path).with_suffix('.mp4')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # probe frame dimensions
    update_fn(0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w_px, h_px = fig.canvas.get_width_height()
    h_px, w_px = buf.reshape(h_px, w_px, 4).shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_px, h_px))

    for frame_i in range(n_frames):
        update_fn(frame_i)
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)
        writer.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))

    writer.release()
    return out_path


def render_reconstruction_video(session: Session, result: DecompositionResult,
                                 top_k: int, out_path: Path, speed: float = 1.0) -> Path:
    """Three-panel video: original | top-k reconstruction | residual."""
    top_idx = result.order[:top_k]
    recon = np.zeros((session.n_frames, session.height, session.width), dtype=np.float64)
    for comp_i in top_idx:
        smap = np.nan_to_num(result.recon_basis[comp_i])
        recon += np.outer(result.timecourse[comp_i], smap.ravel()).reshape(recon.shape)

    original = session.frames.astype(np.float64)
    residual = original - recon

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True, dpi=120)
    vmax = float(np.nanpercentile(np.abs(original), 98))
    ims = [
        axes[0].imshow(original[0], cmap='gray', vmin=-vmax, vmax=vmax),
        axes[1].imshow(recon[0], cmap='gray', vmin=-vmax, vmax=vmax),
        axes[2].imshow(residual[0], cmap='RdBu_r', vmin=-vmax, vmax=vmax),
    ]
    titles = ['Original', f'Reconstruction (top-{top_k})', 'Residual']
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    def update(frame_i):
        ims[0].set_data(original[frame_i])
        ims[1].set_data(recon[frame_i])
        ims[2].set_data(residual[frame_i])
        fig.suptitle(f'{session.id} — frame {frame_i}/{session.n_frames}', fontsize=9)

    saved_path = _write_frames_to_video(fig, update, session.n_frames, out_path, fps=session.fps * speed)
    plt.close(fig)
    return saved_path


def render_component_activation_video(session: Session, result: DecompositionResult,
                                       component_index: int, out_path: Path, speed: float = 1.0) -> Path:
    """Single-component activation video: spatial_map * timecourse(t)."""
    smap = np.nan_to_num(result.recon_basis[component_index])
    tc = result.timecourse[component_index]
    activation = tc[:, None, None] * smap[None, :, :]

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True, dpi=120)
    vmax = float(np.percentile(np.abs(activation), 98))
    im = ax.imshow(activation[0], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{session.id} — C{component_index + 1}', fontsize=9)

    def update(frame_i):
        im.set_data(activation[frame_i])

    saved_path = _write_frames_to_video(fig, update, session.n_frames, out_path, fps=session.fps * speed)
    plt.close(fig)
    return saved_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='Path to decompose YAML config')
    parser.add_argument('--project-config', default='config.yml',
                         help='Project config filename in config/ (selects subject, e.g. config_gus.yml)')
    parser.add_argument('--sessions', nargs='*', default=None, help='Override session ID subset')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('--top-n', type=int, default=8, help='Top-N components shown in figures')
    parser.add_argument('--acf-lags', type=int, nargs='*', default=[1, 5, 10, 20, 30, 40, 50],
                         help='Lags (frames) shown in fig_lag_acf_morans_by_session')
    parser.add_argument('--acf-recon-lags', type=int, nargs='*', default=[1],
                         help='Lag(s) (frames) for fig_acf_reconstruction_across_sessions, e.g. --acf-recon-lags 1 5 10')
    parser.add_argument('--video', choices=['reconstruction', 'component', 'all-components', 'none'], default='none')
    parser.add_argument('--video-session', default=None, help='Session ID to render video for')
    parser.add_argument('--video-component', type=int, default=0, help='Component index for --video component')
    parser.add_argument('--video-speed', type=float, default=1.0, help='Playback speed multiplier')
    args = parser.parse_args()

    cfg = DecomposeConfig.from_yaml(args.config)
    if args.sessions:
        cfg.sessions = args.sessions

    repo_root = find_repo_root()
    project_cfg = load_project_config(repo_root, config_name=args.project_config)
    subject = project_cfg['subjects']['all'][0]
    exclude_ids = get_excluded_sessions(project_cfg, subject, cfg.exclude_sessions)

    preproc_root = repo_root / project_cfg['paths']['preprocessing'] / subject
    standardized_dir = Path(cfg.standardized_dir) if cfg.standardized_dir else \
        preproc_root / 'baseline_only_standardized'
    mask_dir = Path(cfg.mask_dir) if cfg.mask_dir else \
        preproc_root / 'tissue_masks'
    out_dir = repo_root / cfg.output_dir / cfg.method

    sessions = load_sessions(standardized_dir, mask_dir=mask_dir, exclude_ids=exclude_ids)
    if cfg.sessions:
        wanted = set(cfg.sessions)
        sessions = [s for s in sessions if s.id in wanted]
    if not sessions:
        raise SystemExit("No sessions to process — check config paths / session filters.")

    print(f"Decomposing {len(sessions)} sessions with method={cfg.method}, "
          f"n_components={cfg.n_components}, mask={cfg.mask}, "
          f"calibration_frames={cfg.calibration_frames}")

    results: list[DecompositionResult] = []
    for session in sessions:
        try:
            result = decompose_session(session, cfg)
        except Exception as exc:
            warnings.warn(f"Skipping {session.id}: {exc}", stacklevel=2)
            continue
        session_dir = out_dir / session.id
        out_path = save_result(result, session_dir)
        print(f"  {session.id}: saved {out_path.name}  "
              f"(n_components={result.spatial.shape[0]}, T={result.timecourse.shape[1]})")
        results.append(result)

        if not args.no_figures:
            top_n = min(args.top_n, result.spatial.shape[0])
            fig_spatial_component_grid(result, top_n, session_dir)
            fig_timecourses(result, top_n, session_dir)
            fig_ranking(result, session_dir)
            fig_psd_per_component(result, top_n, cfg.freq_gate_hz, 0.448, session_dir)
            fig_morans_vs_peakiness(result, session_dir)
            if result.n_components_method == "auto_cv":
                candidates, mean_rmse = result.n_components_selection
                fig_component_selection(candidates, mean_rmse, result.spatial.shape[0],
                                         result.session_id, session_dir)
            elif result.n_components_method == "auto_pa":
                real_eig, null_thresh, null_eig = result.n_components_selection
                fig_parallel_analysis(real_eig, null_thresh, null_eig, result.spatial.shape[0],
                                       result.session_id, session_dir)

    if not args.no_figures and len(results) > 1:
        min_n_comp = min(r.spatial.shape[0] for r in results)
        fig_pooled_ranking(results, out=out_dir)
        if results[0].n_components_method is not None:
            fig_pooled_k_selection(results, candidate_ks=[2, 4, 6, 8, 10, 15, 20], out=out_dir)
        fig_pooled_stability(results, top_n=min(args.top_n, min_n_comp), out=out_dir)
        fig_spatial_temporal_tradeoff(results, top_n=min(args.top_n, min_n_comp),
                                       lags=args.acf_lags, out=out_dir)
        fig_acf_reconstruction_across_sessions(results, top_k=min(args.top_n, min_n_comp), out=out_dir,
                                                lags=args.acf_recon_lags)

    if args.video != 'none':
        if not args.video_session:
            raise SystemExit("--video requires --video-session")
        session = next((s for s in sessions if s.id == args.video_session), None)
        result = next((r for r in results if r.session_id == args.video_session), None)
        if session is None or result is None:
            raise SystemExit(f"Session {args.video_session} not found among processed sessions.")
        video_dir = out_dir / session.id
        if args.video == 'reconstruction':
            out_path = video_dir / 'reconstruction.mp4'
            saved_path = render_reconstruction_video(session, result, args.top_n, out_path,
                                                       speed=args.video_speed)
            print(f"Saved video: {saved_path}")
        elif args.video == 'component':
            out_path = video_dir / f'component_{args.video_component}.mp4'
            saved_path = render_component_activation_video(session, result, args.video_component, out_path,
                                                             speed=args.video_speed)
            print(f"Saved video: {saved_path}")
        else:  # all-components
            n_comp = result.spatial.shape[0]
            for comp_i in range(n_comp):
                out_path = video_dir / f'component_{comp_i}.mp4'
                saved_path = render_component_activation_video(session, result, comp_i, out_path,
                                                                 speed=args.video_speed)
                print(f"Saved video: {saved_path}")

    print(f"\nDone. {len(results)}/{len(sessions)} sessions decomposed. Output: {out_dir}")


if __name__ == '__main__':
    main()
