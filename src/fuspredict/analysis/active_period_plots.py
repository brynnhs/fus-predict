"""
active_period_plots.py
----------------------
Plotting functions for active-period analysis.

Every function takes arrays and an output path — no data loading,
no config, no model inference. Uses Agg backend (non-interactive).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .active_period import sigma_crossings as _sigma_crossings

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family":    "serif",
        "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":      9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi":     300,
        "savefig.dpi":    300,
        "savefig.bbox":   "tight",
    }
)

_SIGMA_COLORS = ["#ffdddd", "#ffaaaa", "#ff7777"]


def fig_activation_delta(
    delta_map: np.ndarray,
    roi_mask: np.ndarray,
    session_id: str,
    out_path: str | Path,
) -> None:
    """Save a two-panel figure: delta map alone and with ROI overlay.

    Parameters
    ----------
    delta_map : np.ndarray, shape (H, W)
        log10 activation delta (post-onset mean - baseline mean).
    roi_mask : np.ndarray, shape (H, W), bool
    session_id : str
    out_path : str or Path
        Output file path (PNG or PDF).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmax = max(float(np.percentile(np.abs(delta_map), 98)), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im = axes[0].imshow(delta_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="Δlog10 power")
    axes[0].set_title("Activation delta\n(post-onset mean − baseline mean)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(delta_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    axes[1].contour(roi_mask.astype(float), levels=[0.5], colors=["yellow"], linewidths=[1.5])
    roi_patch = mpatches.Patch(edgecolor="yellow", facecolor="none", linewidth=1.5, label="ROI")
    axes[1].legend(handles=[roi_patch], loc="lower right", fontsize=8)
    axes[1].set_title(f"ROI overlay  ({roi_mask.sum()} px)", fontsize=9)
    axes[1].axis("off")

    fig.suptitle(f"Session {session_id} — activation delta", fontsize=10)
    fig.savefig(out_path)
    plt.close(fig)


def fig_roi_timeseries(
    signal_pct: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    fps: float,
    session_id: str,
    out_path: str | Path,
    window_s: float = 15.0,
    pre_signal: np.ndarray | None = None,
) -> None:
    """Save a time-series figure of ROI-averaged z-score around active-period onset.

    Parameters
    ----------
    signal_pct : np.ndarray, shape (T,)
        ROI-averaged z-score for the task period (frame 0 = onset).
    baseline_mean : float
    baseline_std : float
    fps : float
    session_id : str
    out_path : str or Path
    window_s : float
        Seconds to display before and after onset.
    pre_signal : np.ndarray, shape (T_pre,), optional
        ROI-averaged z-score for the pre-onset baseline period, in chronological
        order (last frame = one frame before onset). If provided, window_s seconds
        of pre-onset signal are shown to the left of t=0.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_post = min(len(signal_pct), int(round(window_s * fps)))
    sig_post = signal_pct[:n_post]
    t_post   = np.arange(n_post) / fps

    if pre_signal is not None and len(pre_signal) > 0:
        n_pre    = min(len(pre_signal), int(round(window_s * fps)))
        sig_pre  = pre_signal[-n_pre:]
        t_pre    = np.arange(-n_pre, 0) / fps
        t_all    = np.concatenate([t_pre, t_post])
        sig_all  = np.concatenate([sig_pre, sig_post])
    else:
        t_all   = t_post
        sig_all = sig_post

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    band_cap   = baseline_mean + 3.5 * baseline_std
    band_edges = [
        baseline_mean,
        baseline_mean + baseline_std,
        baseline_mean + 2 * baseline_std,
        band_cap,
    ]
    for i, col in enumerate(_SIGMA_COLORS):
        ax.axhspan(band_edges[i], band_edges[i + 1], color=col, lw=0, zorder=1)

    ax.plot(t_all, sig_all, color="#2ca02c", lw=2.0, zorder=5)
    ax.axhline(baseline_mean, color="#888888", lw=1.0, ls=":", zorder=2)
    ax.axhline(0, color="#cccccc", lw=0.7, ls="-", zorder=1)
    ax.axvline(0, color="black", lw=1.2, ls="--", zorder=6)

    x_right = t_all[-1]
    for n, edge in zip((1, 2, 3), band_edges[1:]):
        y_label = (band_edges[n - 1] + edge) / 2
        ax.text(x_right, y_label, f" +{n}σ", va="center", ha="left",
                fontsize=8, color="#cc3333", clip_on=False)

    y_bot = float(sig_all.min()) - 0.5 * baseline_std
    y_top = max(float(sig_all.max()), band_cap) + 0.5 * baseline_std
    ax.set_xlim(t_all[0], t_all[-1])
    ax.set_ylim(y_bot, y_top)
    ax.set_xlabel("Time relative to onset (s)", fontsize=11)
    ax.set_ylabel("ROI-mean z-score (σ)", fontsize=11)
    ax.set_title(
        f"{session_id} — ROI signal around active-period onset\n"
        f"baseline: mean={baseline_mean:.2f}σ, std={baseline_std:.2f}σ  |  ROI size read from mask",
        fontsize=10,
    )
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.2)

    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers shared by fig_transition_analysis
# ---------------------------------------------------------------------------

def _rolling_mean(arr: np.ndarray, k: int = 10) -> np.ndarray:
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = np.nanmean(arr[max(0, i - k + 1) : i + 1])
    return out



def _roi_wavg(
    diff_roi_flat: np.ndarray,
    w_flat: np.ndarray | None,
) -> float:
    if w_flat is None:
        return float(np.abs(diff_roi_flat).mean())
    return float((np.abs(diff_roi_flat) * w_flat).sum())


def _roi_mean(frames: np.ndarray, roi_mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    flat     = frames.reshape(frames.shape[0], -1)
    roi_px   = flat[:, roi_mask.ravel()]
    if weights is None:
        return roi_px.mean(axis=1)
    return (roi_px * weights[None, :]).sum(axis=1)


# ---------------------------------------------------------------------------
# Dizeux-style transition analysis figure
# ---------------------------------------------------------------------------

def fig_transition_analysis(
    baseline_frames_log10: np.ndarray,
    task_frames_log10: np.ndarray,
    roi_mask: np.ndarray,
    mean_map: np.ndarray,
    std_map: np.ndarray,
    fps: float,
    session_id: str,
    out_path: str | Path,
    checkpoint_path: str | Path | None = None,
    context_s: float = 30.0,
    lag: int = 10,
    hidden_channels: int = 32,
    kernel_size: int = 3,
) -> None:
    """Save a Dizeux-style transition analysis figure (% CBV).

    Panels
    ------
    Row 0  Spatial snapshots at t₀, t₀+1.5 s, t₀+2.5 s:
           left = GT % CBV frame, right = GT − ConvLSTM residual.
    Row 1  ROI time series: GT % CBV + ConvLSTM prediction + rolling-mean
           prediction, with σ bands derived from pre-onset baseline.
    Row 2  Prediction residuals centred on pre-onset mean.
    Row 3  σ-crossing time table.

    Parameters
    ----------
    baseline_frames_log10 : np.ndarray, shape (T_bl, H, W)
        Log10 baseline frames (z-scored baseline pipeline output, in log10
        space before z-scoring — i.e. ``session.frames`` before zscore_frames).
        Used to compute the % CBV reference mean and baseline σ statistics.
    task_frames_log10 : np.ndarray, shape (T_task, H, W)
        Log10 task frames (reoriented/resized, not z-scored).
    roi_mask : np.ndarray, shape (H, W), bool
    mean_map : np.ndarray, shape (H, W)
        Per-pixel baseline log10 mean (from standardized .nc).
    std_map : np.ndarray, shape (H, W)
        Per-pixel baseline log10 std (from standardized .nc).
    fps : float
    session_id : str
    out_path : str or Path
    checkpoint_path : str, Path, or None
        Path to a ``model.pt`` saved by ConvLSTMPredictor (horizon=1).
        If None, ConvLSTM panels are left blank.
    context_s : float
        Seconds of baseline tail and task frames shown either side of t₀.
    lag : int
        ConvLSTM context window length (must match checkpoint).
    hidden_channels : int
        ConvLSTM hidden channels (must match checkpoint).
    kernel_size : int
        ConvLSTM kernel size (must match checkpoint).
    """
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ──────────────────────────────────────────────────────────
    # Checkpoints saved by 11_roi_from_transition.py use _FullFrameConvLSTM
    # (keys: cell.gates.*, head.*).  Define a compatible architecture inline.
    import torch.nn as nn

    class _GatesCell(nn.Module):
        def __init__(self, in_ch: int, hidden: int, ks: int = 3) -> None:
            super().__init__()
            self.hidden_channels = hidden
            self.gates = nn.Conv2d(in_ch + hidden, 4 * hidden, ks, padding=ks // 2)
            nn.init.constant_(self.gates.bias[hidden : 2 * hidden], 1.0)

        def forward(self, x, h, c):
            i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
            c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            return torch.sigmoid(o) * torch.tanh(c_next), c_next

        def init_hidden(self, B: int, H: int, W: int):
            z = torch.zeros(B, self.hidden_channels, H, W,
                            device=next(self.parameters()).device)
            return z, torch.zeros_like(z)

    class _FullFrameConvLSTM(nn.Module):
        def __init__(self, hidden: int = 32) -> None:
            super().__init__()
            self.cell = _GatesCell(1, hidden)
            self.head = nn.Conv2d(hidden, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, H, W = x.shape
            h, c = self.cell.init_hidden(B, H, W)
            for t in range(T):
                h, c = self.cell(x[:, t:t+1], h, c)
            return self.head(h)

    model = None
    if checkpoint_path is not None:
        m = _FullFrameConvLSTM(hidden_channels).to(device)
        m.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
        m.eval()
        model = m

    # ── % CBV conversion ────────────────────────────────────────────────────
    # Reference: mean over ALL baseline frames in linear space
    bl_mean_linear = np.power(10.0, baseline_frames_log10).mean(axis=0).astype(np.float32)

    def _to_pct(log10_frames: np.ndarray) -> np.ndarray:
        return (np.power(10.0, log10_frames) / bl_mean_linear[None] - 1.0) * 100.0

    task_pct = _to_pct(task_frames_log10)
    bl_pct   = _to_pct(baseline_frames_log10)

    # ── z-scored task frames for model input ────────────────────────────────
    _ZSCORE_EPS = 1e-8
    safe_scale  = np.where(std_map > _ZSCORE_EPS, std_map, _ZSCORE_EPS).astype(np.float32)
    task_z      = ((task_frames_log10 - mean_map[None]) / safe_scale[None]).astype(np.float32)

    # ── ROI weights (delta-weighted, fall back to uniform) ───────────────────
    delta_map  = task_frames_log10[:20].mean(axis=0) - mean_map
    delta_roi  = np.where(roi_mask, np.clip(delta_map, 0, None), 0.0)
    weight_sum = delta_roi.sum()
    if weight_sum > 0:
        w_flat = (delta_roi[roi_mask] / weight_sum).astype(np.float32)
    else:
        w_flat = None

    roi_flat = roi_mask.ravel()

    # ── baseline σ stats (from baseline % CBV ROI signal) ───────────────────
    bl_roi_signal = _roi_mean(bl_pct, roi_mask, weights=w_flat)
    bl_mean_pct   = float(bl_roi_signal.mean())
    bl_std_pct    = float(bl_roi_signal.std())

    # ── full concatenated frames for the time-series window ─────────────────
    # t=0 is defined as first task frame (frame index = T_bl in concat)
    all_log10 = np.concatenate([baseline_frames_log10, task_frames_log10], axis=0)
    all_pct   = np.concatenate([bl_pct, task_pct], axis=0)
    all_z     = np.concatenate([
        ((baseline_frames_log10 - mean_map[None]) / safe_scale[None]).astype(np.float32),
        task_z,
    ], axis=0)
    T_bl   = baseline_frames_log10.shape[0]
    T_all  = all_log10.shape[0]
    onset  = T_bl   # index of first task frame = t₀

    context_frames = int(round(context_s * fps))
    t_start = max(lag, onset - context_frames)
    t_end   = min(T_all, onset + context_frames + 1)

    t_axis   = np.arange(T_all)
    t_rel_all = (t_axis - onset) / fps

    # ── GT ROI signal (full concat) ─────────────────────────────────────────
    roi_signal_full = _roi_mean(all_pct, roi_mask, weights=w_flat)

    # ── model predictions and residuals over the window ─────────────────────
    pred_cl_roi   = np.full(T_all, np.nan, dtype=np.float32)
    pred_rm_roi   = np.full(T_all, np.nan, dtype=np.float32)
    resid_cl_roi  = np.full(T_all, np.nan, dtype=np.float32)
    resid_rm_roi  = np.full(T_all, np.nan, dtype=np.float32)

    with torch.no_grad():
        for t in range(t_start, t_end):
            # rolling-mean baseline prediction in % CBV
            rm_log  = all_log10[max(0, t - lag) : t].mean(axis=0)
            rm_pct  = _to_pct(rm_log[None])[0]
            pred_rm_roi[t] = _roi_mean(rm_pct[None], roi_mask, weights=w_flat)[0]
            diff_rm = all_pct[t].ravel()[roi_flat] - rm_pct.ravel()[roi_flat]
            resid_rm_roi[t] = _roi_wavg(diff_rm, w_flat)

            if model is not None and t >= lag:
                ctx = torch.from_numpy(all_z[t - lag : t]).unsqueeze(0).to(device)
                pred_z_map = model(ctx)[0, 0].cpu().numpy().astype(np.float32)
                pred_log   = pred_z_map * safe_scale + mean_map
                pred_pct_map = _to_pct(pred_log[None])[0]
                pred_cl_roi[t] = _roi_mean(pred_pct_map[None], roi_mask, weights=w_flat)[0]
                diff_cl = all_pct[t].ravel()[roi_flat] - pred_pct_map.ravel()[roi_flat]
                resid_cl_roi[t] = _roi_wavg(diff_cl, w_flat)

    # ── σ crossings ──────────────────────────────────────────────────────────
    crossings_gt = _sigma_crossings(
        roi_signal_full[onset:], bl_mean_pct, bl_std_pct, fps
    )

    pre_in_window = np.where(
        (t_axis >= t_start) & (t_axis < onset)
    )[0]

    rm_bl_vals = resid_rm_roi[pre_in_window]
    rm_bl_vals = rm_bl_vals[~np.isnan(rm_bl_vals)]
    if len(rm_bl_vals) > 1:
        crossings_rm = _sigma_crossings(
            _rolling_mean(resid_rm_roi)[onset:],
            float(rm_bl_vals.mean()), float(rm_bl_vals.std()), fps,
        )
    else:
        crossings_rm: dict[int, float | None] = {1: None, 2: None, 3: None}

    cl_bl_vals = resid_cl_roi[pre_in_window]
    cl_bl_vals = cl_bl_vals[~np.isnan(cl_bl_vals)]
    if model is not None and len(cl_bl_vals) > 1:
        crossings_cl = _sigma_crossings(
            _rolling_mean(resid_cl_roi)[onset:],
            float(cl_bl_vals.mean()), float(cl_bl_vals.std()), fps,
        )
    else:
        crossings_cl: dict[int, float | None] = {1: None, 2: None, 3: None}

    # ── spatial snapshots ────────────────────────────────────────────────────
    snap_offsets_s = [0.0, 1.5, 2.5]
    snap_frames    = [min(T_all - 1, onset + round(s * fps)) for s in snap_offsets_s]

    snap_pct_stack = np.stack([all_pct[t] for t in snap_frames])
    snap_vmax_gt   = max(float(np.percentile(np.abs(snap_pct_stack), 98)), 1e-6)

    resid_snaps: list[np.ndarray | None] = []
    for t in snap_frames:
        if model is not None and t >= lag:
            with torch.no_grad():
                ctx = torch.from_numpy(all_z[t - lag : t]).unsqueeze(0).to(device)
                pred_z_map = model(ctx)[0, 0].cpu().numpy().astype(np.float32)
            pred_pct_map = _to_pct((pred_z_map * safe_scale + mean_map)[None])[0]
            resid_snaps.append((all_pct[t] - pred_pct_map).astype(np.float32))
        else:
            resid_snaps.append(None)

    valid_snaps = [r for r in resid_snaps if r is not None]
    snap_vmax_res = max(
        float(np.percentile(np.abs(np.stack(valid_snaps)), 98)), 1e-6
    ) if valid_snaps else 1e-6

    # ── layout ───────────────────────────────────────────────────────────────
    n_cols = len(snap_frames) * 2
    fig = plt.figure(figsize=(16, 20), constrained_layout=True)
    gs  = fig.add_gridspec(4, n_cols, height_ratios=[3.2, 1.4, 1.0, 0.45])

    # Row 0: spatial snapshots
    for col, (t, offset_s, resid) in enumerate(zip(snap_frames, snap_offsets_s, resid_snaps)):
        ax_gt  = fig.add_subplot(gs[0, col * 2])
        ax_res = fig.add_subplot(gs[0, col * 2 + 1])

        label_t = "t₀" if offset_s == 0 else f"t₀+{offset_s:.1f}s"
        im_gt = ax_gt.imshow(all_pct[t], cmap="RdBu_r", origin="upper",
                             vmin=-snap_vmax_gt, vmax=snap_vmax_gt)
        ax_gt.contour(roi_mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=[1.0])
        ax_gt.set_title(f"GT  {label_t}\n(frame {t})", fontsize=9)
        ax_gt.axis("off")
        fig.colorbar(im_gt, ax=ax_gt, shrink=0.7, label="% CBV", pad=0.02)

        if resid is not None:
            im_res = ax_res.imshow(resid, cmap="RdBu_r", origin="upper",
                                   vmin=-snap_vmax_res, vmax=snap_vmax_res)
            ax_res.contour(roi_mask.astype(float), levels=[0.5], colors=["yellow"], linewidths=[1.0])
            ax_res.set_title(f"GT−ConvLSTM  {label_t}", fontsize=9)
            ax_res.axis("off")
            fig.colorbar(im_res, ax=ax_res, shrink=0.7, label="% CBV", pad=0.02)
        else:
            ax_res.text(0.5, 0.5, "no model", ha="center", va="center",
                        transform=ax_res.transAxes, fontsize=9)
            ax_res.axis("off")

    # Row 1: time series
    t_slice  = slice(t_start, t_end)
    t_rel    = t_rel_all[t_slice]
    sig      = roi_signal_full[t_slice]
    cl_sig   = pred_cl_roi[t_slice]
    rm_sig   = pred_rm_roi[t_slice]

    ax_ts = fig.add_subplot(gs[1, :])

    band_cap   = bl_mean_pct + 3.5 * bl_std_pct
    band_edges = [bl_mean_pct,
                  bl_mean_pct + bl_std_pct,
                  bl_mean_pct + 2 * bl_std_pct,
                  band_cap]
    for i, col in enumerate(_SIGMA_COLORS):
        ax_ts.axhspan(band_edges[i], band_edges[i + 1], color=col, lw=0, zorder=1)

    ax_ts.plot(t_rel, sig, color="#2ca02c", lw=2.5, label="GT (ROI mean)", zorder=5)
    if not np.all(np.isnan(rm_sig)):
        ax_ts.plot(t_rel, rm_sig, color="#9467bd", lw=1.5, ls="--",
                   label="Rolling mean pred", zorder=4)
    if not np.all(np.isnan(cl_sig)):
        ax_ts.plot(t_rel, cl_sig, color="#d62728", lw=1.5, ls="--",
                   label="ConvLSTM pred", zorder=4)
    ax_ts.axhline(bl_mean_pct, color="#888888", lw=1.0, ls=":", label="Baseline mean", zorder=2)
    ax_ts.axhline(0, color="#bbbbbb", lw=0.7, ls="-", zorder=1)
    ax_ts.axvline(0, color="black", lw=1.2, ls="--", label="onset (t₀)", zorder=6)
    ax_ts.axvspan(-5, 5, color="#eeeeee", alpha=0.5, lw=0, zorder=0, label="±5 s window")

    all_vals = [sig[~np.isnan(sig)]]
    for s in (cl_sig, rm_sig):
        v = s[~np.isnan(s)]
        if len(v):
            all_vals.append(v)
    combined = np.concatenate(all_vals)
    y_bot = float(combined.min()) - 0.5 * bl_std_pct
    y_top = max(float(combined.max()), band_cap) + 0.5 * bl_std_pct

    x_right = t_rel[-1]
    for n, edge in zip((1, 2, 3), band_edges[1:]):
        y_label = (band_edges[n - 1] + edge) / 2
        if y_bot < y_label < y_top:
            ax_ts.text(x_right, y_label, f" +{n}σ", va="center", ha="left",
                       fontsize=8, color="#cc3333", clip_on=False)

    ax_ts.set_xlim(t_rel[0], t_rel[-1])
    ax_ts.set_ylim(y_bot, y_top)
    ax_ts.set_ylabel("ROI-mean\n(% CBV)", fontsize=10)
    ax_ts.set_xlabel("Time relative to onset (s)", fontsize=10)
    ax_ts.set_title("ROI-averaged % CBV around baseline→active transition", fontsize=11)
    ax_ts.legend(fontsize=8, loc="upper left", ncol=4)
    ax_ts.tick_params(labelsize=9)
    ax_ts.grid(False)

    # Row 2: residuals
    ax_resid = fig.add_subplot(gs[2, :])

    rm_centred = resid_rm_roi[t_slice].copy()
    cl_centred = resid_cl_roi[t_slice].copy()
    pre_mask   = t_rel < 0

    def _centre(arr: np.ndarray) -> np.ndarray:
        pre_vals = arr[pre_mask & ~np.isnan(arr)]
        return arr - float(np.nanmean(pre_vals)) if len(pre_vals) else arr

    rm_centred = _centre(rm_centred)
    cl_centred = _centre(cl_centred)

    if not np.all(np.isnan(rm_centred)):
        ax_resid.plot(t_rel, rm_centred, color="#9467bd", lw=0.8, alpha=0.4, label="_nolegend_", zorder=2)
        ax_resid.plot(t_rel, _rolling_mean(rm_centred), color="#9467bd", lw=1.6,
                      label="|GT − rolling mean pred| (% CBV)", zorder=3)
    if not np.all(np.isnan(cl_centred)):
        ax_resid.plot(t_rel, cl_centred, color="#d62728", lw=0.8, alpha=0.4, label="_nolegend_", zorder=2)
        ax_resid.plot(t_rel, _rolling_mean(cl_centred), color="#d62728", lw=1.6,
                      label="|GT − ConvLSTM| (% CBV)", zorder=3)

    resid_vals = np.concatenate([
        v[~np.isnan(v)] for v in (rm_centred, cl_centred) if not np.all(np.isnan(v))
    ]) if not (np.all(np.isnan(rm_centred)) and np.all(np.isnan(cl_centred))) else np.array([0.0])
    r_abs = float(np.percentile(np.abs(resid_vals), 99)) * 1.15
    ax_resid.set_ylim(-r_abs, r_abs)

    ax_resid.axhline(0, color="#888888", lw=0.8, ls=":", zorder=1)
    ax_resid.axvspan(-5, 5, color="#eeeeee", alpha=0.5, lw=0, zorder=0)
    ax_resid.axvline(0, color="black", lw=1.2, ls="--", zorder=6)
    ax_resid.set_xlim(t_rel[0], t_rel[-1])
    ax_resid.set_xlabel("Time relative to onset (s)", fontsize=10)
    ax_resid.set_ylabel("Δ|residual| from\npre-onset mean (% CBV)", fontsize=10)
    ax_resid.set_title(
        "Prediction residuals − pre-onset mean  (faint = raw, bold = 10-frame rolling mean)",
        fontsize=10,
    )
    ax_resid.legend(fontsize=8, loc="upper left", ncol=3)
    ax_resid.tick_params(labelsize=9)
    ax_resid.grid(alpha=0.2)

    # Row 3: crossing-time table
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.axis("off")

    def _fmt(v: float | None) -> str:
        return f"{v:.2f} s" if v is not None else "—"

    col_labels = [
        "Threshold",
        "GT % CBV crosses +nσ (s)",
        "|GT − ConvLSTM| crosses +nσ (s)",
        "|GT − rolling mean pred| crosses +nσ (s)",
    ]
    rows_data = [
        [f"+{n}σ", _fmt(crossings_gt[n]), _fmt(crossings_cl[n]), _fmt(crossings_rm[n])]
        for n in (1, 2, 3)
    ]
    tbl = ax_tbl.table(cellText=rows_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)

    fig.suptitle(
        f"Session {session_id} — Dizeux-style transition analysis (% CBV)\n"
        f"onset frame {onset}  |  fps={fps:.2f}  |  ROI={roi_mask.sum()} px  |  "
        f"baseline: mean={bl_mean_pct:.3f}%, σ={bl_std_pct:.3f}%",
        fontsize=11,
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
