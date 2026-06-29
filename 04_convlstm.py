"""
06_convlstm.py
--------------
Full-frame and patch-mean ConvLSTM frame prediction for fUS data.

Trains one model per (session, horizon) independently with no cross-session
weight sharing. Outputs a per-session results CSV comparable to the linear
baseline pipeline (script 04).

Select variant with --model {full,patch}:

  full  — Full-frame ConvLSTM (default). Reads p frames as [B, 1, H, W] per
          timestep; predicts next frame from final hidden state.

  patch — Patch-mean ConvLSTM. Tiles [H, W] into non-overlapping
          patch_size × patch_size patches, reduces each to a scalar mean,
          and runs an independent scalar LSTM per patch. Prediction is
          broadcast back to a piecewise-constant full frame.
          patch_size must evenly divide both H and W
          (for 112×112: 4, 7, 8, 14, 16, 28).

Outputs → derivatives/modeling/convlstm/{out_variant}/
  per_session_results.csv
  aggregate_summary.csv
  {session_id}/h{horizon}/
    model.pt
    training_curve.npz
    eval_predictions.npz
    training_curve.png
    rmse_vs_time.png
    spatial_comparison.png
    spatial_error_map.png
  summary/
    horizon_degradation.png
    skill_by_pixel_type.png
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from fuspredict.preprocessing.io import STAGE_STANDARDIZED
from fuspredict.project import find_repo_root, load_project_config

# ---------------------------------------------------------------------------
# Hyperparameters (loaded from config in main; these are defaults)
# ---------------------------------------------------------------------------
LAG             = 10
HORIZONS        = [1, 5, 10]
EPOCHS          = 50
BATCH_SIZE      = 128
LR              = 3e-4
HIDDEN_CHANNELS = 32
FIG_DPI         = 160
TRAIN_FRAC      = 0.8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# ConvLSTM cell
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.hidden_channels = hidden_channels
        # Forget gate bias = 1.0 (Jozefowicz 2015): gate order i, f, o, g.
        nn.init.constant_(self.gates.bias[hidden_channels: 2 * hidden_channels], 1.0)

    def forward(self, x, h, c):
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, o   = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_next    = f * c + i * torch.tanh(g)
        h_next    = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, spatial: tuple[int, int]):
        h = torch.zeros(batch_size, self.hidden_channels, *spatial,
                        device=next(self.parameters()).device)
        return h, torch.zeros_like(h)


class FullFrameConvLSTM(nn.Module):
    """Single-layer ConvLSTM: reads p frames, predicts the next frame."""

    def __init__(self, hidden_channels: int = HIDDEN_CHANNELS):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels=1, hidden_channels=hidden_channels)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, p, H, W] → [B, 1, H, W]."""
        B, p, H, W = x.shape
        h, c = self.cell.init_hidden(B, (H, W))
        for t in range(p):
            h, c = self.cell(x[:, t: t + 1], h, c)
        return self.head(h)


# ---------------------------------------------------------------------------
# Patch-mean ConvLSTM
# ---------------------------------------------------------------------------

def _tile_patches(H: int, W: int, patch_size: int) -> list[tuple[slice, slice]]:
    patches = []
    for r0 in range(0, H, patch_size):
        for c0 in range(0, W, patch_size):
            patches.append((
                slice(r0, min(r0 + patch_size, H)),
                slice(c0, min(c0 + patch_size, W)),
            ))
    return patches


class PatchConvLSTM(nn.Module):
    """Independent scalar LSTM per non-overlapping spatial patch (1×1 kernel)."""

    def __init__(self, hidden_channels: int = HIDDEN_CHANNELS):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels=1, hidden_channels=hidden_channels, kernel_size=1)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, p, 1, 1] → [B, 1, 1, 1]."""
        B, p, H, W = x.shape
        h, c = self.cell.init_hidden(B, (H, W))
        for t in range(p):
            h, c = self.cell(x[:, t: t + 1], h, c)
        return self.head(h)


def _extract_patch_means(
    x: torch.Tensor,
    patch_info: list[tuple[slice, slice]],
) -> torch.Tensor:
    """[N, T, H, W] → [N*P, T, 1, 1] patch-mean inputs."""
    N, T, H, W = x.shape
    pieces  = [x[:, :, rs, cs].mean(dim=(-2, -1), keepdim=True) for rs, cs in patch_info]
    stacked = torch.stack(pieces, dim=1)   # [N, P, T, 1, 1]
    return stacked.reshape(N * len(patch_info), T, 1, 1)


def _reconstruct_from_patch_means(
    patch_means: torch.Tensor,
    patch_info: list[tuple[slice, slice]],
    N: int, H: int, W: int,
) -> torch.Tensor:
    """[N*P, 1, 1, 1] → [N, 1, H, W] piecewise-constant frame."""
    P    = len(patch_info)
    preds = patch_means.reshape(N, P, 1, 1, 1)
    out  = torch.zeros(N, 1, H, W, dtype=patch_means.dtype, device=patch_means.device)
    for pi, (rs, cs) in enumerate(patch_info):
        ph = rs.stop - rs.start
        pw = cs.stop - cs.start
        out[:, :, rs, cs] = preds[:, pi].expand(N, 1, ph, pw)
    return out


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _sliding_windows(
    frames: np.ndarray,
    lag: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N, lag, H, W) inputs and (N, 1, H, W) targets."""
    T = frames.shape[0]
    n = T - lag - horizon + 1
    if n <= 0:
        return (
            np.empty((0, lag, *frames.shape[1:]), dtype=np.float32),
            np.empty((0, 1,   *frames.shape[1:]), dtype=np.float32),
        )
    xs = np.stack([frames[i: i + lag]                          for i in range(n)])
    ys = np.stack([frames[i + lag + horizon - 1: i + lag + horizon] for i in range(n)])
    return xs, ys


def _load_vessel_mask(tissue_dir: Path, session_id: str) -> np.ndarray | None:
    """Return bool (H, W) vessel mask from .nc tissue mask, or None."""
    mask_path = tissue_dir / f"tissue_mask_{session_id}.nc"
    if not mask_path.exists():
        return None
    ds = xr.open_dataset(mask_path)
    if "vessel_mask" not in ds:
        return None
    return ds["vessel_mask"].values.astype(bool)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    vessel_weight_map: torch.Tensor | None = None,
) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        if vessel_weight_map is not None:
            loss = (F.mse_loss(model(x), y, reduction="none") * vessel_weight_map).mean()
        else:
            loss = F.mse_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gt, pred, persistence) each (N, H, W)."""
    model.eval()
    gts, preds, perss = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds.append(model(x)[:, 0].cpu().numpy())
            gts.append(y[:, 0].cpu().numpy())
            perss.append(x[:, -1].cpu().numpy())
    return np.concatenate(gts), np.concatenate(preds), np.concatenate(perss)


def _collapse_delta(
    model: nn.Module,
    eval_inputs: torch.Tensor,
    n_samples: int = 64,
) -> float:
    """Mean |f(x_real) − f(0)|: near zero means model ignores recurrent history."""
    model.eval()
    idx    = torch.randperm(eval_inputs.shape[0])[:n_samples]
    x_real = eval_inputs[idx].to(DEVICE)
    with torch.no_grad():
        diff = (model(x_real) - model(torch.zeros_like(x_real))).abs()
    return float(diff.mean().item())


def _rmse_masked(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None,
) -> float:
    sq = (pred[:, mask] - gt[:, mask]) ** 2 if mask is not None else (pred - gt) ** 2
    return float(math.sqrt(sq.mean()))


def _skill(rmse_model: float, rmse_zero: float) -> float:
    return float(1.0 - rmse_model / rmse_zero) if rmse_zero > 0 else float("nan")


# ---------------------------------------------------------------------------
# Per-session per-horizon training
# ---------------------------------------------------------------------------

def _train_session_horizon(
    frames_train: np.ndarray,
    frames_test: np.ndarray,
    horizon: int,
    out_dir: Path,
    vessel_weight_map: torch.Tensor | None = None,
    model_variant: str = "full",
    patch_size: int = 14,
) -> dict:
    """Train one model and return metric dict. Saves checkpoint to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr = _sliding_windows(frames_train, LAG, horizon)
    x_te, y_te = _sliding_windows(frames_test,  LAG, horizon)

    if x_tr.shape[0] == 0 or x_te.shape[0] == 0:
        raise ValueError(
            f"No windows for horizon={horizon}: "
            f"train_T={frames_train.shape[0]}, test_T={frames_test.shape[0]}"
        )

    x_tr_t       = torch.from_numpy(x_tr)
    y_tr_t       = torch.from_numpy(y_tr)
    eval_inputs  = torch.from_numpy(x_te)
    eval_targets = torch.from_numpy(y_te)

    if model_variant == "patch":
        _, _, H, W = x_tr_t.shape
        N_te = eval_inputs.shape[0]
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"patch_size={patch_size} does not evenly divide ({H},{W}). "
                f"For 112×112 use: 4, 7, 8, 14, 16, 28."
            )
        patch_info = _tile_patches(H, W, patch_size)
        P          = len(patch_info)
        print(f"[patch] {P} patches of {patch_size}×{patch_size}", end="  ", flush=True)

        x_tr_pm = _extract_patch_means(x_tr_t,     patch_info)
        y_tr_pm = _extract_patch_means(y_tr_t,     patch_info)
        x_te_pm = _extract_patch_means(eval_inputs, patch_info)
        y_te_pm = _extract_patch_means(eval_targets, patch_info)

        train_loader    = DataLoader(TensorDataset(x_tr_pm, y_tr_pm), batch_size=BATCH_SIZE * P, shuffle=True)
        eval_loader_pm  = DataLoader(TensorDataset(x_te_pm, y_te_pm), batch_size=BATCH_SIZE * P)

        model     = PatchConvLSTM(HIDDEN_CHANNELS).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train_losses: list[float] = []
        for _ in range(1, EPOCHS + 1):
            train_losses.append(_train_epoch(model, train_loader, optimizer))

        torch.save(model.state_dict(), out_dir / "model.pt")
        np.savez(out_dir / "training_curve.npz", train_loss=np.array(train_losses))

        model.eval()
        pred_means_list, pers_means_list = [], []
        with torch.no_grad():
            for xb, _ in eval_loader_pm:
                pred_means_list.append(model(xb.to(DEVICE)).cpu())
                pers_means_list.append(xb[:, -1:])
        pred_means = torch.cat(pred_means_list)
        pers_means = torch.cat(pers_means_list)

        pred_full = _reconstruct_from_patch_means(pred_means, patch_info, N_te, H, W)
        pers_full = _reconstruct_from_patch_means(pers_means, patch_info, N_te, H, W)
        gt        = eval_targets[:, 0].numpy()
        pred      = pred_full[:, 0].numpy()
        pers      = pers_full[:, 0].numpy()
        np.savez(out_dir / "eval_predictions.npz", gt=gt, pred=pred, persistence=pers)
        collapse  = _collapse_delta(model, x_te_pm)

    else:  # full
        train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
        eval_loader  = DataLoader(TensorDataset(eval_inputs, eval_targets), batch_size=BATCH_SIZE)

        model     = FullFrameConvLSTM(HIDDEN_CHANNELS).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train_losses = []
        for _ in range(1, EPOCHS + 1):
            train_losses.append(_train_epoch(model, train_loader, optimizer, vessel_weight_map))

        torch.save(model.state_dict(), out_dir / "model.pt")
        np.savez(out_dir / "training_curve.npz", train_loss=np.array(train_losses))

        gt, pred, pers = _collect_predictions(model, eval_loader)
        np.savez(out_dir / "eval_predictions.npz", gt=gt, pred=pred, persistence=pers)
        collapse = _collapse_delta(model, eval_inputs)

    return {
        "gt": gt, "pred": pred, "persistence": pers,
        "train_losses": train_losses,
        "mean_collapse_delta": collapse,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    persistence: np.ndarray,
    vessel_mask: np.ndarray | None,
) -> dict:
    rmse_full  = _rmse_masked(pred, gt, None)
    rmse_zero  = float(math.sqrt((gt ** 2).mean()))
    metrics    = {"rmse_full": rmse_full, "skill_full": _skill(rmse_full, rmse_zero)}

    if vessel_mask is not None and vessel_mask.shape == gt.shape[1:]:
        nv_mask  = ~vessel_mask
        rmse_v   = _rmse_masked(pred, gt, vessel_mask)
        rmse_nv  = _rmse_masked(pred, gt, nv_mask)
        zero_v   = float(math.sqrt((gt[:, vessel_mask]  ** 2).mean()))
        zero_nv  = float(math.sqrt((gt[:, nv_mask] ** 2).mean()))
        metrics.update({
            "rmse_vessel":     rmse_v,   "skill_vessel":    _skill(rmse_v,  zero_v),
            "rmse_nonvessel":  rmse_nv,  "skill_nonvessel": _skill(rmse_nv, zero_nv),
        })
    else:
        metrics.update({
            "rmse_vessel": float("nan"), "skill_vessel":    float("nan"),
            "rmse_nonvessel": float("nan"), "skill_nonvessel": float("nan"),
        })
    return metrics


# ---------------------------------------------------------------------------
# Per-session diagnostic plots
# ---------------------------------------------------------------------------

_SMOOTH_WIN  = 5
_MODEL_COLOR = "#ff7f0e"
_PERS_COLOR  = "#888888"
_ZERO_COLOR  = "#aaaaaa"


def _rolling(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rolling(_SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy()


def _savefig(fig: plt.Figure, path: Path) -> None:
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_training_curve(train_losses: list[float], out_dir: Path, horizon: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(train_losses, color=_MODEL_COLOR, lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train MSE")
    ax.set_title(f"Training loss — ConvLSTM h={horizon}")
    ax.grid(alpha=0.3)
    _savefig(fig, out_dir / "training_curve")


def _plot_rmse_vs_time(
    gt: np.ndarray, pred: np.ndarray, persistence: np.ndarray,
    out_dir: Path, horizon: int,
) -> None:
    t = np.arange(len(gt))
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(t, _rolling(np.sqrt(np.mean((pred - gt) ** 2,        axis=(1, 2)))),
            color=_MODEL_COLOR, lw=1.4, label="ConvLSTM")
    ax.plot(t, _rolling(np.sqrt(np.mean((persistence - gt) ** 2, axis=(1, 2)))),
            color=_PERS_COLOR,  lw=1.0, ls="--", label="Persistence")
    ax.plot(t, _rolling(np.sqrt(np.mean(gt ** 2,                 axis=(1, 2)))),
            color=_ZERO_COLOR,  lw=1.0, ls=":",  label="Zero")
    ax.set_xlabel("Target frame (test set)")
    ax.set_ylabel("RMSE (z-score)")
    ax.set_title(f"RMSE vs time — ConvLSTM h={horizon} ({_SMOOTH_WIN}-frame smooth)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _savefig(fig, out_dir / "rmse_vs_time")


def _plot_spatial_comparison(
    gt: np.ndarray, pred: np.ndarray, out_dir: Path, horizon: int,
) -> None:
    gt_mean  = gt.mean(axis=0)
    pr_mean  = pred.mean(axis=0)
    residual = pr_mean - gt_mean
    rmse     = float(math.sqrt(np.mean((pred - gt) ** 2)))
    vabs     = float(np.percentile(np.abs(gt_mean), 98))
    res_lim  = float(np.percentile(np.abs(residual), 98))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
    axes[0].imshow(gt_mean,  cmap="RdBu_r", vmin=-vabs,    vmax=vabs)
    axes[0].set_title("Ground truth (mean)")
    axes[1].imshow(pr_mean,  cmap="RdBu_r", vmin=-vabs,    vmax=vabs)
    axes[1].set_title("ConvLSTM (mean)")
    im = axes[2].imshow(residual, cmap="RdBu_r", vmin=-res_lim, vmax=res_lim)
    axes[2].set_title("Residual: pred − GT")
    axes[2].text(0.97, 0.03, f"RMSE={rmse:.4f}", transform=axes[2].transAxes,
                 ha="right", va="bottom", fontsize=9, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.7))
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], shrink=0.75, label="z-score")
    fig.suptitle(f"Spatial comparison — ConvLSTM h={horizon}", fontsize=10)
    _savefig(fig, out_dir / "spatial_comparison")


def _plot_spatial_error_map(
    gt: np.ndarray, pred: np.ndarray,
    vessel_mask: np.ndarray | None,
    out_dir: Path, horizon: int,
) -> None:
    diff = np.sqrt(np.mean((pred - gt) ** 2, axis=0)) - np.sqrt(np.mean(gt ** 2, axis=0))
    lim  = float(np.percentile(np.abs(diff), 98))
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(diff, cmap="bwr", vmin=-lim, vmax=lim)
    if vessel_mask is not None:
        ax.contour(vessel_mask.astype(float), levels=[0.5], colors="black", linewidths=0.8, alpha=0.7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("RMSE(model) − RMSE(zero) [z-score]\nblue = model better")
    ax.set_title(f"Spatial error map — ConvLSTM h={horizon}\n(black contour = vessel mask)", fontsize=10)
    ax.axis("off")
    _savefig(fig, out_dir / "spatial_error_map")


# ---------------------------------------------------------------------------
# Summary plots across horizons
# ---------------------------------------------------------------------------

def _make_summary_plots(results_df: pd.DataFrame, out_dir: Path) -> None:
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    grp = (
        results_df[results_df["rmse_full"].notna()]
        .groupby("horizon")[["rmse_full", "skill_full"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    grp.columns = ["horizon", "rmse_mean", "rmse_std", "skill_mean", "skill_std"]
    grp = grp.sort_values("horizon")
    if grp.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.errorbar(grp["horizon"], grp["rmse_mean"], yerr=grp["rmse_std"],
                fmt="o-", color=_MODEL_COLOR, lw=1.8, capsize=4, label="ConvLSTM")
    ax.set_xlabel("Prediction horizon (frames)")
    ax.set_ylabel("Mean RMSE (z-score) ± std across sessions")
    ax.set_title("Horizon degradation — ConvLSTM", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, summary_dir / "horizon_degradation")

    if results_df["skill_vessel"].notna().any():
        skill_grp = (
            results_df.groupby("horizon")[["skill_full", "skill_vessel", "skill_nonvessel"]]
            .mean().reset_index().sort_values("horizon")
        )
        x = np.arange(len(skill_grp))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(7, len(skill_grp) * 2.5 + 2), 5), constrained_layout=True)
        ax.bar(x - w, skill_grp["skill_full"],     w, color=_MODEL_COLOR, alpha=0.8, label="Full frame")
        ax.bar(x,     skill_grp["skill_vessel"],    w, color="#d62728",    alpha=0.8, label="Vessel pixels")
        ax.bar(x + w, skill_grp["skill_nonvessel"], w, color="#1f77b4",    alpha=0.8, label="Non-vessel pixels")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in skill_grp["horizon"]])
        ax.set_ylabel("Skill vs zero (mean across sessions)")
        ax.set_title("Skill by pixel type — ConvLSTM", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        _savefig(fig, summary_dir / "skill_by_pixel_type")

    print(f"Summary plots → {summary_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ConvLSTM evaluation for fUS frame prediction")
    parser.add_argument("--model",        choices=["full", "patch"], default="full")
    parser.add_argument("--patch-size",   type=int,   default=14)
    parser.add_argument("--n-sessions",   type=int,   default=None)
    parser.add_argument("--overwrite",    action="store_true")
    parser.add_argument("--no-plots",     action="store_true", help="Skip per-session diagnostic plots")
    parser.add_argument("--vessel-weight", type=float, default=1.0,
                        help="Loss weight multiplier for vessel pixels (full only)")
    args = parser.parse_args()

    repo_root = find_repo_root()
    config    = load_project_config(repo_root)
    mod_cfg   = config.get("modeling", {})
    ar_cfg    = config.get("ar_analysis", {})

    global LAG, HORIZONS, EPOCHS, BATCH_SIZE, LR, HIDDEN_CHANNELS, TRAIN_FRAC
    LAG             = int(mod_cfg.get("n_lags",    LAG))
    HORIZONS        = [int(h) for h in mod_cfg.get("horizons", HORIZONS)]
    EPOCHS          = int(mod_cfg.get("convlstm", {}).get("n_epochs",        EPOCHS))
    BATCH_SIZE      = int(mod_cfg.get("convlstm", {}).get("batch_size",      BATCH_SIZE))
    LR              = float(mod_cfg.get("convlstm", {}).get("learning_rate", LR))
    HIDDEN_CHANNELS = int(mod_cfg.get("convlstm", {}).get("hidden_channels", HIDDEN_CHANNELS))
    TRAIN_FRAC      = float(mod_cfg.get("train_frac", TRAIN_FRAC))

    EXCLUDE_SIDS: list[str] = [str(s) for s in ar_cfg.get("within_session_exclude", [])]

    if args.model == "patch":
        out_variant = f"patch_p{args.patch_size}"
    elif args.vessel_weight != 1.0:
        out_variant = f"vessel_weight_{args.vessel_weight:.1f}"
    else:
        out_variant = "unweighted"

    OUT = repo_root / "derivatives" / "modeling" / "convlstm" / out_variant
    OUT.mkdir(parents=True, exist_ok=True)
    results_csv = OUT / "per_session_results.csv"

    # Resume from existing CSV if present
    if not args.overwrite and results_csv.exists():
        existing_df   = pd.read_csv(results_csv)
        done_sessions = set(existing_df["session_id"].unique())
        print(f"Resuming: {len(done_sessions)} session(s) already cached")
    else:
        existing_df   = pd.DataFrame()
        done_sessions = set()

    # Discover session files
    preproc_root     = repo_root / config["paths"]["preprocessing"]
    standardized_dir = preproc_root / "secundo" / "baseline_only_standardized"
    tissue_dir       = preproc_root / "secundo" / "tissue_masks"

    nc_paths = sorted(
        p for p in standardized_dir.glob(f"baseline_*_unfiltered_{STAGE_STANDARDIZED}.nc")
        if not any(ex in p.stem for ex in EXCLUDE_SIDS)
    )
    if args.n_sessions is not None:
        nc_paths = nc_paths[:args.n_sessions]
    assert nc_paths, f"No .nc sessions found in {standardized_dir}"

    print(f"Device: {DEVICE}  |  Sessions: {len(nc_paths)}  |  "
          f"Horizons: {HORIZONS}  |  Lag: {LAG}  |  Epochs: {EPOCHS}")

    all_rows: list[dict] = list(existing_df.to_dict("records")) if not existing_df.empty else []

    for si, path in enumerate(nc_paths):
        ds         = xr.open_dataset(path)
        session_id = str(ds.attrs.get("session_id", path.stem))
        da         = ds["frames"]
        T          = da.sizes["time"]
        n_train    = int(T * TRAIN_FRAC)
        frames_train = da.values[:n_train].astype(np.float32)
        frames_test  = da.values[n_train:].astype(np.float32)
        ds.close()

        if session_id in done_sessions:
            print(f"[{si+1}/{len(nc_paths)}] {session_id} — skipped (cached)")
            continue

        print(f"\n[{si+1}/{len(nc_paths)}] {session_id}")

        vessel_mask = _load_vessel_mask(tissue_dir, session_id)
        if vessel_mask is not None:
            print(f"  vessel mask: {vessel_mask.sum()} vessel pixels")
        else:
            print("  no vessel mask — vessel metrics will be NaN")

        vessel_weight_map: torch.Tensor | None = None
        if args.vessel_weight != 1.0 and vessel_mask is not None:
            wmap = np.ones(vessel_mask.shape, dtype=np.float32)
            wmap[vessel_mask] = args.vessel_weight
            vessel_weight_map = torch.from_numpy(wmap)[None, None].to(DEVICE)
        elif args.vessel_weight != 1.0:
            print("  WARNING: --vessel-weight ignored (no vessel mask found)")

        for horizon in HORIZONS:
            print(f"  horizon={horizon} ...", end=" ", flush=True)
            ckpt_dir = OUT / session_id / f"h{horizon}"

            try:
                result = _train_session_horizon(
                    frames_train, frames_test, horizon, ckpt_dir,
                    vessel_weight_map, args.model, args.patch_size)
            except Exception as exc:
                print(f"FAILED: {exc}")
                continue

            metrics = _compute_metrics(
                result["gt"], result["pred"], result["persistence"], vessel_mask)

            all_rows.append({
                "session_id":          session_id,
                "horizon":             horizon,
                "rmse_full":           metrics["rmse_full"],
                "skill_full":          metrics["skill_full"],
                "rmse_vessel":         metrics["rmse_vessel"],
                "skill_vessel":        metrics["skill_vessel"],
                "rmse_nonvessel":      metrics["rmse_nonvessel"],
                "skill_nonvessel":     metrics["skill_nonvessel"],
                "mean_collapse_delta": result["mean_collapse_delta"],
            })
            print(f"rmse={metrics['rmse_full']:.4f}  skill={metrics['skill_full']:+.3f}  "
                  f"collapse_delta={result['mean_collapse_delta']:.4f}")

            if not args.no_plots:
                _plot_training_curve(result["train_losses"], ckpt_dir, horizon)
                _plot_rmse_vs_time(result["gt"], result["pred"], result["persistence"],
                                   ckpt_dir, horizon)
                _plot_spatial_comparison(result["gt"], result["pred"], ckpt_dir, horizon)
                _plot_spatial_error_map(result["gt"], result["pred"], vessel_mask,
                                        ckpt_dir, horizon)

        # Flush CSV after every session
        pd.DataFrame(all_rows).to_csv(results_csv, index=False)
        print(f"  CSV → {results_csv}")

    results_df = pd.DataFrame(all_rows)
    if results_df.empty:
        print("No results to summarize.")
        return

    results_df.to_csv(results_csv, index=False)
    print(f"\nPer-session results → {results_csv}")

    agg = (
        results_df.groupby("horizon")
        .agg(
            n_sessions            = ("session_id",          "nunique"),
            rmse_mean             = ("rmse_full",            "mean"),
            rmse_std              = ("rmse_full",            "std"),
            skill_mean            = ("skill_full",           "mean"),
            skill_vessel_mean     = ("skill_vessel",         "mean"),
            skill_nonvessel_mean  = ("skill_nonvessel",      "mean"),
            collapse_delta_mean   = ("mean_collapse_delta",  "mean"),
        )
        .reset_index()
    )
    agg.to_csv(OUT / "aggregate_summary.csv", index=False)
    print(f"Aggregate summary → {OUT / 'aggregate_summary.csv'}")
    print(agg.to_string(index=False))

    _make_summary_plots(results_df, OUT)
    print(f"\nAll outputs → {OUT}")


if __name__ == "__main__":
    main()