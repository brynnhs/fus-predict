from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

from .io import (
    STAGE_REORIENTED_RESIZED,
    derive_session_id_from_path,
    sanitize_attrs,
)
from ._utils import _validate_frames_thw


def _select_preview_indices(T: int, n_frames: int) -> np.ndarray:
    if T <= 0:
        return np.array([], dtype=np.int64)
    n = max(1, min(int(n_frames), int(T)))
    if n == 1:
        return np.array([0], dtype=np.int64)
    return np.unique(np.linspace(0, T - 1, num=n, dtype=np.int64))


def _compute_preview_limits(
    before: np.ndarray,
    idxs: np.ndarray,
    q_low: float = 1.0,
    q_high: float = 99.0,
) -> tuple[float, float]:
    """Compute shared preview intensity limits from before frames only."""
    _validate_frames_thw(before, ctx="before")
    arr = np.asarray(before)
    vals = arr[idxs].reshape(-1) if idxs.size > 0 else arr.reshape(-1)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(vals, q_low))
    vmax = float(np.percentile(vals, q_high))

    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        vmin, vmax = float(np.min(vals)), float(np.max(vals))

    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        return 0.0, 1.0

    return vmin, vmax


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def rotate_frames(frames: np.ndarray, k: int) -> np.ndarray:
    """Rotate each frame in a (T, H, W) array with np.rot90 around spatial axes."""
    _validate_frames_thw(frames, ctx="rotate_frames")
    arr = np.asarray(frames)
    return np.rot90(arr, k=int(k), axes=(1, 2))


def flip_frames_lr(frames: np.ndarray) -> np.ndarray:
    """Flip each frame in a (T, H, W) array left-right (width axis)."""
    _validate_frames_thw(frames, ctx="flip_frames_lr")
    arr = np.asarray(frames)
    return np.flip(arr, axis=2)


def pad_or_crop_to_square(frames: np.ndarray, target_size: int) -> np.ndarray:
    """
    Center pad or crop a (T, H, W) array to (T, target_size, target_size).
    No interpolation is performed.
    """
    _validate_frames_thw(frames, ctx="pad_or_crop_to_square")
    arr = np.asarray(frames)
    target = int(target_size)
    if target <= 0:
        raise ValueError(f"target_size must be > 0, got {target_size}")

    _, H, W = arr.shape
    out = arr

    if H > target:
        top = (H - target) // 2
        out = out[:, top: top + target, :]
    elif H < target:
        pad_top = (target - H) // 2
        out = np.pad(out, ((0, 0), (pad_top, target - H - pad_top), (0, 0)), mode="constant")

    _, H2, W2 = out.shape
    if W2 > target:
        left = (W2 - target) // 2
        out = out[:, :, left: left + target]
    elif W2 < target:
        pad_left = (target - W2) // 2
        out = np.pad(out, ((0, 0), (0, 0), (pad_left, target - W2 - pad_left)), mode="constant")

    return out


def reorient_and_resize(
    frames: np.ndarray,
    *,
    rotate_k: int,
    flip_lr: bool,
    target_size: int,
) -> np.ndarray:
    """Apply rotate → optional left-right flip → center pad/crop to square."""
    out = rotate_frames(frames, k=rotate_k)
    if flip_lr:
        out = flip_frames_lr(out)
    return pad_or_crop_to_square(out, target_size=target_size)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def save_before_after_preview(
    before: np.ndarray,
    after: np.ndarray,
    out_png_path: str,
    n_frames: int = 6,
) -> None:
    """Save a before/after preview grid with shared intensity scaling from before frames."""
    import matplotlib.pyplot as plt

    _validate_frames_thw(before, ctx="save_before_after_preview(before)")
    _validate_frames_thw(after,  ctx="save_before_after_preview(after)")
    before_arr = np.asarray(before)
    after_arr  = np.asarray(after)

    T    = min(before_arr.shape[0], after_arr.shape[0])
    idxs = _select_preview_indices(T=T, n_frames=n_frames)
    if idxs.size == 0:
        return

    vmin, vmax = _compute_preview_limits(before_arr, idxs)

    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = len(idxs)
    fig, axes = plt.subplots(rows, 2, figsize=(7.5, max(2.5, 2.2 * rows)), squeeze=False)

    for row_i, frame_i in enumerate(idxs.tolist()):
        axes[row_i, 0].imshow(before_arr[frame_i], cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_i, 0].set_title(f"Before t={frame_i}")
        axes[row_i, 0].axis("off")

        axes[row_i, 1].imshow(after_arr[frame_i], cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_i, 1].set_title(f"After t={frame_i}")
        axes[row_i, 1].axis("off")

    fig.suptitle("Reorientation / Resize Preview")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def reorient_baseline_sessions(
    in_nc_paths: list[str],
    out_dir: str | os.PathLike[str],
    *,
    rotate_k: int = -1,
    flip_session_ids: set[str] | None = None,
    target_size: int = 112,
    overwrite: bool = False,
    save_previews: bool = True,
    preview_dir: str | os.PathLike[str] | None = None,
) -> list[str]:
    """
    Run the geometry stage over a list of session .nc files.

    Applies rotate → optional flip → pad/crop to square.
    Saves one .nc file per session to out_dir.
    Skips sessions where the output already exists unless overwrite=True.
    """
    out_root     = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    preview_root = Path(preview_dir) if preview_dir is not None else out_root / "previews"
    flip_ids     = flip_session_ids or set()

    outputs: list[str] = []
    for in_path_str in in_nc_paths:
        in_path = Path(in_path_str)
        da      = xr.open_dataarray(in_path)

        session_id = da.attrs.get("session_id") or derive_session_id_from_path(in_path)
        out_path   = out_root / f"baseline_{session_id}_{STAGE_REORIENTED_RESIZED}.nc"

        if out_path.exists() and not overwrite:
            outputs.append(str(out_path))
            continue

        frames_in = da.values  # (T, H, W)
        flip_lr   = session_id in flip_ids

        # Already reoriented upstream — carry attrs forward without re-applying
        if str(da.attrs.get("stage", "")) == STAGE_REORIENTED_RESIZED:
            da.attrs["session_id"] = session_id
            da.to_netcdf(out_path)
            if save_previews:
                preview_root.mkdir(parents=True, exist_ok=True)
                preview_path = preview_root / f"preview_{session_id}_{STAGE_REORIENTED_RESIZED}.png"
                if overwrite or not preview_path.exists():
                    save_before_after_preview(frames_in, frames_in, str(preview_path))
            outputs.append(str(out_path))
            continue

        frames_out = reorient_and_resize(
            frames_in,
            rotate_k=rotate_k,
            flip_lr=flip_lr,
            target_size=target_size,
        )

        attrs = sanitize_attrs({
            **da.attrs,
            "session_id":  session_id,
            "stage":       STAGE_REORIENTED_RESIZED,
            "rotate_k":    rotate_k,
            "flip_lr":     flip_lr,
            "target_size": target_size,
        })

        T, H, W = frames_out.shape
        da_out = xr.DataArray(
            data=frames_out,
            dims=["time", "x", "y"],
            coords={
                "time": np.arange(T) / float(da.attrs.get("frame_rate", 2.5)),
                "x":    np.arange(H),
                "y":    np.arange(W),
            },
            attrs=attrs,
            name=session_id,
        )
        da_out.to_netcdf(out_path)
        print(f"  Reoriented {in_path.name} → {out_path.name}")

        if save_previews:
            preview_root.mkdir(parents=True, exist_ok=True)
            preview_path = preview_root / f"preview_{session_id}_{STAGE_REORIENTED_RESIZED}.png"
            save_before_after_preview(frames_in, frames_out, str(preview_path))

        outputs.append(str(out_path))

    return outputs