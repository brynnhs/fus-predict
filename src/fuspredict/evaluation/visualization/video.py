"""
video.py
--------
Triplet (GT | prediction | residual) mp4 rendering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_triplet_video(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    session_id: str = "",
    frame_indices: np.ndarray | None = None,
    fps: float = 2.5,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    dpi: int = 120,
) -> None:
    """Render GT | prediction | residual as a side-by-side mp4 video.

    Parameters
    ----------
    gt : np.ndarray
        Ground-truth frames, shape ``(N, H, W)``.
    pred : np.ndarray
        Predicted frames, shape ``(N, H, W)``.
    out_path : Path
        Output ``.mp4`` path; parent directories are created automatically.
    session_id : str
        Label shown in the figure title.
    frame_indices : np.ndarray, optional
        Indices into the first axis of ``gt``/``pred`` to render.
        Defaults to all frames ``np.arange(N)``.
    fps : float
        Output video frame rate (default: 2.5).
    vmin, vmax : float, optional
        Colormap limits in z-score units. If omitted, symmetric 99th-percentile
        of |gt| over the selected frames is used.
    cmap : str
        Matplotlib colormap name (default: ``"RdBu_r"``).
    dpi : int
        Figure DPI for each rendered frame (default: 120).
    """
    import cv2

    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    N = gt.shape[0]

    if frame_indices is None:
        frame_indices = np.arange(N)

    indices = [int(i) for i in frame_indices if 0 <= int(i) < N]
    if not indices:
        return

    if vmin is None or vmax is None:
        abs_max = float(np.percentile(np.abs(gt[indices]), 99))
        if vmin is None:
            vmin = -abs_max
        if vmax is None:
            vmax = abs_max

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True, dpi=dpi)
    title = f"{session_id}  |  GT | Prediction | Residual" if session_id else "GT | Prediction | Residual"
    fig.suptitle(title, fontsize=10)

    ims = []
    dummy = np.zeros_like(gt[0])
    for ax, col_title in zip(axes, ["Ground truth", "Prediction", "Residual (GT − pred)"]):
        ax.set_title(col_title, fontsize=9)
        ax.axis("off")
        ims.append(ax.imshow(dummy, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                             interpolation="nearest"))
    cbar_ax = fig.add_axes([0.02, 0.08, 0.96, 0.03])
    fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal", label="z-score")
    frame_text = fig.text(0.5, 0.97, "", ha="center", va="top", fontsize=8, color="#555555")

    # probe frame dimensions
    ims[0].set_data(dummy)
    frame_text.set_text("probe")
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w_px, h_px = fig.canvas.get_width_height()
    h_px, w_px = buf.reshape(h_px, w_px, 4).shape[:2]

    out_path = Path(out_path).with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_px, h_px))

    for rank, t in enumerate(indices):
        residual = gt[t] - pred[t]
        ims[0].set_data(gt[t])
        ims[1].set_data(pred[t])
        ims[2].set_data(residual)
        frame_text.set_text(f"frame {t}  ({rank + 1}/{len(indices)})")

        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)
        writer.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))

    writer.release()
    plt.close(fig)
