from __future__ import annotations

import numpy as np


def _validate_frames_thw(frames: np.ndarray, ctx: str) -> tuple[int, int, int]:
    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError(f"{ctx}: expected shape (T, H, W), got {arr.shape}")
    t, h, w = arr.shape
    if t == 0:
        raise ValueError(f"{ctx}: expected at least 1 frame, got shape {arr.shape}")
    return int(t), int(h), int(w)
