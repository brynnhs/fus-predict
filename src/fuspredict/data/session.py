"""
session.py
----------
Pure data container for a single fUS baseline session.

A Session holds the loaded, standardized frames for one recording session
and is the canonical unit passed through downstream analysis and modeling.
No I/O or computation is performed here; see loading.py for construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Session:
    """
    Container for a single fUS baseline recording session.

    Attributes
    ----------
    id : str
        Session identifier derived from the source filename
        (e.g. ``"20240315"``).
    frames : np.ndarray
        Z-scored Power Doppler frames, shape ``(T, H, W)``, dtype float32.
        T is the number of baseline frames; H=W=112 for standard acquisitions.
    fps : float
        Acquisition frame rate in Hz (typically 2.5).
    vessel_mask : np.ndarray or None
        Boolean vessel mask, shape ``(H, W)``. True where vessel signal was
        detected. None if no mask file was found for this session.
    metadata : dict
        Attributes from the source ``.nc`` file that are worth propagating
        (e.g. ``stage``, ``condition``, ``n_baseline_frames``).
        Does not include ``session_id``, ``frame_rate``, or ``zscored``
        since those are already represented by dedicated fields.
    """

    id: str
    frames: np.ndarray
    fps: float
    vessel_mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties (read-only, no computation)
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of time frames."""
        return self.frames.shape[0]

    @property
    def height(self) -> int:
        """Spatial height (x dimension) in pixels."""
        return self.frames.shape[1]

    @property
    def width(self) -> int:
        """Spatial width (y dimension) in pixels."""
        return self.frames.shape[2]

    @property
    def duration_s(self) -> float:
        """Total recording duration in seconds."""
        return self.n_frames / self.fps

    def __repr__(self) -> str:
        mask_info = "yes" if self.vessel_mask is not None else "no"
        return (
            f"Session(id={self.id!r}, frames={self.frames.shape}, "
            f"fps={self.fps}, mask={mask_info})"
        )
