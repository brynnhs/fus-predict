"""
zero.py
-------
Trivial baseline predictor that always forecasts a zero frame.

Useful as a lower-bound sanity check: any model that does not outperform
this one is not learning anything useful about the z-scored fUS signal.
"""

from __future__ import annotations

import numpy as np


class ZeroPredictor:
    """
    Baseline predictor that always outputs an all-zero frame.

    Since input frames are z-scored, the zero frame corresponds to the
    per-pixel training mean and serves as a naive lower-bound baseline.

    Attributes
    ----------
    name : str
        Human-readable model identifier, ``"zero"``.
    """

    name: str = "zero"

    def __init__(self) -> None:
        """Initialize the predictor. No hyperparameters are required."""
        self._params: dict[int, dict] = {}

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
    ) -> None:
        """
        No-op fit. Records the frame shape per horizon for ``predict``.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session, each of shape ``(T, H, W)``.
        horizons : list of int
            Prediction horizons this model will later be asked to predict.
        """
        if not train_frames:
            raise ValueError("train_frames must contain at least one session")
        shape = train_frames[0].shape[1:]
        for h in horizons:
            self._params[h] = {"shape": shape}

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict an all-zero frame.

        Parameters
        ----------
        context : np.ndarray, shape (T, H, W)
            Recent past frames. Only the spatial shape is used.
        horizon : int
            Prediction horizon. Must be a key in ``self._params``.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Array of zeros.
        """
        if horizon not in self._params:
            raise KeyError(f"horizon {horizon} was not fitted")
        h, w = context.shape[1], context.shape[2]
        return np.zeros((h, w), dtype=np.float32)

    def __repr__(self) -> str:
        return "ZeroPredictor()"
