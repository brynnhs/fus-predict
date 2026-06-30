"""
rolling_mean.py
----------------
Rolling-mean baseline predictor.

Forecasts the mean of the last ``window`` context frames, irrespective of
horizon. Captures slow drift/baseline level without modeling any temporal
dynamics, and serves as a simple non-trivial baseline above the zero
predictor.
"""

from __future__ import annotations

import numpy as np


class RollingMeanPredictor:
    """
    Baseline predictor that forecasts the mean of the last ``window`` frames.

    The same prediction is returned for every horizon, since the rolling
    mean does not depend on how far ahead the forecast is requested.

    Attributes
    ----------
    name : str
        Human-readable model identifier, ``"rolling_mean"``.
    window : int
        Number of trailing context frames to average over.
    """

    name: str = "rolling_mean"

    def __init__(self, window: int = 10) -> None:
        """
        Initialize the predictor.

        Parameters
        ----------
        window : int
            Number of trailing context frames to average. Default: 10.
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self._params: dict[int, dict] = {}

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
    ) -> None:
        """
        No-op fit. Records that each horizon is supported.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session, each of shape ``(T, H, W)``.
        horizons : list of int
            Prediction horizons this model will later be asked to predict.
        """
        if not train_frames:
            raise ValueError("train_frames must contain at least one session")
        for h in horizons:
            self._params[h] = {}

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict the mean of the last ``window`` context frames.

        Parameters
        ----------
        context : np.ndarray, shape (T, H, W)
            Recent past frames. ``T`` must be >= 1; if ``T < window`` all
            available frames are averaged.
        horizon : int
            Prediction horizon. Must be a key in ``self._params``.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Mean of the last ``min(window, T)`` context frames.
        """
        if horizon not in self._params:
            raise KeyError(f"horizon {horizon} was not fitted")
        window_frames = context[-self.window:]
        return window_frames.mean(axis=0).astype(np.float32)

    def __repr__(self) -> str:
        return f"RollingMeanPredictor(window={self.window})"
