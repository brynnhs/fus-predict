"""
base.py
-------
Predictor protocol and shared evaluation primitives for fUS frame forecasting.

All models in this package implement the ``Predictor`` protocol. The benchmark
runner depends only on this interface — it never imports concrete model classes
directly.

Forecasting setup
-----------------
The task is **direct multi-step forecasting**: given a context window of past
frames, predict the frame exactly ``horizon`` steps ahead. A separate model is
fitted per horizon (no rollout). The benchmark runner calls
:func:`split_frames` once per session to obtain train/test arrays, then calls
``fit`` with all training arrays and ``predict`` in a sliding-window loop over
the test portion.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Predictor protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Predictor(Protocol):
    """
    Interface that every fUS frame forecasting model must satisfy.

    Implementations may be stateless (e.g. zero predictor) or stateful
    (e.g. ConvLSTM with learned weights). ``fit`` is always called before
    ``predict``, even for models that ignore training data.

    Notes
    -----
    ``fit`` receives *all* training sessions at once so that models that pool
    across sessions (e.g. pixel-wise AR) can do so without needing a separate
    accumulation step.

    ``predict`` returns a single spatial frame — looping over windows and
    aggregating results is the benchmark runner's responsibility.
    """

    name: str
    """Human-readable model identifier used in results tables and plot labels."""

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
    ) -> None:
        """
        Fit the model on training data for one or more prediction horizons.

        For direct forecasting, a separate internal model should be fitted per
        horizon. Models that do not require fitting (e.g. zero predictor) must
        still accept this call and return without error.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session, each of shape ``(T, H, W)``, dtype float32,
            z-scored. ``T`` may differ between sessions.
        horizons : list of int
            Prediction horizons (in frames) that this model will later be asked
            to predict. Guaranteed to be positive integers in ascending order.
        """
        ...

    def predict(
        self,
        context: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        Predict the frame ``horizon`` steps after the end of ``context``.

        Parameters
        ----------
        context : np.ndarray, shape (T, H, W)
            Recent past frames in z-scored space. ``T`` is the context window
            length; the model may use as many or as few as it needs.
        horizon : int
            Number of steps ahead to predict. Must be one of the horizons
            passed to :meth:`fit`.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Predicted frame in z-scored space.
        """
        ...


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split_frames(
    frames: np.ndarray,
    train_frac: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a session's frames into a contiguous train and test portion.

    The split is strictly temporal — no shuffling — to respect the
    autocorrelation structure of fUS time series. The cut point is the same
    for all sessions and all models; this function is the single source of
    truth for the evaluation contract.

    Parameters
    ----------
    frames : np.ndarray, shape (T, H, W)
        Z-scored frames for one session.
    train_frac : float
        Fraction of frames to assign to training. Must be in (0, 1).
        Default: 0.8.

    Returns
    -------
    train : np.ndarray, shape (T_train, H, W)
        First ``floor(T * train_frac)`` frames.
    test : np.ndarray, shape (T_test, H, W)
        Remaining frames.

    Raises
    ------
    ValueError
        If ``train_frac`` is not in (0, 1) or if ``frames`` is not 3-D.

    Examples
    --------
    >>> train, test = split_frames(session.frames)
    >>> model.fit([train], horizons=[1, 5, 10])
    """
    if frames.ndim != 3:
        raise ValueError(
            f"frames must be 3-D (T, H, W), got shape {frames.shape}"
        )
    if not (0.0 < train_frac < 1.0):
        raise ValueError(
            f"train_frac must be in (0, 1), got {train_frac}"
        )

    T = frames.shape[0]
    cut = int(T * train_frac)
    return frames[:cut], frames[cut:]
