"""
pixel_ar.py
-----------
Per-pixel, ridge-regularized direct-horizon autoregressive predictor.

Each pixel is modeled independently: its value at time ``t + horizon`` is
predicted as a linear function of its own previous ``p`` values (lags),
fitted by ridge regression in closed form via the normal equations.
Fitting is fully vectorized over pixels using ``np.einsum``.
"""

from __future__ import annotations

import numpy as np


class PixelAR:
    """
    Per-pixel direct-horizon autoregressive predictor with ridge regularization.

    For each pixel ``(i, j)`` and each horizon ``h``, fits weights
    ``w in R^p`` and bias ``b`` such that::

        frame[t + h, i, j] ~= b[i, j] + sum_k w[i, j, k] * frame[t - k, i, j]

    Weights are estimated independently per pixel via the closed-form ridge
    solution, batched across all pixels with ``np.einsum`` for efficiency.

    Attributes
    ----------
    name : str
        Human-readable model identifier, ``"pixel_ar"``.
    p : int
        Number of autoregressive lags used as predictors.
    ridge_lambda : float
        L2 regularization strength applied to the AR weights (not the bias).
    """

    name: str = "pixel_ar"

    def __init__(self, p: int = 10, ridge_lambda: float = 0.01) -> None:
        """
        Initialize the predictor.

        Parameters
        ----------
        p : int
            Number of autoregressive lags. Default: 10.
        ridge_lambda : float
            L2 regularization strength. Default: 0.01.
        """
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        if ridge_lambda < 0:
            raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")
        self.p = p
        self.ridge_lambda = ridge_lambda
        self._params: dict[int, dict] = {}

    def _build_design(
        self,
        frames: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build per-pixel lag design matrix and targets for one session.

        Parameters
        ----------
        frames : np.ndarray, shape (T, H, W)
            Session frames.
        horizon : int
            Prediction horizon.

        Returns
        -------
        X : np.ndarray, shape (N, p, H, W)
            Lag features, ``X[n, k] = frames[n + p - 1 - k]``.
        Y : np.ndarray, shape (N, H, W)
            Targets, ``Y[n] = frames[n + p - 1 + horizon]``.
        """
        T = frames.shape[0]
        n_samples = T - self.p - horizon + 1
        if n_samples <= 0:
            empty_x = np.empty((0, self.p, *frames.shape[1:]), dtype=np.float32)
            empty_y = np.empty((0, *frames.shape[1:]), dtype=np.float32)
            return empty_x, empty_y

        X = np.stack(
            [
                frames[self.p - 1 - k : self.p - 1 - k + n_samples]
                for k in range(self.p)
            ],
            axis=1,
        )
        Y = frames[self.p - 1 + horizon : self.p - 1 + horizon + n_samples]
        return X.astype(np.float32), Y.astype(np.float32)

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
    ) -> None:
        """
        Fit per-pixel ridge AR weights for each horizon via normal equations.

        Accumulates ``X^T X`` and ``X^T Y`` (per pixel) across all sliding
        windows from all training sessions, then solves the regularized
        normal equations once per horizon.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session, each of shape ``(T, H, W)``.
        horizons : list of int
            Prediction horizons to fit.
        """
        if not train_frames:
            raise ValueError("train_frames must contain at least one session")
        h_img, w_img = train_frames[0].shape[1:]
        p = self.p
        dim = p + 1  # +1 for bias

        for horizon in horizons:
            # XtX: (H, W, dim, dim), XtY: (H, W, dim)
            XtX = np.zeros((h_img, w_img, dim, dim), dtype=np.float64)
            XtY = np.zeros((h_img, w_img, dim), dtype=np.float64)

            for frames in train_frames:
                X, Y = self._build_design(frames, horizon)
                if X.shape[0] == 0:
                    continue
                # X: (N, p, H, W) -> (N, H, W, p) -> append bias column of ones
                X = np.moveaxis(X, 1, -1)  # (N, H, W, p)
                ones = np.ones((*X.shape[:-1], 1), dtype=np.float32)
                Xb = np.concatenate([X, ones], axis=-1)  # (N, H, W, dim)

                XtX += np.einsum("nhwa,nhwb->hwab", Xb, Xb, optimize=True)
                XtY += np.einsum("nhwa,nhw->hwa", Xb, Y, optimize=True)

            reg = self.ridge_lambda * np.eye(dim, dtype=np.float64)
            reg[-1, -1] = 0.0  # do not regularize bias
            XtX_reg = XtX + reg  # broadcasts to (H, W, dim, dim)

            weights = np.linalg.solve(XtX_reg, XtY[..., None])[..., 0]  # (H, W, dim)
            A = weights[..., :p].astype(np.float32)  # (H, W, p)
            b = weights[..., p].astype(np.float32)  # (H, W)

            self._params[horizon] = {"A": A, "b": b}

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict the frame ``horizon`` steps ahead using per-pixel AR weights.

        Parameters
        ----------
        context : np.ndarray, shape (T, H, W)
            Recent past frames, ``T >= p``. Only the last ``p`` frames are
            used.
        horizon : int
            Prediction horizon. Must be a key in ``self._params``.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Predicted frame.
        """
        if horizon not in self._params:
            raise KeyError(f"horizon {horizon} was not fitted")
        params = self._params[horizon]
        A = params["A"]  # (H, W, p)
        b = params["b"]  # (H, W)

        lags = context[-self.p:]  # (p, H, W), lags[0] is oldest of the window
        # lags[i] should correspond to A[:, :, i] where i=0 is most recent lag
        lags_recent_first = lags[::-1]  # (p, H, W), index 0 = most recent

        pred = b.copy()
        for i in range(self.p):
            pred = pred + A[:, :, i] * lags_recent_first[i]
        return pred.astype(np.float32)

    def __repr__(self) -> str:
        return f"PixelAR(p={self.p}, ridge_lambda={self.ridge_lambda})"
