"""
frozen_basis.py
----------------
Full-frame frozen-basis (PCA/ICA) predictors.

``FrozenBasisAR``
    Fits a single PCA or ICA basis once on all frames passed to ``fit``
    (intended usage: the full baseline period), then fits ridge-regularized
    AR weights in the resulting latent space. At predict time — intended
    usage: task-period frames — new context frames are projected onto the
    *frozen* basis (``transform``, never refit) before the AR step.

``FrozenBasisRollingMean``
    Same frozen-basis idea, but the prediction is the mean of the last
    ``window`` latent vectors, decoded back to pixel space, instead of an
    AR model.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA, FastICA

from fuspredict.models.pca_ar._ridge import _fit_ridge_ar, _solve_ridge_ar


# ---------------------------------------------------------------------------
# FrozenBasisAR
# ---------------------------------------------------------------------------

class FrozenBasisAR:
    """
    AR predictor on a frozen full-frame PCA or ICA basis.

    The basis is fitted once, on all frames passed to ``fit`` (intended
    usage: the full baseline period), and is never refit afterwards. Ridge AR
    weights are then fitted in that frozen latent space, independently per
    horizon. At predict time — intended usage: task-period context frames —
    new frames are projected onto the frozen basis (``transform``, not
    ``fit_transform``) before the AR step, so the basis never sees task-period
    data during fitting.

    Attributes
    ----------
    name : str
        Human-readable model identifier, ``"frozen_pca_ar"`` or
        ``"frozen_ica_ar"`` depending on ``method``.
    method : str
        ``"pca"`` or ``"ica"``.
    n_components : int
        Number of components in the frozen basis.
    ar_lag : int
        Number of autoregressive lags in latent space.
    ridge_lambda : float
        L2 regularization strength.
    seed : int
        Random seed for the basis solver.
    """

    name: str = "frozen_basis_ar"  # overwritten per-instance in __init__ with method suffix

    def __init__(
        self,
        method: str = "pca",
        n_components: int = 10,
        ar_lag: int = 10,
        ridge_lambda: float = 0.01,
        ica_algorithm: dict | None = None,
        seed: int = 0,
    ) -> None:
        """
        Initialize the predictor.

        Parameters
        ----------
        method : str
            ``"pca"`` or ``"ica"``. Default: ``"pca"``.
        n_components : int
            Number of basis components to retain. Default: 10.
        ar_lag : int
            Number of autoregressive lags. Default: 10.
        ridge_lambda : float
            L2 regularization strength. Default: 0.01.
        ica_algorithm : dict, optional
            Extra kwargs forwarded to ``sklearn.decomposition.FastICA``
            (e.g. ``{"algorithm": "parallel", "fun": "logcosh"}``). Ignored
            for ``method="pca"``.
        seed : int
            Random seed for deterministic basis fitting. Default: 0.
        """
        if method not in ("pca", "ica"):
            raise ValueError(f"method must be 'pca' or 'ica', got {method!r}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if ar_lag < 1:
            raise ValueError(f"ar_lag must be >= 1, got {ar_lag}")
        if ridge_lambda < 0:
            raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")
        self.method = method
        self.name = f"frozen_{method}_ar"
        self.n_components = n_components
        self.ar_lag = ar_lag
        self.ridge_lambda = ridge_lambda
        self.ica_algorithm = ica_algorithm or {}
        self.seed = seed
        self._basis: PCA | FastICA | None = None
        self._frame_shape: tuple[int, int] | None = None
        self._params: dict[int, dict] = {}

    def _project(self, flat: np.ndarray) -> np.ndarray:
        """Project already-flattened, float64 frames onto the frozen basis."""
        return self._basis.transform(flat)

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
    ) -> None:
        """
        Fit the frozen basis on all given frames (baseline period), then fit
        ridge AR weights per horizon on the same projected frames.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session, each of shape ``(T, H, W)``. Intended to
            cover the full baseline period — the basis is fitted on all of
            it and frozen for later use on task-period frames.
        horizons : list of int
            Prediction horizons to fit.
        """
        if not train_frames:
            raise ValueError("train_frames must contain at least one session")
        self._frame_shape = train_frames[0].shape[1:]

        fit_flat = np.concatenate(
            [f.reshape(f.shape[0], -1) for f in train_frames], axis=0
        ).astype(np.float64)

        n_comp = min(self.n_components, fit_flat.shape[0] - 1, fit_flat.shape[1])
        if self.method == "pca":
            basis = PCA(n_components=n_comp, random_state=self.seed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in divide", RuntimeWarning)
                basis.fit(fit_flat)
        else:
            kwargs = dict(whiten="unit-variance", max_iter=500, random_state=self.seed)
            kwargs.update(self.ica_algorithm)
            basis = FastICA(n_components=n_comp, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                basis.fit(fit_flat)
        self._basis = basis

        latents = [
            self._project(f.reshape(f.shape[0], -1).astype(np.float64))
            for f in train_frames
        ]

        for horizon in horizons:
            dim = self.ar_lag * n_comp + 1
            XtX = np.zeros((dim, dim), dtype=np.float64)
            XtY = np.zeros((dim, n_comp), dtype=np.float64)
            for Z in latents:
                acc = _fit_ridge_ar(Z, horizon, self.ar_lag, self.ridge_lambda)
                XtX += acc["XtX"]
                XtY += acc["XtY"]
            W, bias = _solve_ridge_ar(XtX, XtY, self.ridge_lambda)
            self._params[horizon] = {"W": W, "bias": bias}

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict the frame ``horizon`` steps ahead via frozen-basis latent AR.

        Parameters
        ----------
        context : np.ndarray, shape (T, H, W)
            Recent past frames, ``T >= ar_lag``. Only the last ``ar_lag``
            frames are used.
        horizon : int
            Prediction horizon. Must be a key in ``self._params``.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Predicted frame, reconstructed from the latent prediction.
        """
        if horizon not in self._params:
            raise KeyError(f"horizon {horizon} was not fitted")
        if self._basis is None or self._frame_shape is None:
            raise RuntimeError("model must be fitted before calling predict")

        params = self._params[horizon]
        W, bias = params["W"], params["bias"]

        lag_frames = context[-self.ar_lag:]  # (lag, H, W), oldest first
        lag_flat = lag_frames.reshape(self.ar_lag, -1).astype(np.float64)
        lag_latent = self._project(lag_flat)  # (lag, n_components)

        x = lag_latent.reshape(-1)  # (lag * n_components,)
        z_pred = x @ W + bias  # (n_components,)

        frame_flat = self._basis.inverse_transform(z_pred.reshape(1, -1))[0]
        return frame_flat.reshape(self._frame_shape).astype(np.float32)

    def reconstruct_oracle(self, target: np.ndarray) -> np.ndarray:
        """
        Reconstruct a target frame by projecting it onto the frozen basis
        and inverting, bypassing the AR step entirely.

        Gives the best-case RMSE achievable by any AR model sharing this
        frozen basis: the error floor imposed by the basis's limited
        capacity, as opposed to error from the AR prediction itself. Note
        the basis was frozen on the baseline period, so this is an oracle
        with respect to *that* basis, not one refit on task-period data.

        Parameters
        ----------
        target : np.ndarray, shape (H, W)
            Ground-truth frame to reconstruct.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Target frame reconstructed through the frozen basis.
        """
        if self._basis is None or self._frame_shape is None:
            raise RuntimeError("model must be fitted before calling reconstruct_oracle")
        flat = target.reshape(1, -1).astype(np.float64)
        z = self._project(flat)
        recon = self._basis.inverse_transform(z)[0]
        return recon.reshape(self._frame_shape).astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"FrozenBasisAR(method={self.method!r}, n_components={self.n_components}, "
            f"ar_lag={self.ar_lag}, ridge_lambda={self.ridge_lambda}, seed={self.seed})"
        )


# ---------------------------------------------------------------------------
# FrozenBasisRollingMean
# ---------------------------------------------------------------------------

class FrozenBasisRollingMean:
    """
    Rolling-mean predictor computed in a frozen PCA or ICA latent space.

    Same idea as :class:`FrozenBasisAR`'s frozen basis (fit once on the
    baseline period, then only ever ``transform``/``inverse_transform`` on
    task-period frames), but instead of fitting AR weights in the latent
    space, the prediction is just the mean of the last ``window`` latent
    vectors, decoded back to pixel space. Directly comparable to
    :class:`~fuspredict.models.rolling_mean.RollingMeanPredictor`, isolating
    how much of that baseline's error is "real" temporal-averaging error vs.
    error imposed by the basis's limited reconstruction capacity.

    Attributes
    ----------
    name : str
        Human-readable model identifier, ``"frozen_pca_rolling_mean"`` or
        ``"frozen_ica_rolling_mean"`` depending on ``method``.
    method : str
        ``"pca"`` or ``"ica"``.
    n_components : int
        Number of components in the frozen basis.
    window : int
        Number of trailing context frames to average in latent space.
    seed : int
        Random seed for the basis solver.
    """

    name: str = "frozen_basis_rolling_mean"  # overwritten per-instance in __init__

    def __init__(
        self,
        method: str = "pca",
        n_components: int = 10,
        window: int = 10,
        ica_algorithm: dict | None = None,
        seed: int = 0,
    ) -> None:
        """
        Initialize the predictor.

        Parameters
        ----------
        method : str
            ``"pca"`` or ``"ica"``. Default: ``"pca"``.
        n_components : int
            Number of basis components to retain. Default: 10.
        window : int
            Number of trailing context frames to average in latent space.
            Default: 10.
        ica_algorithm : dict, optional
            Extra kwargs forwarded to ``sklearn.decomposition.FastICA``
            (e.g. ``{"algorithm": "parallel", "fun": "logcosh"}``). Ignored
            for ``method="pca"``.
        seed : int
            Random seed for deterministic basis fitting. Default: 0.
        """
        if method not in ("pca", "ica"):
            raise ValueError(f"method must be 'pca' or 'ica', got {method!r}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.method = method
        self.name = f"frozen_{method}_rolling_mean"
        self.n_components = n_components
        self.window = window
        self.ica_algorithm = ica_algorithm or {}
        self.seed = seed
        self._basis: PCA | FastICA | None = None
        self._frame_shape: tuple[int, int] | None = None
        self._fitted_horizons: set[int] = set()

    def _project(self, flat: np.ndarray) -> np.ndarray:
        """Project already-flattened, float64 frames onto the frozen basis."""
        return self._basis.transform(flat)

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
    ) -> None:
        """
        Fit the frozen basis on all given frames (baseline period).

        There are no AR weights to fit — the rolling mean has no
        trainable parameters beyond the basis itself — but ``horizons``
        is still recorded so ``predict`` can validate its argument the
        same way :class:`FrozenBasisAR` does.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session, each of shape ``(T, H, W)``. Intended to
            cover the full baseline period — the basis is fitted on all of
            it and frozen for later use on task-period frames.
        horizons : list of int
            Prediction horizons this model will later be asked to predict.
        """
        if not train_frames:
            raise ValueError("train_frames must contain at least one session")
        self._frame_shape = train_frames[0].shape[1:]

        fit_flat = np.concatenate(
            [f.reshape(f.shape[0], -1) for f in train_frames], axis=0
        ).astype(np.float64)

        n_comp = min(self.n_components, fit_flat.shape[0] - 1, fit_flat.shape[1])
        if self.method == "pca":
            basis = PCA(n_components=n_comp, random_state=self.seed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in divide", RuntimeWarning)
                basis.fit(fit_flat)
        else:
            kwargs = dict(whiten="unit-variance", max_iter=500, random_state=self.seed)
            kwargs.update(self.ica_algorithm)
            basis = FastICA(n_components=n_comp, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                basis.fit(fit_flat)
        self._basis = basis
        self._fitted_horizons = set(horizons)

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict the mean of the last ``window`` context frames, in latent
        space, decoded back to pixel space.

        The same prediction is returned for every horizon, since the
        rolling mean does not depend on how far ahead the forecast is
        requested — matching
        :meth:`~fuspredict.models.rolling_mean.RollingMeanPredictor.predict`.

        Parameters
        ----------
        context : np.ndarray, shape (T, H, W)
            Recent past frames. ``T`` must be >= 1; if ``T < window`` all
            available frames are averaged.
        horizon : int
            Prediction horizon. Must have been passed to ``fit``.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Latent-space rolling mean, reconstructed through the frozen
            basis.
        """
        if horizon not in self._fitted_horizons:
            raise KeyError(f"horizon {horizon} was not fitted")
        if self._basis is None or self._frame_shape is None:
            raise RuntimeError("model must be fitted before calling predict")

        window_frames = context[-self.window:]  # (min(window, T), H, W)
        flat = window_frames.reshape(window_frames.shape[0], -1).astype(np.float64)
        latent = self._project(flat)  # (min(window, T), n_components)
        z_mean = latent.mean(axis=0, keepdims=True)  # (1, n_components)

        frame_flat = self._basis.inverse_transform(z_mean)[0]
        return frame_flat.reshape(self._frame_shape).astype(np.float32)

    def reconstruct_oracle(self, target: np.ndarray) -> np.ndarray:
        """
        Reconstruct a target frame by projecting it onto the frozen basis
        and inverting, bypassing the rolling-mean step entirely.

        Same semantics as :meth:`FrozenBasisAR.reconstruct_oracle` — gives
        the best-case RMSE achievable by any model sharing this frozen
        basis, i.e. the error floor imposed by the basis's limited
        capacity rather than by the rolling-mean prediction itself.

        Parameters
        ----------
        target : np.ndarray, shape (H, W)
            Ground-truth frame to reconstruct.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Target frame reconstructed through the frozen basis.
        """
        if self._basis is None or self._frame_shape is None:
            raise RuntimeError("model must be fitted before calling reconstruct_oracle")
        flat = target.reshape(1, -1).astype(np.float64)
        z = self._project(flat)
        recon = self._basis.inverse_transform(z)[0]
        return recon.reshape(self._frame_shape).astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"FrozenBasisRollingMean(method={self.method!r}, "
            f"n_components={self.n_components}, window={self.window}, seed={self.seed})"
        )
