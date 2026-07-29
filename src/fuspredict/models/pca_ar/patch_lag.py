"""
patch_lag.py
------------
Patch-local, spatiotemporal-lag PCA/ICA AR predictors.

``PatchLagPCAAR``
    Tiles each frame into non-overlapping spatial patches. For each patch,
    builds a spatiotemporal lag matrix, fits a patch-local PCA basis, and
    then ridge-regularized AR weights in that patch's latent space,
    independently per horizon. Predictions are reconstructed per patch and
    stitched back into a full frame.

``PatchLagICAAR``
    Same as ``PatchLagPCAAR``, but each patch's spatiotemporal lag matrix
    is compressed with FastICA instead of PCA.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA, FastICA

from fuspredict.models.pca_ar._ridge import _solve_ridge_ar


# ---------------------------------------------------------------------------
# PatchLagPCAAR
# ---------------------------------------------------------------------------

class PatchLagPCAAR:
    """
    Patch-local, spatiotemporal-lag PCA AR predictor.

    Frames are tiled into non-overlapping ``patch_size x patch_size``
    patches. For each patch, a spatiotemporal lag matrix of shape
    ``(T - p, S * p)`` is built (``S`` = pixels per patch, ``p`` = lag), a
    PCA basis is fitted on that matrix, and ridge AR weights are fitted in
    the resulting latent space, independently per horizon. Predictions are
    reconstructed per patch and stitched into a full frame.

    Attributes
    ----------
    name : str
        Human-readable model identifier, ``"patch_lag_pca_ar"``.
    patch_size : int
        Side length of each square, non-overlapping patch.
    n_components : int
        Number of PCA components per patch.
    ar_lag : int
        Number of autoregressive lags used to build the spatiotemporal
        lag matrix and the latent AR model.
    ridge_lambda : float
        L2 regularization strength.
    seed : int
        Random seed for the randomized PCA solver.
    """

    name: str = "patch_lag_pca_ar"

    @staticmethod
    def tile_patches(H: int, W: int, patch_radius: int) -> list[tuple[slice, slice]]:
        """
        Return non-overlapping patch slices that tile an ``(H, W)`` frame.

        Parameters
        ----------
        H : int
            Frame height.
        W : int
            Frame width.
        patch_radius : int
            Half-size of each square patch (patch side length is
            ``2 * patch_radius``).

        Returns
        -------
        list of (slice, slice)
            Row/column slice pairs tiling the frame. Trailing partial
            patches (when dimensions are not exact multiples of the patch
            size) are included with reduced extent.
        """
        size = 2 * patch_radius
        patches = []
        r0 = 0
        while r0 < H:
            r1 = min(r0 + size, H)
            c0 = 0
            while c0 < W:
                c1 = min(c0 + size, W)
                patches.append((slice(r0, r1), slice(c0, c1)))
                c0 = c1
            r0 = r1
        return patches

    def __init__(
        self,
        patch_size: int = 14,
        n_components: int = 10,
        ar_lag: int = 10,
        ridge_lambda: float = 0.01,
        seed: int = 0,
    ) -> None:
        """
        Initialize the predictor.

        Parameters
        ----------
        patch_size : int
            Side length of each square patch. Default: 14.
        n_components : int
            Number of PCA components per patch. Default: 10.
        ar_lag : int
            Number of autoregressive lags. Default: 10.
        ridge_lambda : float
            L2 regularization strength. Default: 0.01.
        seed : int
            Random seed for deterministic PCA fitting. Default: 0.
        """
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if ar_lag < 1:
            raise ValueError(f"ar_lag must be >= 1, got {ar_lag}")
        if ridge_lambda < 0:
            raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")
        self.patch_size = patch_size
        self.n_components = n_components
        self.ar_lag = ar_lag
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        # Per-patch state, keyed by (row, col) patch index.
        self._patch_pca: dict[tuple[int, int], PCA] = {}
        self._frame_shape: tuple[int, int] | None = None
        self._params: dict[int, dict[tuple[int, int], dict]] = {}

    def _patch_origins(self, h_img: int, w_img: int) -> list[tuple[int, int]]:
        """
        Compute top-left pixel coordinates of all non-overlapping patches.

        Parameters
        ----------
        h_img : int
            Frame height.
        w_img : int
            Frame width.

        Returns
        -------
        list of (int, int)
            ``(row, col)`` pixel coordinates of each patch's top-left corner.
            Trailing partial patches (when dimensions are not exact
            multiples of ``patch_size``) are included with reduced extent.
        """
        rows = list(range(0, h_img, self.patch_size))
        cols = list(range(0, w_img, self.patch_size))
        return [(r, c) for r in rows for c in cols]

    def _extract_patch(
        self,
        frames: np.ndarray,
        r: int,
        c: int,
    ) -> np.ndarray:
        """
        Extract a patch's pixel time series, flattened spatially.

        Parameters
        ----------
        frames : np.ndarray, shape (T, H, W)
            Source frames.
        r : int
            Patch row origin.
        c : int
            Patch column origin.

        Returns
        -------
        np.ndarray, shape (T, S)
            Patch pixel values flattened spatially (``S`` = patch pixel count).
        """
        patch = frames[:, r : r + self.patch_size, c : c + self.patch_size]
        T = patch.shape[0]
        return patch.reshape(T, -1)

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
    ) -> None:
        """
        Fit per-patch PCA bases and ridge AR weights per horizon.

        For each patch, a PCA basis is fitted on the spatiotemporal lag
        matrix (history compression). Ridge regression then maps each
        lag window's latent code directly to the target patch's pixel
        values ``horizon`` steps ahead (no PCA inverse-transform is used
        for the target, since the PCA basis represents lagged history,
        not the prediction target).

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
        self._frame_shape = (h_img, w_img)
        origins = self._patch_origins(h_img, w_img)
        p = self.ar_lag

        for horizon in horizons:
            self._params[horizon] = {}

        for r, c in origins:
            patch_series = [self._extract_patch(f, r, c) for f in train_frames]
            S = patch_series[0].shape[1]

            # Build spatiotemporal lag matrices (T - p, S * p) per session,
            # where row n holds the p most recent patch frames ending at
            # time n + p - 1 (most recent lag first).
            lag_matrices = []
            for series in patch_series:
                T = series.shape[0]
                n_lag_samples = T - p
                if n_lag_samples <= 0:
                    lag_matrices.append(np.empty((0, S * p), dtype=np.float64))
                    continue
                lag_stack = np.stack(
                    [series[p - 1 - k : p - 1 - k + n_lag_samples] for k in range(p)],
                    axis=1,
                )  # (n_lag_samples, p, S)
                lag_matrices.append(
                    lag_stack.reshape(n_lag_samples, p * S).astype(np.float64)
                )

            concat_lag = np.concatenate(
                [m for m in lag_matrices if m.shape[0] > 0], axis=0
            )
            n_components = min(
                self.n_components, concat_lag.shape[0], concat_lag.shape[1]
            )
            pca = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=self.seed,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in divide", RuntimeWarning)
                pca.fit(concat_lag)
            self._patch_pca[(r, c)] = pca

            # Latent code per lag window: row n is the PCA projection of
            # the p-frame history ending at time n + p - 1.
            latents = [
                pca.transform(m) if m.shape[0] > 0 else np.empty((0, n_components))
                for m in lag_matrices
            ]

            for horizon in horizons:
                dim = n_components + 1
                XtX = np.zeros((dim, dim), dtype=np.float64)
                XtY = np.zeros((dim, S), dtype=np.float64)
                for Z, series in zip(latents, patch_series):
                    n_lag_samples = Z.shape[0]
                    n_samples = n_lag_samples - horizon
                    if n_samples <= 0:
                        continue
                    # Latent code for history ending at lag-window index n
                    # (which corresponds to original time p - 1 + n).
                    Zw = Z[:n_samples]
                    ones = np.ones((n_samples, 1), dtype=np.float64)
                    Xb = np.concatenate([Zw, ones], axis=1)  # (n_samples, dim)

                    # Target patch is `horizon` steps after the window end.
                    target_start = p - 1 + horizon
                    Y = series[target_start : target_start + n_samples].astype(
                        np.float64
                    )  # (n_samples, S)

                    XtX += Xb.T @ Xb
                    XtY += Xb.T @ Y

                W, bias = _solve_ridge_ar(XtX, XtY, self.ridge_lambda)
                self._params[horizon][(r, c)] = {
                    "W": W,
                    "bias": bias,
                    "n_components": n_components,
                }

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict the frame ``horizon`` steps ahead, patch by patch.

        Parameters
        ----------
        context : np.ndarray, shape (T, H, W)
            Recent past frames, ``T >= ar_lag``. Only the last ``ar_lag``
            frames are used to build each patch's lag vector.
        horizon : int
            Prediction horizon. Must be a key in ``self._params``.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Predicted frame, stitched from per-patch reconstructions.
        """
        if horizon not in self._params:
            raise KeyError(f"horizon {horizon} was not fitted")
        if self._frame_shape is None:
            raise RuntimeError("model must be fitted before calling predict")

        h_img, w_img = self._frame_shape
        out = np.zeros((h_img, w_img), dtype=np.float32)
        p = self.ar_lag
        lag_context = context[-p:]  # (p, H, W), oldest first

        for (r, c), pca in self._patch_pca.items():
            patch_lag = self._extract_patch(lag_context, r, c)  # (p, S)
            S = patch_lag.shape[1]
            lag_vec = patch_lag.reshape(1, p * S).astype(np.float64)
            z = pca.transform(lag_vec)[0]  # (n_components,)

            params = self._params[horizon][(r, c)]
            W, bias = params["W"], params["bias"]
            patch_flat = z @ W + bias  # (S,)

            patch_size_r = min(self.patch_size, h_img - r)
            patch_size_c = min(self.patch_size, w_img - c)
            out[r : r + patch_size_r, c : c + patch_size_c] = patch_flat.reshape(
                patch_size_r, patch_size_c
            )

        return out

    def __repr__(self) -> str:
        return (
            f"PatchLagPCAAR(patch_size={self.patch_size}, "
            f"n_components={self.n_components}, ar_lag={self.ar_lag}, "
            f"ridge_lambda={self.ridge_lambda}, seed={self.seed})"
        )


# ---------------------------------------------------------------------------
# PatchLagICAAR
# ---------------------------------------------------------------------------

class PatchLagICAAR:
    """
    Patch-local, spatiotemporal-lag ICA AR predictor.

    Identical in structure to ``PatchLagPCAAR``, but each patch's
    spatiotemporal lag matrix is compressed with ``FastICA`` instead of
    PCA. Ridge AR weights are then fitted in the resulting latent space,
    independently per horizon. Predictions are reconstructed per patch and
    stitched into a full frame.

    Attributes
    ----------
    name : str
        Human-readable model identifier, ``"patch_lag_ica_ar"``.
    patch_size : int
        Side length of each square, non-overlapping patch.
    n_components : int
        Number of ICA components per patch.
    ar_lag : int
        Number of autoregressive lags used to build the spatiotemporal
        lag matrix and the latent AR model.
    ridge_lambda : float
        L2 regularization strength.
    ica_algorithm : dict
        Extra kwargs forwarded to ``sklearn.decomposition.FastICA``.
    seed : int
        Random seed for the FastICA solver.
    """

    name: str = "patch_lag_ica_ar"

    tile_patches = PatchLagPCAAR.tile_patches

    def __init__(
        self,
        patch_size: int = 14,
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
        patch_size : int
            Side length of each square patch. Default: 14.
        n_components : int
            Number of ICA components per patch. Default: 10.
        ar_lag : int
            Number of autoregressive lags. Default: 10.
        ridge_lambda : float
            L2 regularization strength. Default: 0.01.
        ica_algorithm : dict, optional
            Extra kwargs forwarded to ``sklearn.decomposition.FastICA``
            (e.g. ``{"algorithm": "parallel", "fun": "logcosh"}``).
        seed : int
            Random seed for deterministic ICA fitting. Default: 0.
        """
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if ar_lag < 1:
            raise ValueError(f"ar_lag must be >= 1, got {ar_lag}")
        if ridge_lambda < 0:
            raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")
        self.patch_size = patch_size
        self.n_components = n_components
        self.ar_lag = ar_lag
        self.ridge_lambda = ridge_lambda
        self.ica_algorithm = ica_algorithm or {}
        self.seed = seed
        # Per-patch state, keyed by (row, col) patch index.
        self._patch_ica: dict[tuple[int, int], FastICA] = {}
        self._frame_shape: tuple[int, int] | None = None
        self._params: dict[int, dict[tuple[int, int], dict]] = {}

    _patch_origins = PatchLagPCAAR._patch_origins
    _extract_patch = PatchLagPCAAR._extract_patch

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
    ) -> None:
        """
        Fit per-patch ICA bases and ridge AR weights per horizon.

        For each patch, an ICA basis is fitted on the spatiotemporal lag
        matrix (history compression). Ridge regression then maps each
        lag window's latent code directly to the target patch's pixel
        values ``horizon`` steps ahead (no ICA inverse-transform is used
        for the target, since the ICA basis represents lagged history,
        not the prediction target).

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
        self._frame_shape = (h_img, w_img)
        origins = self._patch_origins(h_img, w_img)
        p = self.ar_lag

        for horizon in horizons:
            self._params[horizon] = {}

        for r, c in origins:
            patch_series = [self._extract_patch(f, r, c) for f in train_frames]
            S = patch_series[0].shape[1]

            # Build spatiotemporal lag matrices (T - p, S * p) per session,
            # where row n holds the p most recent patch frames ending at
            # time n + p - 1 (most recent lag first).
            lag_matrices = []
            for series in patch_series:
                T = series.shape[0]
                n_lag_samples = T - p
                if n_lag_samples <= 0:
                    lag_matrices.append(np.empty((0, S * p), dtype=np.float64))
                    continue
                lag_stack = np.stack(
                    [series[p - 1 - k : p - 1 - k + n_lag_samples] for k in range(p)],
                    axis=1,
                )  # (n_lag_samples, p, S)
                lag_matrices.append(
                    lag_stack.reshape(n_lag_samples, p * S).astype(np.float64)
                )

            concat_lag = np.concatenate(
                [m for m in lag_matrices if m.shape[0] > 0], axis=0
            )
            n_components = min(
                self.n_components, concat_lag.shape[0] - 1, concat_lag.shape[1]
            )
            kwargs = dict(whiten="unit-variance", max_iter=500, random_state=self.seed)
            kwargs.update(self.ica_algorithm)
            ica = FastICA(n_components=n_components, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ica.fit(concat_lag)
            self._patch_ica[(r, c)] = ica

            # Latent code per lag window: row n is the ICA projection of
            # the p-frame history ending at time n + p - 1.
            latents = [
                ica.transform(m) if m.shape[0] > 0 else np.empty((0, n_components))
                for m in lag_matrices
            ]

            for horizon in horizons:
                dim = n_components + 1
                XtX = np.zeros((dim, dim), dtype=np.float64)
                XtY = np.zeros((dim, S), dtype=np.float64)
                for Z, series in zip(latents, patch_series):
                    n_lag_samples = Z.shape[0]
                    n_samples = n_lag_samples - horizon
                    if n_samples <= 0:
                        continue
                    # Latent code for history ending at lag-window index n
                    # (which corresponds to original time p - 1 + n).
                    Zw = Z[:n_samples]
                    ones = np.ones((n_samples, 1), dtype=np.float64)
                    Xb = np.concatenate([Zw, ones], axis=1)  # (n_samples, dim)

                    # Target patch is `horizon` steps after the window end.
                    target_start = p - 1 + horizon
                    Y = series[target_start : target_start + n_samples].astype(
                        np.float64
                    )  # (n_samples, S)

                    XtX += Xb.T @ Xb
                    XtY += Xb.T @ Y

                W, bias = _solve_ridge_ar(XtX, XtY, self.ridge_lambda)
                self._params[horizon][(r, c)] = {
                    "W": W,
                    "bias": bias,
                    "n_components": n_components,
                }

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict the frame ``horizon`` steps ahead, patch by patch.

        Parameters
        ----------
        context : np.ndarray, shape (T, H, W)
            Recent past frames, ``T >= ar_lag``. Only the last ``ar_lag``
            frames are used to build each patch's lag vector.
        horizon : int
            Prediction horizon. Must be a key in ``self._params``.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Predicted frame, stitched from per-patch reconstructions.
        """
        if horizon not in self._params:
            raise KeyError(f"horizon {horizon} was not fitted")
        if self._frame_shape is None:
            raise RuntimeError("model must be fitted before calling predict")

        h_img, w_img = self._frame_shape
        out = np.zeros((h_img, w_img), dtype=np.float32)
        p = self.ar_lag
        lag_context = context[-p:]  # (p, H, W), oldest first

        for (r, c), ica in self._patch_ica.items():
            patch_lag = self._extract_patch(lag_context, r, c)  # (p, S)
            S = patch_lag.shape[1]
            lag_vec = patch_lag.reshape(1, p * S).astype(np.float64)
            z = ica.transform(lag_vec)[0]  # (n_components,)

            params = self._params[horizon][(r, c)]
            W, bias = params["W"], params["bias"]
            patch_flat = z @ W + bias  # (S,)

            patch_size_r = min(self.patch_size, h_img - r)
            patch_size_c = min(self.patch_size, w_img - c)
            out[r : r + patch_size_r, c : c + patch_size_c] = patch_flat.reshape(
                patch_size_r, patch_size_c
            )

        return out

    def __repr__(self) -> str:
        return (
            f"PatchLagICAAR(patch_size={self.patch_size}, "
            f"n_components={self.n_components}, ar_lag={self.ar_lag}, "
            f"ridge_lambda={self.ridge_lambda}, seed={self.seed})"
        )
