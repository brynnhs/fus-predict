"""
predictor.py
------------
Direct-horizon frame predictor based on a single-layer ConvLSTM, trained
on either raw pixel frames or a full-frame PCA/ICA latent projection.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA, FastICA

from fuspredict.models.convlstm.cell import _ConvLSTMForecaster


class ConvLSTMPredictor:
    """
    Direct-horizon frame predictor based on a single-layer ConvLSTM.

    A separate ConvLSTM model is trained per horizon using sliding windows
    of ``lag`` consecutive context frames as input and the frame
    ``horizon`` steps after the window as the regression target. Training
    minimizes MSE with Adam and gradient-norm clipping.

    Attributes
    ----------
    name : str
        Human-readable model identifier, ``"convlstm"``.
    hidden_channels : int
        Number of hidden/cell channels in the ConvLSTM cell.
    kernel_size : int
        Size of the (square) ConvLSTM convolution kernel.
    lag : int
        Number of context frames fed into the ConvLSTM per prediction.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size used during training.
    n_epochs : int
        Number of training epochs per horizon.
    grad_clip_norm : float
        Maximum gradient L2 norm for clipping.
    seed : int
        Random seed for deterministic weight initialization and batching.
    input_mode : str
        ``"frames"`` (default) runs the ConvLSTM directly on raw pixel
        frames. ``"pca"`` or ``"ica"`` first project each frame onto a
        ``n_components``-dimensional basis (fitted once on the training
        frames) and run the ConvLSTM on the resulting component sequence,
        treated as a ``(n_components, 1, 1)`` "image" (i.e. components are
        channels with no spatial extent, so ``kernel_size`` is forced to 1
        in this mode). Predictions are inverse-transformed back to pixel
        space.
    n_components : int
        Number of PCA/ICA components to retain. Only used when
        ``input_mode`` is ``"pca"`` or ``"ica"``.
    ica_algorithm : dict, optional
        Extra kwargs forwarded to ``sklearn.decomposition.FastICA``. Only
        used when ``input_mode == "ica"``.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        lag: int = 10,
        lr: float = 3e-4,
        batch_size: int = 128,
        n_epochs: int = 50,
        grad_clip_norm: float = 1.0,
        seed: int = 0,
        input_mode: str = "frames",
        n_components: int = 20,
        ica_algorithm: dict | None = None,
    ) -> None:
        """
        Initialize the predictor.

        Parameters
        ----------
        hidden_channels : int
            Number of hidden/cell channels. Default: 32.
        kernel_size : int
            ConvLSTM convolution kernel size. Ignored (forced to 1) when
            ``input_mode`` is ``"pca"`` or ``"ica"``. Default: 3.
        lag : int
            Number of context frames used per prediction. Default: 10.
        lr : float
            Adam learning rate. Default: 3e-4.
        batch_size : int
            Mini-batch size during training. Default: 128.
        n_epochs : int
            Number of training epochs per horizon. Default: 50.
        grad_clip_norm : float
            Maximum gradient L2 norm for clipping. Default: 1.0.
        seed : int
            Random seed for deterministic training. Default: 0.
        input_mode : str
            ``"frames"``, ``"pca"``, or ``"ica"``. Default: ``"frames"``.
        n_components : int
            Number of PCA/ICA components to retain (latent modes only).
            Default: 20.
        ica_algorithm : dict, optional
            Extra kwargs forwarded to ``FastICA`` (``input_mode="ica"`` only).
        """
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be >= 1, got {hidden_channels}")
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        if input_mode not in ("frames", "pca", "ica"):
            raise ValueError(
                f"input_mode must be 'frames', 'pca', or 'ica', got {input_mode!r}"
            )
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size if input_mode == "frames" else 1
        self.lag = lag
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.input_mode = input_mode
        self.n_components = n_components
        self.ica_algorithm = ica_algorithm or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ConvLSTM running on: {self.device}")
        self.name = (
            "convlstm" if input_mode == "frames" else f"convlstm_{input_mode}_latent"
        )
        self._params: dict[int, dict] = {}
        self._basis: PCA | FastICA | None = None
        self._frame_shape: tuple[int, int] | None = None

    def _fit_basis(self, train_frames: list[np.ndarray]) -> None:
        """Fit the PCA/ICA basis on all training frames (latent modes only)."""
        self._frame_shape = train_frames[0].shape[1:]
        flattened = np.concatenate(
            [f.reshape(f.shape[0], -1) for f in train_frames], axis=0
        ).astype(np.float64)
        n_comp = min(self.n_components, flattened.shape[0] - 1, flattened.shape[1])
        if self.input_mode == "pca":
            basis = PCA(n_components=n_comp, random_state=self.seed)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "invalid value encountered in divide", RuntimeWarning
                )
                basis.fit(flattened)
        else:
            kwargs = dict(whiten="unit-variance", max_iter=500, random_state=self.seed)
            kwargs.update(self.ica_algorithm)
            basis = FastICA(n_components=n_comp, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                basis.fit(flattened)
        self._basis = basis

    def _to_latent(self, frames: np.ndarray) -> np.ndarray:
        """Project (T, H, W) frames onto the fitted basis -> (T, n_components)."""
        flat = frames.reshape(frames.shape[0], -1).astype(np.float64)
        return self._basis.transform(flat).astype(np.float32)

    def _from_latent(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct (H, W) pixel frame(s) from latent component vector(s)."""
        flat = self._basis.inverse_transform(z.reshape(1, -1))[0]
        return flat.reshape(self._frame_shape).astype(np.float32)

    def _sequences_for_windows(self, train_frames: list[np.ndarray]) -> list[np.ndarray]:
        """Return per-session (T, C) sequences to build windows from."""
        if self.input_mode == "frames":
            return train_frames
        return [self._to_latent(f) for f in train_frames]

    def _build_windows(
        self,
        train_frames: list[np.ndarray],
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build sliding-window input/target tensors across all sessions.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session. In ``"frames"`` mode each has shape
            ``(T, H, W)``; in ``"pca"``/``"ica"`` mode each has already been
            projected to shape ``(T, C)`` (``C`` = ``n_components``).
        horizon : int
            Prediction horizon.

        Returns
        -------
        X : torch.Tensor, shape (N, lag, C, H', W')
            Input context windows (``H'=W'=1`` and ``C=n_components`` in
            latent modes; ``C=1`` and ``H'=H, W'=W`` in frame mode).
        Y : torch.Tensor, shape (N, C, H', W')
            Target frames/vectors, ``horizon`` steps after each window.
        """
        p = self.lag
        latent = self.input_mode != "frames"
        X_list = []
        Y_list = []
        for seq in train_frames:
            T = seq.shape[0]
            n_samples = T - p - horizon + 1
            if n_samples <= 0:
                continue
            # sliding_window_view returns a view (no copy) of shape (n_samples, ..., p);
            # move the window axis next to the sample axis and copy once for contiguity.
            windows = np.lib.stride_tricks.sliding_window_view(
                seq, window_shape=p, axis=0
            )[:n_samples]  # (n_samples, ..., p)
            windows = np.moveaxis(windows, -1, 1)  # (n_samples, p, ...)
            targets = seq[p - 1 + horizon : p - 1 + horizon + n_samples]
            X_list.append(windows)
            Y_list.append(targets)

        if not X_list:
            raise ValueError(
                f"no training windows available for horizon={horizon} "
                f"with lag={p}; check that train_frames are long enough"
            )

        X = np.concatenate(X_list, axis=0).astype(np.float32)
        Y = np.concatenate(Y_list, axis=0).astype(np.float32)
        if latent:
            X_t = torch.from_numpy(X).unsqueeze(-1).unsqueeze(-1)  # (N, p, C, 1, 1)
            Y_t = torch.from_numpy(Y).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        else:
            X_t = torch.from_numpy(X).unsqueeze(2)  # (N, p, 1, H, W)
            Y_t = torch.from_numpy(Y).unsqueeze(1)  # (N, 1, H, W)
        return X_t, Y_t

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
        vessel_mask: np.ndarray | None = None,
    ) -> None:
        """
        Train one ConvLSTM model per horizon via MSE loss and Adam.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session, each of shape ``(T, H, W)``.
        horizons : list of int
            Prediction horizons to fit.
        vessel_mask : np.ndarray or None, shape (H, W)
            Boolean mask. If provided, loss is computed only over vessel
            pixels. Ignored when ``input_mode`` is ``"pca"`` or ``"ica"``,
            since training happens in latent (non-spatial) space.
        """
        if not train_frames:
            raise ValueError("train_frames must contain at least one session")

        if self.input_mode == "frames":
            channels = 1
            mask_t: torch.Tensor | None = None
            if vessel_mask is not None:
                mask_t = torch.from_numpy(vessel_mask.astype(np.float32)).to(
                    self.device
                )
            sequences = train_frames
        else:
            self._fit_basis(train_frames)
            channels = self._basis.components_.shape[0]
            mask_t = None
            sequences = self._sequences_for_windows(train_frames)

        for horizon in horizons:
            torch.manual_seed(self.seed)
            generator = torch.Generator().manual_seed(self.seed)

            X, Y = self._build_windows(sequences, horizon)
            n_samples = X.shape[0]

            model = _ConvLSTMForecaster(
                hidden_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                channels=channels,
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            model.train()
            for _ in range(self.n_epochs):
                perm = torch.randperm(n_samples, generator=generator)
                for start in range(0, n_samples, self.batch_size):
                    idx = perm[start : start + self.batch_size]
                    xb = X[idx].to(self.device)
                    yb = Y[idx].to(self.device)

                    optimizer.zero_grad()
                    pred = model(xb)
                    if mask_t is not None:
                        loss = ((pred - yb) ** 2 * mask_t).sum() / mask_t.sum()
                    else:
                        loss = nn.functional.mse_loss(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                    optimizer.step()

            model.eval()
            self._params[horizon] = {"model": model}

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict the frame ``horizon`` steps ahead using the trained ConvLSTM.

        Parameters
        ----------
        context : np.ndarray, shape (T, H, W)
            Recent past frames, ``T >= lag``. Only the last ``lag`` frames
            are used.
        horizon : int
            Prediction horizon. Must be a key in ``self._params``.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Predicted frame.
        """
        if horizon not in self._params:
            raise KeyError(f"horizon {horizon} was not fitted")

        model = self._params[horizon]["model"]

        if self.input_mode == "frames":
            lag_frames = context[-self.lag:].astype(np.float32)  # (lag, H, W)
            x = torch.from_numpy(lag_frames).unsqueeze(0).unsqueeze(2)  # (1, lag, 1, H, W)
            x = x.to(self.device)

            model.eval()
            with torch.no_grad():
                pred = model(x)  # (1, 1, H, W)

            return pred[0, 0].cpu().numpy().astype(np.float32)

        lag_latent = self._to_latent(context[-self.lag:])  # (lag, C)
        x = torch.from_numpy(lag_latent).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x.to(self.device)  # (1, lag, C, 1, 1)

        model.eval()
        with torch.no_grad():
            pred = model(x)  # (1, C, 1, 1)

        z_pred = pred[0, :, 0, 0].cpu().numpy().astype(np.float32)  # (C,)
        return self._from_latent(z_pred)

    def predict_batch(self, contexts: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict frames for a batch of context windows in a single forward pass.

        Parameters
        ----------
        contexts : np.ndarray, shape (N, lag, H, W)
            Batch of context windows (already trimmed to ``lag`` frames each).
        horizon : int
            Prediction horizon. Must be a key in ``self._params``.

        Returns
        -------
        np.ndarray, shape (N, H, W), dtype float32
            Predicted frames.
        """
        if horizon not in self._params:
            raise KeyError(f"horizon {horizon} was not fitted")

        model = self._params[horizon]["model"]

        if self.input_mode == "frames":
            x = torch.from_numpy(contexts.astype(np.float32)).unsqueeze(2)  # (N, lag, 1, H, W)
            model.eval()
            results = []
            with torch.no_grad():
                for start in range(0, x.shape[0], self.batch_size):
                    xb = x[start : start + self.batch_size].to(self.device)
                    results.append(model(xb).cpu())
            return torch.cat(results, dim=0)[:, 0].numpy().astype(np.float32)

        lag_latent = np.stack(
            [self._to_latent(ctx) for ctx in contexts], axis=0
        )  # (N, lag, C)
        x = torch.from_numpy(lag_latent).unsqueeze(-1).unsqueeze(-1)  # (N, lag, C, 1, 1)
        model.eval()
        results = []
        with torch.no_grad():
            for start in range(0, x.shape[0], self.batch_size):
                xb = x[start : start + self.batch_size].to(self.device)
                results.append(model(xb).cpu())
        z_preds = torch.cat(results, dim=0)[:, :, 0, 0].numpy().astype(np.float32)  # (N, C)
        return np.stack([self._from_latent(z) for z in z_preds], axis=0)

    def reconstruct_oracle(self, target: np.ndarray) -> np.ndarray:
        """
        Reconstruct a target frame by projecting it onto the fitted PCA/ICA
        basis and inverting, bypassing the ConvLSTM prediction entirely.

        Gives the best-case RMSE achievable by any model sharing this basis:
        the error floor imposed by the basis's limited capacity, as opposed
        to error from the learned temporal prediction. Only meaningful in
        latent (``"pca"``/``"ica"``) input modes; raises in ``"frames"``
        mode since there is no basis to reconstruct through.

        Parameters
        ----------
        target : np.ndarray, shape (H, W)
            Ground-truth frame to reconstruct.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Target frame reconstructed through the frozen basis.
        """
        if self.input_mode == "frames":
            raise AttributeError(
                "reconstruct_oracle is not defined for input_mode='frames'"
            )
        if self._basis is None or self._frame_shape is None:
            raise RuntimeError("model must be fitted before calling reconstruct_oracle")
        z = self._to_latent(target[np.newaxis])[0]
        return self._from_latent(z)

    def __repr__(self) -> str:
        return (
            f"ConvLSTMPredictor(hidden_channels={self.hidden_channels}, "
            f"kernel_size={self.kernel_size}, lag={self.lag}, lr={self.lr}, "
            f"batch_size={self.batch_size}, n_epochs={self.n_epochs}, "
            f"grad_clip_norm={self.grad_clip_norm}, seed={self.seed}, "
            f"input_mode={self.input_mode!r}, n_components={self.n_components})"
        )
