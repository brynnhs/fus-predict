"""
patch_pca.py
------------
ConvLSTM over a spatial grid of per-patch PCA latent codes.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from fuspredict.models.convlstm.cell import _ConvLSTMForecaster


class PatchPCAConvLSTM:
    """
    ConvLSTM over a spatial grid of per-patch PCA latent codes.

    Unlike ``ConvLSTMPredictor(input_mode="pca")``, which fits a single
    full-frame PCA basis and collapses all spatial structure into a
    ``(n_components, 1, 1)`` vector (making the "Conv" in ConvLSTM a no-op),
    this model tiles each frame into non-overlapping patches (same tiling
    convention as ``PatchLagPCAAR``), fits an independent PCA basis per
    patch, and stacks the per-patch latent vectors into a genuine 2D grid of
    shape ``(n_components, H_patches, W_patches)``. The ConvLSTM's
    spatial kernel then convolves over neighboring patches' latent codes,
    same as it would over neighboring pixels in raw-frame mode. Predictions
    are decoded per patch (inverse PCA) and stitched back into a full frame.

    Attributes
    ----------
    name : str
        Human-readable model identifier, ``"patch_pca_convlstm"``.
    patch_size : int
        Side length of each square, non-overlapping patch.
    n_components : int
        Number of PCA components fitted per patch (= channel count of the
        latent grid).
    hidden_channels : int
        Number of hidden/cell channels in the ConvLSTM cell.
    kernel_size : int
        Size of the (square) ConvLSTM convolution kernel, applied over the
        patch grid.
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
        Random seed for deterministic PCA fitting, weight initialization,
        and batching.
    """

    name: str = "patch_pca_convlstm"

    def __init__(
        self,
        patch_size: int = 14,
        n_components: int = 6,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        lag: int = 10,
        lr: float = 3e-4,
        batch_size: int = 128,
        n_epochs: int = 50,
        grad_clip_norm: float = 1.0,
        seed: int = 0,
    ) -> None:
        """
        Initialize the predictor.

        Parameters
        ----------
        patch_size : int
            Side length of each square patch. Default: 14.
        n_components : int
            Number of PCA components fitted per patch. Default: 6.
        hidden_channels : int
            Number of hidden/cell channels. Default: 32.
        kernel_size : int
            ConvLSTM convolution kernel size, applied over the patch grid.
            Default: 3.
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
        """
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be >= 1, got {hidden_channels}")
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        self.patch_size = patch_size
        self.n_components = n_components
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.lag = lag
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PatchPCAConvLSTM running on: {self.device}")
        # Per-patch PCA bases, keyed by (row, col) patch index, in a fixed
        # row-major order matching the latent grid's spatial axes.
        self._patch_pca: dict[tuple[int, int], PCA] = {}
        self._patch_origins: list[tuple[int, int]] = []
        self._grid_shape: tuple[int, int] | None = None  # (n_rows, n_cols)
        self._patch_extents: dict[tuple[int, int], tuple[int, int]] = {}
        self._frame_shape: tuple[int, int] | None = None
        self._params: dict[int, dict] = {}

    def _fit_patch_bases(self, train_frames: list[np.ndarray]) -> None:
        """Fit one PCA basis per patch on all training frames."""
        h_img, w_img = train_frames[0].shape[1:]
        self._frame_shape = (h_img, w_img)

        rows = list(range(0, h_img, self.patch_size))
        cols = list(range(0, w_img, self.patch_size))
        self._grid_shape = (len(rows), len(cols))
        self._patch_origins = [(r, c) for r in rows for c in cols]

        for r, c in self._patch_origins:
            r1 = min(r + self.patch_size, h_img)
            c1 = min(c + self.patch_size, w_img)
            self._patch_extents[(r, c)] = (r1 - r, c1 - c)

            patch_series = [
                f[:, r:r1, c:c1].reshape(f.shape[0], -1) for f in train_frames
            ]
            flat = np.concatenate(patch_series, axis=0).astype(np.float64)
            n_comp = min(self.n_components, flat.shape[0] - 1, flat.shape[1])
            pca = PCA(n_components=n_comp, random_state=self.seed)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "invalid value encountered in divide", RuntimeWarning
                )
                pca.fit(flat)
            self._patch_pca[(r, c)] = pca

    def _to_latent_grid(self, frames: np.ndarray) -> np.ndarray:
        """
        Project (T, H, W) frames onto the per-patch PCA bases.

        Returns
        -------
        np.ndarray, shape (T, n_components, n_rows, n_cols)
            Latent grid; components with index >= a given patch's own
            fitted rank are zero-padded.
        """
        T = frames.shape[0]
        n_rows, n_cols = self._grid_shape
        grid = np.zeros((T, self.n_components, n_rows, n_cols), dtype=np.float32)
        for gi, (r, c) in enumerate(self._patch_origins):
            pr, pc = self._patch_extents[(r, c)]
            patch = frames[:, r : r + pr, c : c + pc].reshape(T, -1).astype(np.float64)
            z = self._patch_pca[(r, c)].transform(patch)  # (T, n_comp_patch)
            gi_row, gi_col = divmod(gi, n_cols)
            grid[:, : z.shape[1], gi_row, gi_col] = z
        return grid

    def _from_latent_grid(self, z_grid: np.ndarray) -> np.ndarray:
        """
        Reconstruct an (H, W) pixel frame from a latent grid.

        Parameters
        ----------
        z_grid : np.ndarray, shape (n_components, n_rows, n_cols)
        """
        h_img, w_img = self._frame_shape
        out = np.zeros((h_img, w_img), dtype=np.float32)
        n_cols = self._grid_shape[1]
        for gi, (r, c) in enumerate(self._patch_origins):
            pca = self._patch_pca[(r, c)]
            n_comp_patch = pca.n_components_
            gi_row, gi_col = divmod(gi, n_cols)
            z = z_grid[:n_comp_patch, gi_row, gi_col].reshape(1, -1).astype(np.float64)
            recon = pca.inverse_transform(z)[0]
            pr, pc = self._patch_extents[(r, c)]
            out[r : r + pr, c : c + pc] = recon.reshape(pr, pc)
        return out

    def _build_windows(
        self,
        latent_seqs: list[np.ndarray],
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build sliding-window input/target tensors across all sessions.

        Parameters
        ----------
        latent_seqs : list of np.ndarray, each (T, n_components, n_rows, n_cols)
        horizon : int
            Prediction horizon.

        Returns
        -------
        X : torch.Tensor, shape (N, lag, n_components, n_rows, n_cols)
        Y : torch.Tensor, shape (N, n_components, n_rows, n_cols)
        """
        p = self.lag
        X_list = []
        Y_list = []
        for seq in latent_seqs:
            T = seq.shape[0]
            n_samples = T - p - horizon + 1
            if n_samples <= 0:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(
                seq, window_shape=p, axis=0
            )[:n_samples]  # (n_samples, C, n_rows, n_cols, p)
            windows = np.moveaxis(windows, -1, 1)  # (n_samples, p, C, n_rows, n_cols)
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
        return torch.from_numpy(X), torch.from_numpy(Y)

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
    ) -> None:
        """
        Fit per-patch PCA bases, then train one ConvLSTM per horizon over
        the resulting latent grid.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session, each of shape ``(T, H, W)``.
        horizons : list of int
            Prediction horizons to fit.
        """
        if not train_frames:
            raise ValueError("train_frames must contain at least one session")

        self._fit_patch_bases(train_frames)
        latent_seqs = [self._to_latent_grid(f) for f in train_frames]

        for horizon in horizons:
            torch.manual_seed(self.seed)
            generator = torch.Generator().manual_seed(self.seed)

            X, Y = self._build_windows(latent_seqs, horizon)
            n_samples = X.shape[0]

            model = _ConvLSTMForecaster(
                hidden_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                channels=self.n_components,
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
            Predicted frame, stitched from per-patch reconstructions.
        """
        if horizon not in self._params:
            raise KeyError(f"horizon {horizon} was not fitted")

        model = self._params[horizon]["model"]

        lag_latent = self._to_latent_grid(context[-self.lag :])  # (lag, C, R, Cc)
        x = torch.from_numpy(lag_latent).unsqueeze(0).to(self.device)  # (1, lag, C, R, Cc)

        model.eval()
        with torch.no_grad():
            pred = model(x)  # (1, C, R, Cc)

        z_grid = pred[0].cpu().numpy().astype(np.float32)
        return self._from_latent_grid(z_grid)

    def reconstruct_oracle(self, target: np.ndarray) -> np.ndarray:
        """
        Reconstruct a target frame by projecting it onto the per-patch PCA
        bases and inverting, bypassing the ConvLSTM prediction entirely.

        Gives the best-case RMSE achievable by any model sharing these
        per-patch bases: the error floor imposed by the bases' limited
        capacity, as opposed to error from the learned temporal prediction.

        Parameters
        ----------
        target : np.ndarray, shape (H, W)
            Ground-truth frame to reconstruct.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Target frame reconstructed through the per-patch PCA bases.
        """
        if self._grid_shape is None or self._frame_shape is None:
            raise RuntimeError("model must be fitted before calling reconstruct_oracle")
        z_grid = self._to_latent_grid(target[np.newaxis])[0]  # (C, n_rows, n_cols)
        return self._from_latent_grid(z_grid)

    def __repr__(self) -> str:
        return (
            f"PatchPCAConvLSTM(patch_size={self.patch_size}, "
            f"n_components={self.n_components}, hidden_channels={self.hidden_channels}, "
            f"kernel_size={self.kernel_size}, lag={self.lag}, lr={self.lr}, "
            f"batch_size={self.batch_size}, n_epochs={self.n_epochs}, "
            f"grad_clip_norm={self.grad_clip_norm}, seed={self.seed})"
        )
