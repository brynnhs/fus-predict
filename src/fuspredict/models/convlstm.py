"""
convlstm.py
-----------
Single-layer ConvLSTM predictor with a 1x1 convolutional readout.

Trains one ConvLSTM model per horizon, using sliding context windows built
from the training frames as input sequences and the frame ``horizon`` steps
after the window as the target. Optimized with Adam and MSE loss, with
gradient-norm clipping for stability.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class ConvLSTMCell(nn.Module):
    """
    Single-layer convolutional LSTM cell.

    Implements the standard ConvLSTM recurrence (Shi et al., 2015) using a
    single convolution producing all four gates. The forget gate bias is
    initialized to 1.0 to encourage long-term memory retention early in
    training.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden/cell state channels.
    kernel_size : int
        Size of the (square) convolutional kernel.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
    ) -> None:
        """
        Initialize the ConvLSTM cell and its gate convolution.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        hidden_channels : int
            Number of hidden/cell state channels.
        kernel_size : int
            Size of the (square) convolutional kernel.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self._init_forget_bias()

    def _init_forget_bias(self) -> None:
        """Initialize the forget gate's bias slice to 1.0."""
        with torch.no_grad():
            bias = self.conv.bias
            c = self.hidden_channels
            # Gate order: input, forget, cell(candidate), output.
            bias[c : 2 * c].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one ConvLSTM time step.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C_in, H, W)
            Input at the current time step.
        h_prev : torch.Tensor, shape (B, C_hidden, H, W)
            Previous hidden state.
        c_prev : torch.Tensor, shape (B, C_hidden, H, W)
            Previous cell state.

        Returns
        -------
        h : torch.Tensor, shape (B, C_hidden, H, W)
            Updated hidden state.
        c : torch.Tensor, shape (B, C_hidden, H, W)
            Updated cell state.
        """
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class _ConvLSTMForecaster(nn.Module):
    """
    ConvLSTM sequence encoder with a 1x1 convolutional readout head.

    Runs the ConvLSTM cell over an input sequence and applies a 1x1
    convolution to the final hidden state to produce a single-channel
    predicted frame.

    Parameters
    ----------
    hidden_channels : int
        Number of hidden/cell state channels.
    kernel_size : int
        Size of the (square) ConvLSTM kernel.
    """

    def __init__(self, hidden_channels: int, kernel_size: int) -> None:
        """
        Initialize the forecaster.

        Parameters
        ----------
        hidden_channels : int
            Number of hidden/cell state channels.
        kernel_size : int
            Size of the (square) ConvLSTM kernel.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(
            in_channels=1,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.readout = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Run the ConvLSTM over a sequence and predict the next frame.

        Parameters
        ----------
        x_seq : torch.Tensor, shape (B, T, 1, H, W)
            Input sequence of context frames.

        Returns
        -------
        torch.Tensor, shape (B, 1, H, W)
            Predicted frame.
        """
        B, T, _, H, W = x_seq.shape
        device = x_seq.device
        h = torch.zeros(B, self.hidden_channels, H, W, device=device)
        c = torch.zeros(B, self.hidden_channels, H, W, device=device)
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c)
        return self.readout(h)


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
    """

    name: str = "convlstm"

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
    ) -> None:
        """
        Initialize the predictor.

        Parameters
        ----------
        hidden_channels : int
            Number of hidden/cell channels. Default: 32.
        kernel_size : int
            ConvLSTM convolution kernel size. Default: 3.
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
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be >= 1, got {hidden_channels}")
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.lag = lag
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ConvLSTM running on: {self.device}")
        self._params: dict[int, dict] = {}

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
            One array per session, each of shape ``(T, H, W)``.
        horizon : int
            Prediction horizon.

        Returns
        -------
        X : torch.Tensor, shape (N, lag, 1, H, W)
            Input context windows.
        Y : torch.Tensor, shape (N, 1, H, W)
            Target frames, ``horizon`` steps after each window.
        """
        p = self.lag
        X_list = []
        Y_list = []
        for frames in train_frames:
            T = frames.shape[0]
            n_samples = T - p - horizon + 1
            if n_samples <= 0:
                continue
            # sliding_window_view returns a view (no copy) of shape (n_samples, H, W, p);
            # transpose to (n_samples, p, H, W) and copy once for contiguity.
            windows = np.lib.stride_tricks.sliding_window_view(
                frames, window_shape=p, axis=0
            )[:n_samples]  # (n_samples, H, W, p)
            windows = windows.transpose(0, 3, 1, 2)  # (n_samples, p, H, W)
            targets = frames[p - 1 + horizon : p - 1 + horizon + n_samples]
            X_list.append(windows)
            Y_list.append(targets)

        if not X_list:
            raise ValueError(
                f"no training windows available for horizon={horizon} "
                f"with lag={p}; check that train_frames are long enough"
            )

        X = np.concatenate(X_list, axis=0).astype(np.float32)
        Y = np.concatenate(Y_list, axis=0).astype(np.float32)
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
            Boolean mask. If provided, loss is computed only over vessel pixels.
        """
        if not train_frames:
            raise ValueError("train_frames must contain at least one session")

        mask_t: torch.Tensor | None = None
        if vessel_mask is not None:
            mask_t = torch.from_numpy(vessel_mask.astype(np.float32)).to(self.device)

        for horizon in horizons:
            torch.manual_seed(self.seed)
            generator = torch.Generator().manual_seed(self.seed)

            X, Y = self._build_windows(train_frames, horizon)
            n_samples = X.shape[0]

            model = _ConvLSTMForecaster(
                hidden_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
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
        lag_frames = context[-self.lag:].astype(np.float32)  # (lag, H, W)
        x = torch.from_numpy(lag_frames).unsqueeze(0).unsqueeze(2)  # (1, lag, 1, H, W)
        x = x.to(self.device)

        model.eval()
        with torch.no_grad():
            pred = model(x)  # (1, 1, H, W)

        return pred[0, 0].cpu().numpy().astype(np.float32)

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
        x = torch.from_numpy(contexts.astype(np.float32)).unsqueeze(2)  # (N, lag, 1, H, W)
        model.eval()
        results = []
        with torch.no_grad():
            for start in range(0, x.shape[0], self.batch_size):
                xb = x[start : start + self.batch_size].to(self.device)
                results.append(model(xb).cpu())
        return torch.cat(results, dim=0)[:, 0].numpy().astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"ConvLSTMPredictor(hidden_channels={self.hidden_channels}, "
            f"kernel_size={self.kernel_size}, lag={self.lag}, lr={self.lr}, "
            f"batch_size={self.batch_size}, n_epochs={self.n_epochs}, "
            f"grad_clip_norm={self.grad_clip_norm}, seed={self.seed})"
        )


class ConvLSTMVesselMaskedInput(ConvLSTMPredictor):
    """
    ConvLSTM variant that zeros out non-vessel pixels in all input and target
    frames before training, so the model only ever sees vessel signal.

    The vessel_mask must be supplied via ``set_vessel_mask`` before calling
    ``fit``; if no mask is set it falls back to standard ConvLSTM behaviour.
    """

    name: str = "convlstm_vessel_masked_input"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vessel_mask: np.ndarray | None = None

    def set_vessel_mask(self, mask: np.ndarray) -> None:
        self._vessel_mask = mask

    def _apply_mask(self, frames: np.ndarray) -> np.ndarray:
        """Zero non-vessel pixels in (T, H, W) frames."""
        if self._vessel_mask is None:
            return frames
        return frames * self._vessel_mask[np.newaxis]

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
        vessel_mask: np.ndarray | None = None,
    ) -> None:
        if vessel_mask is not None:
            self._vessel_mask = vessel_mask
        masked = [self._apply_mask(f) for f in train_frames]
        super().fit(masked, horizons)

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return super().predict(self._apply_mask(context), horizon)


class ConvLSTMVesselLoss(ConvLSTMPredictor):
    """
    ConvLSTM variant that restricts training loss to vessel pixels only.

    Identical to ConvLSTMPredictor in all hyperparameters. The vessel_mask
    must be supplied via ``set_vessel_mask`` before calling ``fit``; if no
    mask is set it falls back to standard full-image MSE.
    """

    name: str = "convlstm_vessel_loss"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vessel_mask: np.ndarray | None = None

    def set_vessel_mask(self, mask: np.ndarray) -> None:
        """Set the (H, W) boolean vessel mask used during training."""
        self._vessel_mask = mask

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
        vessel_mask: np.ndarray | None = None,
    ) -> None:
        super().fit(
            train_frames,
            horizons,
            vessel_mask=vessel_mask if vessel_mask is not None else self._vessel_mask,
        )


class ConvLSTMPCADenoised(ConvLSTMPredictor):
    """
    ConvLSTM variant that denoises frames via PCA truncation before training
    and prediction.

    A PCA basis is fitted on all training frames (flattened). Each frame is
    then projected into the K-component subspace and reconstructed, keeping
    only the dominant spatial variance and discarding high-frequency noise.
    The standard ConvLSTM is then trained on these denoised frames.

    Parameters
    ----------
    n_components : int
        Number of PCA components to retain for denoising. Default: 20.
    seed : int
        Random seed for PCA (passed through from parent; shared).
    All other parameters are inherited from ConvLSTMPredictor.
    """

    name: str = "convlstm_pca_denoised"

    def __init__(self, n_components: int = 20, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_components = n_components
        self._pca: PCA | None = None
        self._frame_shape: tuple[int, int] | None = None

    def _fit_pca(self, train_frames: list[np.ndarray]) -> None:
        self._frame_shape = train_frames[0].shape[1:]
        flattened = np.concatenate(
            [f.reshape(f.shape[0], -1) for f in train_frames], axis=0
        ).astype(np.float64)
        n_components = min(self.n_components, flattened.shape[0], flattened.shape[1])
        self._pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=self.seed,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in divide", RuntimeWarning)
            self._pca.fit(flattened)

    def _denoise(self, frames: np.ndarray) -> np.ndarray:
        """Project (T, H, W) frames through PCA truncation and reconstruct."""
        if self._pca is None or self._frame_shape is None:
            return frames
        T = frames.shape[0]
        flat = frames.reshape(T, -1).astype(np.float64)
        denoised = self._pca.inverse_transform(self._pca.transform(flat))
        return denoised.reshape(T, *self._frame_shape).astype(np.float32)

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
        vessel_mask: np.ndarray | None = None,
    ) -> None:
        self._fit_pca(train_frames)
        denoised = [self._denoise(f) for f in train_frames]
        super().fit(denoised, horizons, vessel_mask=vessel_mask)

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return super().predict(self._denoise(context), horizon)
