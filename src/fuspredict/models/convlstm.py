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

import numpy as np
import torch
import torch.nn as nn


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
            windows = np.stack(
                [frames[i : i + p] for i in range(n_samples)], axis=0
            )  # (n_samples, p, H, W)
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
    ) -> None:
        """
        Train one ConvLSTM model per horizon via MSE loss and Adam.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session, each of shape ``(T, H, W)``.
        horizons : list of int
            Prediction horizons to fit.
        """
        if not train_frames:
            raise ValueError("train_frames must contain at least one session")

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
            loss_fn = nn.MSELoss()

            model.train()
            for _ in range(self.n_epochs):
                perm = torch.randperm(n_samples, generator=generator)
                for start in range(0, n_samples, self.batch_size):
                    idx = perm[start : start + self.batch_size]
                    xb = X[idx].to(self.device)
                    yb = Y[idx].to(self.device)

                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
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

    def __repr__(self) -> str:
        return (
            f"ConvLSTMPredictor(hidden_channels={self.hidden_channels}, "
            f"kernel_size={self.kernel_size}, lag={self.lag}, lr={self.lr}, "
            f"batch_size={self.batch_size}, n_epochs={self.n_epochs}, "
            f"grad_clip_norm={self.grad_clip_norm}, seed={self.seed})"
        )
