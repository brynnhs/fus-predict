"""
cell.py
-------
Core ConvLSTM architecture: the recurrent cell and the sequence-encoder +
1x1-readout forecaster used by both :class:`ConvLSTMPredictor` and
:class:`PatchPCAConvLSTM`.
"""

from __future__ import annotations

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
    channels : int
        Number of input/output channels (1 for raw pixel frames, or
        ``n_components`` when operating in PCA/ICA latent space).
    """

    def __init__(
        self, hidden_channels: int, kernel_size: int, channels: int = 1
    ) -> None:
        """
        Initialize the forecaster.

        Parameters
        ----------
        hidden_channels : int
            Number of hidden/cell state channels.
        kernel_size : int
            Size of the (square) ConvLSTM kernel.
        channels : int
            Number of input/output channels. Default: 1.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(
            in_channels=channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.readout = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Run the ConvLSTM over a sequence and predict the next frame.

        Parameters
        ----------
        x_seq : torch.Tensor, shape (B, T, C, H, W)
            Input sequence of context frames/latent codes.

        Returns
        -------
        torch.Tensor, shape (B, C, H, W)
            Predicted frame/latent code.
        """
        B, T, _, H, W = x_seq.shape
        device = x_seq.device
        h = torch.zeros(B, self.hidden_channels, H, W, device=device)
        c = torch.zeros(B, self.hidden_channels, H, W, device=device)
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c)
        return self.readout(h)
