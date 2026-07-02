"""
videomae.py
-----------
VideoMAE-based frame predictor for fUS forecasting.

Uses a frozen (or fine-tunable) VideoMAE encoder (MCG-NJU/videomae-base) to
extract spatio-temporal patch embeddings from a sequence of past frames, then
decodes them into a predicted future frame via a lightweight spatial
convolutional head.

Architecture
~~~~~~~~~~~~
1. Boundary-pad ``lag`` frames to ``num_frames=16`` (VideoMAE-base native clip
   length) via repeat-border padding.
2. Expand grayscale to 3-channel; bilinear upsample H×W -> 224×224.
3. ``VideoMAEModel`` encoder -> ``last_hidden_state`` [B, 1568, 768].
4. Reshape + mean-pool temporal patches -> spatial grid [B, 768, 14, 14].
5. ``SpatialConvDecoder``: Conv2d + 3× ConvTranspose2d -> [B, 1, H, W].

One model is trained per horizon (direct forecasting, no rollout). Training
uses Adam with gradient-norm clipping and optionally restricts loss to vessel
pixels.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import VideoMAEModel as _VideoMAEModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal nn.Module components
# ---------------------------------------------------------------------------

class _VideoMAEEncoder(nn.Module):
    """Thin wrapper around HuggingFace VideoMAEModel.

    Parameters
    ----------
    ckpt : str
        HuggingFace model ID or local path.
    frozen : bool
        If True, encoder weights are frozen and kept in eval mode.
    """

    def __init__(self, ckpt: str, frozen: bool) -> None:
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for VideoMAEPredictor. "
                "Install with: pip install 'transformers>=4.38.0'"
            )
        print(f"Loading VideoMAE encoder from {ckpt} (may download ~344 MB on first run)...")
        self.encoder = _VideoMAEModel.from_pretrained(ckpt)
        self.hidden_size: int = self.encoder.config.hidden_size  # 768
        self._frozen = frozen
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [B, T, 3, H, W]; returns [B, num_tokens, hidden_size]."""
        out = self.encoder(pixel_values=pixel_values, bool_masked_pos=None)
        return out.last_hidden_state

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self.encoder.eval()
        return self


class _SpatialConvDecoder(nn.Module):
    """Decode spatial patch-token grid [B, hidden, 14, 14] -> [B, 1, 112, 112].

    Architecture::

        Conv2d(hidden, 64, 1)                -> [B, 64, 14, 14]
        ConvTranspose2d(64, 32, k=4, s=2, p=1) -> [B, 32, 28, 28]
        ConvTranspose2d(32, 16, k=4, s=2, p=1) -> [B, 16, 56, 56]
        ConvTranspose2d(16,  1, k=4, s=2, p=1) -> [B,  1, 112, 112]

    The final layer has no activation; data is z-scored so the target range
    is unbounded.
    """

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(hidden_size, 64, kernel_size=1)
        self.ct1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.ct2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.ct3 = nn.ConvTranspose2d(16,  1, kernel_size=4, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, hidden, 14, 14]; returns [B, 1, 112, 112]."""
        x = self.act(self.proj(tokens))
        x = self.act(self.ct1(x))
        x = self.act(self.ct2(x))
        return self.ct3(x)


class _VideoMAEForecaster(nn.Module):
    """VideoMAE encoder + spatial conv decoder for fUS frame prediction.

    Parameters
    ----------
    ckpt : str
        HuggingFace model ID or local path for the VideoMAE encoder.
    frozen : bool
        Whether to freeze the encoder.
    lag : int
        Number of context frames fed in per prediction.
    num_frames : int
        Native clip length expected by VideoMAE (16 for videomae-base).
    input_size : int
        Spatial resolution expected by VideoMAE (224 for videomae-base).
    """

    # VideoMAE-base: patch_size=16, tubelet_size=2
    _PATCH_SIZE = 16
    _TUBELET_SIZE = 2

    def __init__(
        self,
        ckpt: str,
        frozen: bool,
        lag: int,
        num_frames: int,
        input_size: int,
    ) -> None:
        super().__init__()
        self.lag = lag
        self.num_frames = num_frames
        self.input_size = input_size

        self.encoder = _VideoMAEEncoder(ckpt=ckpt, frozen=frozen)
        self.decoder = _SpatialConvDecoder(hidden_size=self.encoder.hidden_size)

        self._n_temporal = num_frames // self._TUBELET_SIZE           # 8
        self._n_spatial = (input_size // self._PATCH_SIZE) ** 2       # 196
        self._spatial_dim = input_size // self._PATCH_SIZE            # 14

    def _adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt [B, lag, H, W] -> [B, T, 3, 224, 224] for VideoMAE.

        Boundary-pads the lag axis to ``num_frames`` using border replication,
        expands grayscale to 3-channel, then bilinearly upsamples to
        ``input_size × input_size``.
        """
        B, lag, H, W = x.shape
        pad_before = (self.num_frames - lag) // 2
        pad_after = self.num_frames - lag - pad_before
        idx = torch.cat([
            torch.zeros(pad_before, dtype=torch.long, device=x.device),
            torch.arange(lag, dtype=torch.long, device=x.device),
            torch.full((pad_after,), lag - 1, dtype=torch.long, device=x.device),
        ])
        x_t = x[:, idx]                                                # [B, T, H, W]
        x_rgb = x_t.unsqueeze(2).expand(-1, -1, 3, -1, -1)            # [B, T, 3, H, W]

        Bt, T, C, _, _ = x_rgb.shape
        x_flat = x_rgb.reshape(Bt * T, C, H, W)
        x_flat = F.interpolate(
            x_flat, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )
        return x_flat.reshape(Bt, T, C, self.input_size, self.input_size)  # [B, T, 3, 224, 224]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, lag, H, W]; returns [B, 1, H, W]."""
        H, W = x.shape[-2], x.shape[-1]

        pv = self._adapt_input(x)                                      # [B, T, 3, 224, 224]
        tokens = self.encoder(pv)                                       # [B, num_tokens, hidden]

        B = tokens.shape[0]
        tokens = tokens.reshape(B, self._n_temporal, self._n_spatial, -1)
        tokens = tokens.mean(dim=1)                                    # [B, 196, hidden]
        tokens = tokens.permute(0, 2, 1).reshape(
            B, -1, self._spatial_dim, self._spatial_dim,
        )                                                               # [B, hidden, 14, 14]

        pred = self.decoder(tokens)                                    # [B, 1, 112, 112]

        if H != pred.shape[-2] or W != pred.shape[-1]:
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)

        return pred                                                     # [B, 1, H, W]


# ---------------------------------------------------------------------------
# Public predictor class
# ---------------------------------------------------------------------------

class VideoMAEPredictor:
    """
    Direct-horizon frame predictor based on a frozen VideoMAE encoder.

    A separate model is trained per horizon using sliding windows of ``lag``
    consecutive context frames as input. The VideoMAE encoder (frozen by
    default) extracts patch embeddings; a lightweight ``SpatialConvDecoder``
    maps them to the predicted frame. Training minimizes MSE with Adam and
    gradient-norm clipping.

    Parameters
    ----------
    ckpt : str
        HuggingFace model ID or local path for the VideoMAE encoder.
        Default: ``"MCG-NJU/videomae-base"``.
    frozen : bool
        If True (default), the encoder is frozen and only the decoder head
        is trained. Set to False to fine-tune end-to-end.
    lag : int
        Number of context frames fed into the model per prediction.
        Default: 10.
    num_frames : int
        Native clip length expected by VideoMAE-base. Default: 16.
    input_size : int
        Spatial resolution expected by VideoMAE-base. Default: 224.
    lr : float
        Adam learning rate. Default: 1e-4.
    batch_size : int
        Mini-batch size during training. Default: 4.
    n_epochs : int
        Number of training epochs per horizon. Default: 50.
    grad_clip_norm : float
        Maximum gradient L2 norm for clipping. Default: 1.0.
    seed : int
        Random seed for deterministic weight initialization and batching.
        Default: 0.
    """

    name: str = "videomae"

    def __init__(
        self,
        ckpt: str = "MCG-NJU/videomae-base",
        frozen: bool = True,
        lag: int = 10,
        num_frames: int = 16,
        input_size: int = 224,
        lr: float = 1e-4,
        batch_size: int = 4,
        n_epochs: int = 50,
        grad_clip_norm: float = 1.0,
        seed: int = 0,
    ) -> None:
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        if lag > num_frames:
            raise ValueError(
                f"lag ({lag}) must be <= num_frames ({num_frames}); "
                "VideoMAE cannot be padded to fewer frames than the input"
            )
        self.ckpt = ckpt
        self.frozen = frozen
        self.lag = lag
        self.num_frames = num_frames
        self.input_size = input_size
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"VideoMAEPredictor running on: {self.device}")
        self._models: dict[int, _VideoMAEForecaster] = {}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _build_windows(
        self,
        train_frames: list[np.ndarray],
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build sliding-window input/target tensors across all sessions.

        Parameters
        ----------
        train_frames : list of np.ndarray
            One array per session, each of shape ``(T, H, W)``.
        horizon : int
            Prediction horizon.

        Returns
        -------
        X : torch.Tensor, shape (N, lag, H, W)
        Y : torch.Tensor, shape (N, 1, H, W)
        """
        p = self.lag
        X_list, Y_list = [], []
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
        X_t = torch.from_numpy(X)                   # (N, p, H, W)
        Y_t = torch.from_numpy(Y).unsqueeze(1)      # (N, 1, H, W)
        return X_t, Y_t

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        train_frames: list[np.ndarray],
        horizons: list[int],
        vessel_mask: np.ndarray | None = None,
    ) -> None:
        """Train one VideoMAE model per horizon.

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

            model = _VideoMAEForecaster(
                ckpt=self.ckpt,
                frozen=self.frozen,
                lag=self.lag,
                num_frames=self.num_frames,
                input_size=self.input_size,
            ).to(self.device)

            trainable = (
                model.decoder.parameters() if self.frozen else model.parameters()
            )
            optimizer = torch.optim.Adam(trainable, lr=self.lr)

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
                        loss = F.mse_loss(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                    optimizer.step()

            model.eval()
            self._models[horizon] = model

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Predict the frame ``horizon`` steps ahead.

        Parameters
        ----------
        context : np.ndarray, shape (T, H, W)
            Recent past frames, ``T >= lag``. Only the last ``lag`` frames
            are used.
        horizon : int
            Prediction horizon. Must be a key in ``self._models``.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Predicted frame.
        """
        if horizon not in self._models:
            raise KeyError(f"horizon {horizon} was not fitted")

        model = self._models[horizon]
        lag_frames = context[-self.lag :].astype(np.float32)   # (lag, H, W)
        x = torch.from_numpy(lag_frames).unsqueeze(0)          # (1, lag, H, W)
        x = x.to(self.device)

        model.eval()
        with torch.no_grad():
            pred = model(x)                                    # (1, 1, H, W)

        return pred[0, 0].cpu().numpy().astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"VideoMAEPredictor(ckpt={self.ckpt!r}, frozen={self.frozen}, "
            f"lag={self.lag}, num_frames={self.num_frames}, "
            f"input_size={self.input_size}, lr={self.lr}, "
            f"batch_size={self.batch_size}, n_epochs={self.n_epochs}, "
            f"grad_clip_norm={self.grad_clip_norm}, seed={self.seed})"
        )


class VideoMAEVesselLoss(VideoMAEPredictor):
    """
    VideoMAE variant that restricts training loss to vessel pixels only.

    Identical to ``VideoMAEPredictor`` in all hyperparameters. The vessel mask
    must be supplied via ``set_vessel_mask`` before calling ``fit``; if no mask
    is set it falls back to standard full-image MSE.
    """

    name: str = "videomae_vessel_loss"

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
