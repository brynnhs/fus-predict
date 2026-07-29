"""
registry.py
-----------
Shared model registry for the fUS frame-prediction benchmarks.

Single source of truth for "which models exist" and "how are they built
from the project config", so every entrypoint script (run_benchmark.py,
run_task_benchmark.py, ...) that runs the model comparison stays in sync.
"""

from __future__ import annotations

from typing import Callable

from fuspredict.models.base import Predictor
from fuspredict.models.convlstm.patch_pca import PatchPCAConvLSTM
from fuspredict.models.convlstm.predictor import ConvLSTMPredictor
from fuspredict.models.pca_ar.frozen_basis import FrozenBasisAR, FrozenBasisRollingMean
from fuspredict.models.pca_ar.patch_lag import PatchLagICAAR, PatchLagPCAAR
from fuspredict.models.pixel_ar import PixelAR
from fuspredict.models.rolling_mean import RollingMeanPredictor
from fuspredict.models.zero import ZeroPredictor

ALL_MODEL_NAMES = [
    "zero",
    "rolling_mean",
    "pixel_ar",
    "patch_lag_pca_ar",
    "frozen_pca_ar",
    "frozen_ica_ar",
    "frozen_pca_rolling_mean",
    "frozen_ica_rolling_mean",
    "convlstm",
    "convlstm_pca_latent",
    "convlstm_ica_latent",
    "patch_pca_convlstm",
]


def build_predictor_factories(modeling_cfg: dict) -> dict[str, Callable[[], Predictor]]:
    """
    Build a name -> zero-arg constructor mapping for all available models.

    Parameters
    ----------
    modeling_cfg : dict
        The ``modeling`` section of the project config, containing
        ``n_lags`` and per-model hyperparameter blocks.

    Returns
    -------
    dict of str to callable
        Maps model name to a zero-argument callable returning a freshly
        constructed :class:`~fuspredict.models.base.Predictor`.
    """
    n_lags = modeling_cfg["n_lags"]
    pixel_ar_cfg = modeling_cfg["pixel_ar"]
    patch_lag_cfg = modeling_cfg["patch_lag_pca_ar"]
    patch_lag_ica_cfg = modeling_cfg.get("patch_lag_ica_ar", patch_lag_cfg)
    frozen_cfg = modeling_cfg["frozen_basis_ar"]
    convlstm_cfg = modeling_cfg["convlstm"]
    convlstm_pca_latent_cfg = modeling_cfg.get("convlstm_pca_latent", convlstm_cfg)
    convlstm_ica_latent_cfg = modeling_cfg.get("convlstm_ica_latent", convlstm_cfg)
    patch_pca_convlstm_cfg = modeling_cfg.get("patch_pca_convlstm", convlstm_cfg)
    pca_basis_cfg = modeling_cfg["pca_basis"]

    return {
        "zero": lambda: ZeroPredictor(),
        "rolling_mean": lambda: RollingMeanPredictor(window=n_lags),
        "pixel_ar": lambda: PixelAR(
            lag=n_lags,
            ridge_lambda=pixel_ar_cfg["ridge_lambda"],
        ),
        "patch_lag_pca_ar": lambda: PatchLagPCAAR(
            patch_size=patch_lag_cfg["patch_size"],
            n_components=patch_lag_cfg["n_components"],
            ar_lag=n_lags,
            ridge_lambda=patch_lag_cfg["ridge_lambda"],
            seed=pca_basis_cfg["seed"],
        ),
        "patch_lag_ica_ar": lambda: PatchLagICAAR(
            patch_size=patch_lag_ica_cfg["patch_size"],
            n_components=patch_lag_ica_cfg["n_components"],
            ar_lag=n_lags,
            ridge_lambda=patch_lag_ica_cfg["ridge_lambda"],
            ica_algorithm=patch_lag_ica_cfg.get("ica_algorithm"),
            seed=pca_basis_cfg["seed"],
        ),
        "frozen_pca_ar": lambda: FrozenBasisAR(
            method="pca",
            n_components=frozen_cfg["n_components"],
            ar_lag=n_lags,
            ridge_lambda=frozen_cfg["ridge_lambda"],
            seed=pca_basis_cfg["seed"],
        ),
        "frozen_ica_ar": lambda: FrozenBasisAR(
            method="ica",
            n_components=frozen_cfg["n_components"],
            ar_lag=n_lags,
            ridge_lambda=frozen_cfg["ridge_lambda"],
            ica_algorithm=frozen_cfg.get("ica_algorithm"),
            seed=pca_basis_cfg["seed"],
        ),
        "frozen_pca_rolling_mean": lambda: FrozenBasisRollingMean(
            method="pca",
            n_components=frozen_cfg["n_components"],
            window=n_lags,
            seed=pca_basis_cfg["seed"],
        ),
        "frozen_ica_rolling_mean": lambda: FrozenBasisRollingMean(
            method="ica",
            n_components=frozen_cfg["n_components"],
            window=n_lags,
            ica_algorithm=frozen_cfg.get("ica_algorithm"),
            seed=pca_basis_cfg["seed"],
        ),
        "convlstm": lambda: ConvLSTMPredictor(
            hidden_channels=convlstm_cfg["hidden_channels"],
            kernel_size=convlstm_cfg["kernel_size"],
            lag=n_lags,
            lr=convlstm_cfg["learning_rate"],
            batch_size=convlstm_cfg["batch_size"],
            n_epochs=convlstm_cfg["n_epochs"],
            grad_clip_norm=convlstm_cfg["grad_clip_norm"],
            seed=pca_basis_cfg["seed"],
        ),
        "convlstm_pca_latent": lambda: ConvLSTMPredictor(
            hidden_channels=convlstm_pca_latent_cfg["hidden_channels"],
            lag=n_lags,
            lr=convlstm_pca_latent_cfg["learning_rate"],
            batch_size=convlstm_pca_latent_cfg["batch_size"],
            n_epochs=convlstm_pca_latent_cfg["n_epochs"],
            grad_clip_norm=convlstm_pca_latent_cfg["grad_clip_norm"],
            input_mode="pca",
            n_components=convlstm_pca_latent_cfg.get("n_components", 12),
            seed=pca_basis_cfg["seed"],
        ),
        "convlstm_ica_latent": lambda: ConvLSTMPredictor(
            hidden_channels=convlstm_ica_latent_cfg["hidden_channels"],
            lag=n_lags,
            lr=convlstm_ica_latent_cfg["learning_rate"],
            batch_size=convlstm_ica_latent_cfg["batch_size"],
            n_epochs=convlstm_ica_latent_cfg["n_epochs"],
            grad_clip_norm=convlstm_ica_latent_cfg["grad_clip_norm"],
            input_mode="ica",
            n_components=convlstm_ica_latent_cfg.get("n_components", 12),
            ica_algorithm=convlstm_ica_latent_cfg.get("ica_algorithm"),
            seed=pca_basis_cfg["seed"],
        ),
        "patch_pca_convlstm": lambda: PatchPCAConvLSTM(
            patch_size=patch_pca_convlstm_cfg.get(
                "patch_size", patch_lag_cfg["patch_size"]
            ),
            n_components=patch_pca_convlstm_cfg.get("n_components", 6),
            hidden_channels=patch_pca_convlstm_cfg["hidden_channels"],
            kernel_size=patch_pca_convlstm_cfg.get("kernel_size", 3),
            lag=n_lags,
            lr=patch_pca_convlstm_cfg["learning_rate"],
            batch_size=patch_pca_convlstm_cfg["batch_size"],
            n_epochs=patch_pca_convlstm_cfg["n_epochs"],
            grad_clip_norm=patch_pca_convlstm_cfg["grad_clip_norm"],
            seed=pca_basis_cfg["seed"],
        ),
    }
