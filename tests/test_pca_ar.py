import numpy as np
import pytest

from fuspredict.models.pca_ar._ridge import _fit_ridge_ar, _solve_ridge_ar
from fuspredict.models.pca_ar.frozen_basis import FrozenBasisAR, FrozenBasisRollingMean
from fuspredict.models.pca_ar.patch_lag import PatchLagICAAR, PatchLagPCAAR


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(60, 8, 8)).astype(np.float32) for _ in range(2)]


def test_frozen_basis_ar_fit_predict_shape(frames):
    m = FrozenBasisAR(method="pca", n_components=3, ar_lag=4, seed=0)
    m.fit(frames, horizons=[1, 2])
    pred = m.predict(frames[0][:10], horizon=1)
    assert pred.shape == (8, 8)
    assert pred.dtype == np.float32


def test_frozen_basis_ar_unfitted_horizon_raises(frames):
    m = FrozenBasisAR(method="pca", n_components=3, ar_lag=4, seed=0)
    m.fit(frames, horizons=[1])
    with pytest.raises(KeyError):
        m.predict(frames[0][:10], horizon=5)


def test_frozen_basis_ar_reconstruct_oracle_shape(frames):
    m = FrozenBasisAR(method="pca", n_components=3, ar_lag=4, seed=0)
    m.fit(frames, horizons=[1])
    recon = m.reconstruct_oracle(frames[0][0])
    assert recon.shape == (8, 8)


def test_frozen_basis_ar_name_reflects_method():
    m = FrozenBasisAR(method="ica", n_components=2, ar_lag=2)
    assert m.name == "frozen_ica_ar"


def test_frozen_basis_ar_invalid_method_raises():
    with pytest.raises(ValueError):
        FrozenBasisAR(method="bogus")


def test_frozen_basis_rolling_mean_fit_predict_shape(frames):
    m = FrozenBasisRollingMean(method="pca", n_components=3, window=4, seed=0)
    m.fit(frames, horizons=[1])
    pred = m.predict(frames[0][:10], horizon=1)
    assert pred.shape == (8, 8)


def test_frozen_basis_rolling_mean_same_prediction_regardless_of_horizon(frames):
    m = FrozenBasisRollingMean(method="pca", n_components=3, window=4, seed=0)
    m.fit(frames, horizons=[1, 3])
    pred1 = m.predict(frames[0][:10], horizon=1)
    pred3 = m.predict(frames[0][:10], horizon=3)
    np.testing.assert_array_equal(pred1, pred3)


def test_patch_lag_pca_ar_fit_predict_shape(frames):
    m = PatchLagPCAAR(patch_size=4, n_components=2, ar_lag=3, seed=0)
    m.fit(frames, horizons=[1])
    pred = m.predict(frames[0][:10], horizon=1)
    assert pred.shape == (8, 8)


def test_patch_lag_pca_ar_tile_patches_covers_full_frame():
    patches = PatchLagPCAAR.tile_patches(8, 8, 2)
    covered = np.zeros((8, 8), dtype=bool)
    for rs, cs in patches:
        covered[rs, cs] = True
    assert covered.all()


def test_patch_lag_pca_ar_invalid_patch_size_raises():
    with pytest.raises(ValueError):
        PatchLagPCAAR(patch_size=0)


def test_patch_lag_ica_ar_shares_patch_helpers_with_pca_variant():
    assert PatchLagICAAR.tile_patches is PatchLagPCAAR.tile_patches
    assert PatchLagICAAR._patch_origins is PatchLagPCAAR._patch_origins
    assert PatchLagICAAR._extract_patch is PatchLagPCAAR._extract_patch


def test_patch_lag_ica_ar_fit_predict_shape(frames):
    m = PatchLagICAAR(patch_size=4, n_components=2, ar_lag=3, seed=0)
    m.fit(frames, horizons=[1])
    pred = m.predict(frames[0][:10], horizon=1)
    assert pred.shape == (8, 8)


def test_fit_solve_ridge_ar_roundtrip():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(50, 3))
    acc = _fit_ridge_ar(Z, horizon=1, lag=2, ridge_lambda=0.1)
    W, bias = _solve_ridge_ar(acc["XtX"], acc["XtY"], ridge_lambda=0.1)
    assert W.shape == (2 * 3, 3)
    assert bias.shape == (3,)


def test_fit_ridge_ar_insufficient_samples_returns_zeros():
    Z = np.zeros((3, 2))
    acc = _fit_ridge_ar(Z, horizon=5, lag=5, ridge_lambda=0.1)
    assert np.all(acc["XtX"] == 0)
    assert np.all(acc["XtY"] == 0)
