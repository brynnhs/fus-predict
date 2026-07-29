import numpy as np
import pytest

from fuspredict.models.base import split_frames
from fuspredict.models.pixel_ar import PixelAR
from fuspredict.models.rolling_mean import RollingMeanPredictor
from fuspredict.models.zero import ZeroPredictor


# ---------------------------------------------------------------------------
# ZeroPredictor
# ---------------------------------------------------------------------------

def test_zero_predictor_predicts_zeros():
    m = ZeroPredictor()
    frames = [np.random.default_rng(0).normal(size=(20, 5, 6)).astype(np.float32)]
    m.fit(frames, horizons=[1, 3])
    pred = m.predict(frames[0][:10], horizon=1)
    assert pred.shape == (5, 6)
    assert pred.dtype == np.float32
    np.testing.assert_array_equal(pred, 0.0)


def test_zero_predictor_unfitted_horizon_raises():
    m = ZeroPredictor()
    m.fit([np.zeros((10, 3, 3), dtype=np.float32)], horizons=[1])
    with pytest.raises(KeyError):
        m.predict(np.zeros((5, 3, 3)), horizon=2)


def test_zero_predictor_empty_train_frames_raises():
    m = ZeroPredictor()
    with pytest.raises(ValueError):
        m.fit([], horizons=[1])


# ---------------------------------------------------------------------------
# RollingMeanPredictor
# ---------------------------------------------------------------------------

def test_rolling_mean_predicts_window_average():
    m = RollingMeanPredictor(window=3)
    m.fit([np.zeros((10, 2, 2), dtype=np.float32)], horizons=[1])
    context = np.stack([np.full((2, 2), v, dtype=np.float32) for v in [1, 2, 3, 6]])
    pred = m.predict(context, horizon=1)
    # last 3 frames: 2, 3, 6 -> mean 3.6667
    np.testing.assert_allclose(pred, np.full((2, 2), 11 / 3), atol=1e-5)


def test_rolling_mean_same_prediction_regardless_of_horizon():
    m = RollingMeanPredictor(window=3)
    m.fit([np.zeros((10, 2, 2), dtype=np.float32)], horizons=[1, 5])
    context = np.random.default_rng(0).normal(size=(4, 2, 2)).astype(np.float32)
    pred1 = m.predict(context, horizon=1)
    pred5 = m.predict(context, horizon=5)
    np.testing.assert_array_equal(pred1, pred5)


def test_rolling_mean_uses_fewer_frames_if_context_shorter_than_window():
    m = RollingMeanPredictor(window=10)
    m.fit([np.zeros((10, 1, 1), dtype=np.float32)], horizons=[1])
    context = np.array([[[2.0]], [[4.0]]], dtype=np.float32)
    pred = m.predict(context, horizon=1)
    np.testing.assert_allclose(pred, [[3.0]])


def test_rolling_mean_invalid_window_raises():
    with pytest.raises(ValueError):
        RollingMeanPredictor(window=0)


# ---------------------------------------------------------------------------
# PixelAR
# ---------------------------------------------------------------------------

def test_pixel_ar_fit_predict_shape():
    rng = np.random.default_rng(0)
    frames = [rng.normal(size=(40, 4, 4)).astype(np.float32)]
    m = PixelAR(lag=3, ridge_lambda=0.1)
    m.fit(frames, horizons=[1, 2])
    pred = m.predict(frames[0][:10], horizon=1)
    assert pred.shape == (4, 4)
    assert pred.dtype == np.float32


def test_pixel_ar_perfectly_predictable_signal():
    # Constant per-pixel signal: AR should learn to reproduce it closely.
    frames = [np.full((30, 2, 2), 5.0, dtype=np.float32)]
    m = PixelAR(lag=2, ridge_lambda=1e-6)
    m.fit(frames, horizons=[1])
    pred = m.predict(frames[0][:5], horizon=1)
    np.testing.assert_allclose(pred, 5.0, atol=1e-2)


def test_pixel_ar_unfitted_horizon_raises():
    frames = [np.random.default_rng(0).normal(size=(20, 3, 3)).astype(np.float32)]
    m = PixelAR(lag=2)
    m.fit(frames, horizons=[1])
    with pytest.raises(KeyError):
        m.predict(frames[0][:5], horizon=99)


def test_pixel_ar_invalid_lag_raises():
    with pytest.raises(ValueError):
        PixelAR(lag=0)


def test_pixel_ar_invalid_ridge_lambda_raises():
    with pytest.raises(ValueError):
        PixelAR(ridge_lambda=-1)


# ---------------------------------------------------------------------------
# split_frames
# ---------------------------------------------------------------------------

def test_split_frames_default_fraction():
    frames = np.zeros((100, 3, 3))
    train, test = split_frames(frames)
    assert train.shape[0] == 80
    assert test.shape[0] == 20


def test_split_frames_is_contiguous_and_temporal():
    frames = np.arange(10).reshape(10, 1, 1)
    train, test = split_frames(frames, train_frac=0.6)
    np.testing.assert_array_equal(train.ravel(), np.arange(6))
    np.testing.assert_array_equal(test.ravel(), np.arange(6, 10))


def test_split_frames_invalid_train_frac_raises():
    with pytest.raises(ValueError):
        split_frames(np.zeros((10, 3, 3)), train_frac=1.5)


def test_split_frames_invalid_ndim_raises():
    with pytest.raises(ValueError):
        split_frames(np.zeros((10, 3)))
