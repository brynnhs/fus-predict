import numpy as np
import pytest
import torch

from fuspredict.models.convlstm.cell import ConvLSTMCell, _ConvLSTMForecaster
from fuspredict.models.convlstm.patch_pca import PatchPCAConvLSTM
from fuspredict.models.convlstm.predictor import ConvLSTMPredictor


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(30, 8, 8)).astype(np.float32) for _ in range(2)]


def _tiny_predictor(**kwargs):
    defaults = dict(hidden_channels=4, kernel_size=3, lag=3, n_epochs=1, batch_size=8, seed=0)
    defaults.update(kwargs)
    return ConvLSTMPredictor(**defaults)


def test_convlstm_cell_forward_shapes():
    cell = ConvLSTMCell(in_channels=1, hidden_channels=4, kernel_size=3)
    x = torch.zeros(2, 1, 5, 5)
    h = torch.zeros(2, 4, 5, 5)
    c = torch.zeros(2, 4, 5, 5)
    h_new, c_new = cell(x, h, c)
    assert h_new.shape == (2, 4, 5, 5)
    assert c_new.shape == (2, 4, 5, 5)


def test_forecaster_forward_shape():
    model = _ConvLSTMForecaster(hidden_channels=4, kernel_size=3, channels=1)
    x_seq = torch.zeros(2, 3, 1, 5, 5)
    out = model(x_seq)
    assert out.shape == (2, 1, 5, 5)


def test_convlstm_predictor_frames_mode_fit_predict_shape(frames):
    m = _tiny_predictor()
    m.fit(frames, horizons=[1])
    pred = m.predict(frames[0][:5], horizon=1)
    assert pred.shape == (8, 8)
    assert pred.dtype == np.float32


def test_convlstm_predictor_pca_mode_fit_predict_shape(frames):
    m = _tiny_predictor(input_mode="pca", n_components=3, kernel_size=3)
    m.fit(frames, horizons=[1])
    pred = m.predict(frames[0][:5], horizon=1)
    assert pred.shape == (8, 8)


def test_convlstm_predictor_name_reflects_input_mode():
    assert _tiny_predictor(input_mode="frames").name == "convlstm"
    assert _tiny_predictor(input_mode="pca").name == "convlstm_pca_latent"
    assert _tiny_predictor(input_mode="ica").name == "convlstm_ica_latent"


def test_convlstm_predictor_invalid_input_mode_raises():
    with pytest.raises(ValueError):
        _tiny_predictor(input_mode="bogus")


def test_convlstm_predictor_unfitted_horizon_raises(frames):
    m = _tiny_predictor()
    m.fit(frames, horizons=[1])
    with pytest.raises(KeyError):
        m.predict(frames[0][:5], horizon=99)


def test_convlstm_predictor_reconstruct_oracle_raises_in_frames_mode(frames):
    m = _tiny_predictor(input_mode="frames")
    m.fit(frames, horizons=[1])
    with pytest.raises(AttributeError):
        m.reconstruct_oracle(frames[0][0])


def test_convlstm_predictor_reconstruct_oracle_works_in_pca_mode(frames):
    m = _tiny_predictor(input_mode="pca", n_components=3)
    m.fit(frames, horizons=[1])
    recon = m.reconstruct_oracle(frames[0][0])
    assert recon.shape == (8, 8)


def test_convlstm_predictor_predict_batch_matches_predict(frames):
    m = _tiny_predictor()
    m.fit(frames, horizons=[1])
    # predict_batch expects contexts already trimmed to exactly `lag` frames,
    # unlike predict() which trims internally via context[-self.lag:].
    context = frames[0][:m.lag]
    single = m.predict(context, horizon=1)
    batch = m.predict_batch(context[np.newaxis], horizon=1)
    assert batch.shape == (1, 8, 8)
    np.testing.assert_allclose(single, batch[0], atol=1e-5)


def test_patch_pca_convlstm_fit_predict_shape(frames):
    m = PatchPCAConvLSTM(patch_size=4, n_components=2, hidden_channels=4, lag=3, n_epochs=1, batch_size=8, seed=0)
    m.fit(frames, horizons=[1])
    pred = m.predict(frames[0][:5], horizon=1)
    assert pred.shape == (8, 8)


def test_patch_pca_convlstm_reconstruct_oracle_shape(frames):
    m = PatchPCAConvLSTM(patch_size=4, n_components=2, hidden_channels=4, lag=3, n_epochs=1, batch_size=8, seed=0)
    m.fit(frames, horizons=[1])
    recon = m.reconstruct_oracle(frames[0][0])
    assert recon.shape == (8, 8)


def test_patch_pca_convlstm_invalid_patch_size_raises():
    with pytest.raises(ValueError):
        PatchPCAConvLSTM(patch_size=0)
