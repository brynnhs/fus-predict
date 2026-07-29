import numpy as np
import pytest
import xarray as xr

from fuspredict.preprocessing.filters import (
    apply_optional_filters,
    filter_reoriented_sessions,
    high_pass_filter,
    low_pass_filter,
    percentile_clip,
)


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    return rng.normal(size=(60, 4, 4)).astype(np.float32)


def test_low_pass_filter_preserves_shape(frames):
    out = low_pass_filter(frames, fps=2.5, cutoff_hz=0.5, order=4)
    assert out.shape == frames.shape
    assert out.dtype == np.float32


def test_low_pass_filter_invalid_fps_raises(frames):
    with pytest.raises(ValueError):
        low_pass_filter(frames, fps=0, cutoff_hz=0.5)


def test_low_pass_filter_zero_cutoff_raises(frames):
    with pytest.raises(ValueError):
        low_pass_filter(frames, fps=2.5, cutoff_hz=0.0)


def test_high_pass_filter_preserves_shape(frames):
    out = high_pass_filter(frames, fps=2.5, cutoff_hz=0.01, order=3)
    assert out.shape == frames.shape


def test_percentile_clip_bounds_values():
    frames = np.array([[[0.0, 100.0], [50.0, -100.0]]], dtype=np.float32)
    lo = float(np.percentile(frames, 25.0))
    hi = float(np.percentile(frames, 75.0))
    out = percentile_clip(frames, bottom=25.0, top=75.0)
    assert out.max() <= hi
    assert out.min() >= lo


def test_percentile_clip_invalid_bounds_raises():
    with pytest.raises(ValueError):
        percentile_clip(np.zeros((1, 2, 2)), bottom=90, top=10)


def test_apply_optional_filters_noop_when_all_disabled(frames):
    da = xr.DataArray(frames, dims=["time", "x", "y"], attrs={"session_id": "s1", "frame_rate": 2.5})
    out = apply_optional_filters(da)
    np.testing.assert_array_equal(out.values, frames)
    assert out.attrs["did_lowpass"] == "False"
    assert out.attrs["did_highpass"] == "False"
    assert out.attrs["did_clip"] == "False"


def test_apply_optional_filters_records_provenance(frames):
    da = xr.DataArray(frames, dims=["time", "x", "y"], attrs={"session_id": "s1", "frame_rate": 2.5})
    out = apply_optional_filters(da, enable_lowpass=True, lowpass_cutoff_hz=0.5, lowpass_order=4)
    assert out.attrs["did_lowpass"] == "True"
    assert out.attrs["lowpass_cutoff_hz"] == 0.5
    assert out.attrs["stage"] == "filtered"


def test_apply_optional_filters_missing_fps_raises(frames):
    da = xr.DataArray(frames, dims=["time", "x", "y"], attrs={"session_id": "s1"})
    with pytest.raises(ValueError):
        apply_optional_filters(da, enable_lowpass=True)


def test_filter_reoriented_sessions_writes_output(tmp_path, frames):
    da = xr.DataArray(
        frames, dims=["time", "x", "y"],
        attrs={"session_id": "s1", "frame_rate": 2.5, "stage": "reoriented_resized"},
    )
    in_path = tmp_path / "baseline_s1_reoriented_resized.nc"
    da.to_netcdf(in_path)

    out_dir = tmp_path / "out"
    outputs = filter_reoriented_sessions(
        [str(in_path)], str(out_dir),
        enable_lowpass=True, lowpass_cutoff_hz=0.5, lowpass_order=4,
        overwrite=True,
    )
    assert len(outputs) == 1
    result = xr.open_dataarray(outputs[0])
    assert result.attrs["did_lowpass"] == "True"
    assert result.attrs["stage"] == "filtered"
    assert result.shape == frames.shape


def test_filter_reoriented_sessions_skips_existing_when_not_overwrite(tmp_path, frames):
    da = xr.DataArray(
        frames, dims=["time", "x", "y"],
        attrs={"session_id": "s1", "frame_rate": 2.5, "stage": "reoriented_resized"},
    )
    in_path = tmp_path / "baseline_s1_reoriented_resized.nc"
    da.to_netcdf(in_path)
    out_dir = tmp_path / "out"

    first = filter_reoriented_sessions([str(in_path)], str(out_dir), enable_lowpass=True, overwrite=True)
    second = filter_reoriented_sessions([str(in_path)], str(out_dir), enable_lowpass=True, overwrite=False)
    assert first == second
