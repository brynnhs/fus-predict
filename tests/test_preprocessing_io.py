import numpy as np
import pytest

from fuspredict.preprocessing.io_common import (
    derive_session_id_from_path,
    mismatch,
    sanitize_attrs,
    spatial_mean_filter_frames,
)
from fuspredict.preprocessing.io_mouse import extract_baseline_mask_from_timing


def test_sanitize_attrs_converts_bool_none_list():
    attrs = {"a": True, "b": False, "c": None, "d": [1, 2, 3], "e": "keep", "f": 1.5}
    out = sanitize_attrs(attrs)
    assert out == {"a": "True", "b": "False", "c": "none", "d": "1,2,3", "e": "keep", "f": 1.5}


def test_derive_session_id_from_path_strips_baseline_prefix_and_stage_suffix():
    assert derive_session_id_from_path("baseline_Se01092020_baseline_extracted.nc") == "Se01092020"


def test_derive_session_id_from_path_no_stage_suffix():
    assert derive_session_id_from_path("baseline_Se01092020.nc") == "Se01092020"


def test_derive_session_id_from_path_no_baseline_prefix():
    assert derive_session_id_from_path("task_Se01092020_task_extracted.nc") == "task_Se01092020"


def test_mismatch_trims_to_shortest():
    images = np.zeros((10, 4, 4))
    labels = np.zeros(7)
    out_images, out_labels = mismatch(images, labels)
    assert out_images.shape[0] == 7
    assert out_labels.shape[0] == 7


def test_mismatch_equal_length_passthrough():
    images = np.zeros((5, 4, 4))
    labels = np.zeros(5)
    out_images, out_labels = mismatch(images, labels)
    assert out_images.shape[0] == 5
    assert out_labels.shape[0] == 5


def test_spatial_mean_filter_frames_kernel_1_is_copy():
    frames = np.random.rand(3, 5, 5).astype(np.float32)
    out = spatial_mean_filter_frames(frames, kernel_size=1)
    np.testing.assert_array_equal(out, frames)
    assert out is not frames


def test_spatial_mean_filter_frames_smooths():
    frames = np.zeros((1, 5, 5), dtype=np.float32)
    frames[0, 2, 2] = 25.0
    out = spatial_mean_filter_frames(frames, kernel_size=3)
    # center of a 3x3 mean kernel over a single spike of 25 -> 25/9
    assert out[0, 2, 2] == pytest.approx(25.0 / 9, rel=1e-5)


def test_spatial_mean_filter_frames_rejects_bad_ndim():
    with pytest.raises(ValueError):
        spatial_mean_filter_frames(np.zeros((5, 5)), kernel_size=3)


def test_spatial_mean_filter_frames_rejects_bad_kernel_size():
    with pytest.raises(ValueError):
        spatial_mean_filter_frames(np.zeros((1, 5, 5)), kernel_size=0)


def test_extract_baseline_mask_from_timing_excludes_stimulus_windows():
    frame_times = np.arange(0, 20, 1.0)  # 0..19s
    timing = {"baseline_s": 5.0, "stim_on_s": 2.0, "stim_off_s": 3.0, "n_trials": 2}
    mask = extract_baseline_mask_from_timing(frame_times, timing)
    # trial 0: onset=5, offset=7 -> frames 5,6 excluded
    # trial 1: onset=10, offset=12 -> frames 10,11 excluded
    assert not mask[5] and not mask[6]
    assert not mask[10] and not mask[11]
    assert mask[0] and mask[4] and mask[7] and mask[9] and mask[12] and mask[19]
