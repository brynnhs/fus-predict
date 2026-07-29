import numpy as np
import pandas as pd
import pytest

from fuspredict.evaluation.benchmark import _rmse, aggregate_results, build_eval_windows


# ---------------------------------------------------------------------------
# build_eval_windows
# ---------------------------------------------------------------------------

def test_build_eval_windows_count_and_shapes():
    frames = np.arange(20 * 3 * 3, dtype=np.float32).reshape(20, 3, 3)
    windows = build_eval_windows(frames, lag=5, horizon=2)
    # valid t in range(5, 20 - 2 + 1) = range(5, 19) -> 14 windows
    assert len(windows) == 14
    context, target = windows[0]
    assert context.shape == (5, 3, 3)
    assert target.shape == (3, 3)


def test_build_eval_windows_context_and_target_alignment():
    frames = np.arange(10 * 2 * 2, dtype=np.float32).reshape(10, 2, 2)
    windows = build_eval_windows(frames, lag=3, horizon=1)
    context0, target0 = windows[0]
    np.testing.assert_array_equal(context0, frames[0:3])
    np.testing.assert_array_equal(target0, frames[3])


def test_build_eval_windows_empty_when_too_short():
    frames = np.zeros((5, 2, 2), dtype=np.float32)
    windows = build_eval_windows(frames, lag=10, horizon=1)
    assert windows == []


# ---------------------------------------------------------------------------
# _rmse
# ---------------------------------------------------------------------------

def test_rmse_no_mask():
    preds = np.zeros((3, 2, 2))
    targets = np.ones((3, 2, 2))
    assert _rmse(preds, targets) == pytest.approx(1.0)


def test_rmse_with_mask_selects_subset():
    preds = np.zeros((2, 2, 2))
    targets = np.zeros((2, 2, 2))
    targets[:, 0, 0] = 10.0  # only unmasked pixel differs
    mask = np.array([[False, False], [False, True]])
    result = _rmse(preds, targets, mask=mask)
    assert result == pytest.approx(0.0)


def test_rmse_empty_mask_returns_nan():
    preds = np.zeros((2, 2, 2))
    targets = np.zeros((2, 2, 2))
    mask = np.zeros((2, 2), dtype=bool)
    result = _rmse(preds, targets, mask=mask)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------

def _per_session_df():
    rows = []
    for session in ["s1", "s2", "s3"]:
        for model in ["zero", "pixel_ar"]:
            rows.append({
                "session_id": session,
                "model": model,
                "horizon": 1,
                "rmse_full": 1.0 if model == "zero" else 0.5,
                "rmse_vessel": 1.0,
                "rmse_nonvessel": 1.0,
                "skill_vs_zero": 0.0 if model == "zero" else 0.5,
                "rmse_oracle_full": np.nan,
                "rmse_oracle_vessel": np.nan,
                "rmse_oracle_nonvessel": np.nan,
            })
    return pd.DataFrame(rows)


def test_aggregate_results_groups_by_model_and_horizon():
    df = _per_session_df()
    agg = aggregate_results(df)
    assert set(agg["model"]) == {"zero", "pixel_ar"}
    pixel_row = agg[agg["model"] == "pixel_ar"].iloc[0]
    assert pixel_row["rmse_mean"] == pytest.approx(0.5)
    assert pixel_row["n_sessions"] == 3


def test_aggregate_results_includes_kernel_size_when_present():
    df = _per_session_df()
    df["kernel_size"] = 3
    agg = aggregate_results(df)
    assert "kernel_size" in agg.columns
    assert (agg["kernel_size"] == 3).all()


def test_aggregate_results_n_sessions_reflects_partial_data():
    df = _per_session_df()
    # Drop pixel_ar's row for s3 -> only 2 sessions for that model.
    df = df[~((df["model"] == "pixel_ar") & (df["session_id"] == "s3"))]
    agg = aggregate_results(df)
    pixel_row = agg[agg["model"] == "pixel_ar"].iloc[0]
    assert pixel_row["n_sessions"] == 2
