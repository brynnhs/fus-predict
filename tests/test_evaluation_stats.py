import numpy as np
import pandas as pd
import pytest

from fuspredict.evaluation.stats import (
    _aligned,
    bootstrap_median_diff_ci,
    compute_wilcoxon,
    ljung_box_test,
    residual_acf_latent,
    rmse,
    significance_stars,
    wilcoxon_test,
)


def test_rmse_scalar_reduction():
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.0, 2.0, 4.0])
    assert rmse(preds, targets) == pytest.approx(np.sqrt(1 / 3))


def test_rmse_identical_arrays_is_zero():
    a = np.random.default_rng(0).normal(size=(5, 4, 4))
    assert rmse(a, a) == pytest.approx(0.0)


def test_rmse_axis_reduction_shape():
    preds = np.zeros((3, 4, 4))
    targets = np.ones((3, 4, 4))
    per_pixel = rmse(preds, targets, axis=0)
    assert per_pixel.shape == (4, 4)
    per_frame = rmse(preds, targets, axis=(1, 2))
    assert per_frame.shape == (3,)


def test_wilcoxon_test_returns_nan_for_all_zero_diff():
    a = np.array([1.0, 2.0, 3.0])
    stat, p = wilcoxon_test(a, a)
    assert np.isnan(stat)
    assert np.isnan(p)


def test_wilcoxon_test_returns_nan_for_too_few_samples():
    stat, p = wilcoxon_test(np.array([1.0]), np.array([2.0]))
    assert np.isnan(stat)
    assert np.isnan(p)


def test_wilcoxon_test_detects_consistent_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=5, size=20)
    b = rng.normal(loc=0, size=20)
    stat, p = wilcoxon_test(a, b)
    assert p < 0.01


def test_bootstrap_median_diff_ci_matches_observed_median():
    a = np.array([5.0, 6.0, 7.0, 8.0])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    med, ci_lo, ci_hi = bootstrap_median_diff_ci(a, b, n_resamples=200, seed=0)
    assert med == pytest.approx(np.median(a - b))
    assert ci_lo <= med <= ci_hi


def test_bootstrap_median_diff_ci_too_few_samples_returns_nan_bounds():
    med, ci_lo, ci_hi = bootstrap_median_diff_ci(np.array([1.0]), np.array([2.0]))
    assert med == pytest.approx(-1.0)
    assert np.isnan(ci_lo)
    assert np.isnan(ci_hi)


@pytest.mark.parametrize(
    "p,expected",
    [
        (0.0001, "***"),
        (0.005, "**"),
        (0.02, "*"),
        (0.5, "n.s."),
        (float("nan"), "n/a"),
    ],
)
def test_significance_stars(p, expected):
    assert significance_stars(p) == expected


def _make_results_df():
    rows = []
    for session in ["s1", "s2", "s3"]:
        for model, base in [("zero", 1.0), ("pixel_ar", 0.5), ("convlstm", 0.3)]:
            rows.append({
                "session_id": session,
                "model": model,
                "horizon": 1,
                "rmse_full": base + hash((session, model)) % 100 / 1000,
                "rmse_vessel": base,
                "rmse_nonvessel": base,
            })
    return pd.DataFrame(rows)


def test_aligned_returns_wide_dataframe_with_shared_sessions():
    df = _make_results_df()
    wide = _aligned(df, ["zero", "pixel_ar"], horizon=1)
    assert set(wide.columns) == {"zero", "pixel_ar"}
    assert set(wide.index) == {"s1", "s2", "s3"}


def test_aligned_empty_when_no_matching_horizon():
    df = _make_results_df()
    wide = _aligned(df, ["zero"], horizon=999)
    assert wide.empty


def test_compute_wilcoxon_produces_rows_for_each_model_vs_zero():
    df = _make_results_df()
    stats_df = compute_wilcoxon(df, horizons=[1], models=["zero", "pixel_ar", "convlstm"])
    assert not stats_df.empty
    assert "model_A" in stats_df.columns
    pixel_ar_rows = stats_df[(stats_df["model_A"] == "pixel_ar") & (stats_df["model_B"] == "zero")]
    assert len(pixel_ar_rows) > 0


def test_residual_acf_latent_lag_zero_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    out = residual_acf_latent(x, max_lag=5)
    assert out["acf"].shape == (3, 6)
    np.testing.assert_allclose(out["acf"][:, 0], 1.0)
    assert out["mean_abs_acf_by_lag"].shape == (6,)


def test_residual_acf_latent_invalid_max_lag_raises():
    x = np.zeros((10, 2))
    with pytest.raises(ValueError):
        residual_acf_latent(x, max_lag=10)


def test_residual_acf_latent_invalid_ndim_raises():
    with pytest.raises(ValueError):
        residual_acf_latent(np.zeros(10), max_lag=2)


def test_ljung_box_test_returns_none_and_warns_when_statsmodels_missing(monkeypatch):
    import sys
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "statsmodels.stats.diagnostic" or name.startswith("statsmodels"):
            raise ImportError("simulated missing statsmodels")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.warns(UserWarning, match="statsmodels not available"):
        result = ljung_box_test(np.zeros(20), lags=[1, 2])
    assert result is None
