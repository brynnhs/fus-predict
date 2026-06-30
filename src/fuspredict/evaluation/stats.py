"""
stats.py
--------
Pure statistics library for the model comparison pipeline.

Every function takes arrays or DataFrames and returns arrays, tuples, or
DataFrames. Nothing here plots, performs I/O, or loads config.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test between two equal-length samples.

    Parameters
    ----------
    a : np.ndarray
        First sample.
    b : np.ndarray
        Second sample, paired with ``a``.

    Returns
    -------
    (statistic, p_value) : tuple of float
        The Wilcoxon test statistic and two-sided p-value. Returns
        ``(nan, nan)`` if there are fewer than 2 paired samples or all
        paired differences are zero.
    """
    diff = a - b
    if len(diff) < 2 or np.all(diff == 0):
        return float("nan"), float("nan")
    result = scipy_stats.wilcoxon(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def bootstrap_median_diff_ci(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 9999, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap the median of ``a - b`` and its 95% confidence interval.

    Parameters
    ----------
    a : np.ndarray
        First sample.
    b : np.ndarray
        Second sample, paired with ``a``.
    n_resamples : int, default 9999
        Number of bootstrap resamples.
    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    (median_diff, ci_low, ci_high) : tuple of float
        The observed median of ``a - b`` and the bootstrap 95% confidence
        interval bounds. CI bounds are ``nan`` if there are fewer than 2
        paired samples.
    """
    diff = a - b
    med = float(np.median(diff))
    if len(diff) < 2:
        return med, float("nan"), float("nan")
    bs = scipy_stats.bootstrap(
        (diff,),
        statistic=np.median,
        n_resamples=n_resamples,
        confidence_level=0.95,
        method="percentile",
        random_state=seed,
    )
    return med, float(bs.confidence_interval.low), float(bs.confidence_interval.high)


def significance_stars(p: float) -> str:
    """Map a p-value to a conventional significance marker.

    Parameters
    ----------
    p : float
        Two-sided p-value.

    Returns
    -------
    str
        ``'***'`` if ``p < 0.001``, ``'**'`` if ``p < 0.01``, ``'*'`` if
        ``p < 0.05``, ``'n.s.'`` otherwise, or ``'n/a'`` if ``p`` is NaN.
    """
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _aligned(
    df: pd.DataFrame, models: list[str], horizon: int, rmse_col: str = "rmse_full"
) -> pd.DataFrame:
    """Return a wide DataFrame indexed by session_id, one column per model.

    Only includes models with non-NaN values for ``rmse_col``, and only
    sessions where all included models have values.
    """
    parts = []
    for m in models:
        s = (
            df[(df["model"] == m) & (df["horizon"] == horizon)]
            .set_index("session_id")[rmse_col]
            .rename(m)
        )
        if s.notna().any():
            parts.append(s)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1).dropna()


def compute_wilcoxon(
    df: pd.DataFrame, horizons: list[int], models: list[str]
) -> pd.DataFrame:
    """Wilcoxon signed-rank tests: each model vs zero, plus best linear vs ConvLSTM.

    Computed per horizon and per region (full / vessel / non-vessel).

    Parameters
    ----------
    df : pd.DataFrame
        Long-form results with columns ``session_id``, ``model``,
        ``horizon``, ``rmse_full``, ``rmse_vessel``, ``rmse_nonvessel``.
        Must include a ``"zero"`` model row.
    horizons : list of int
        Horizons to test.
    models : list of str
        Full set of models to include (including ``"zero"`` and
        ``"convlstm"`` if present); the non-zero subset is what's tested
        against zero.

    Returns
    -------
    pd.DataFrame
        Tidy stats table with columns ``region``, ``horizon``, ``model_A``,
        ``model_B``, ``n_sessions``, ``W``, ``p_value``, ``median_diff``,
        ``ci_low``, ``ci_high``.
    """
    model_order = [m for m in models if m != "zero"]
    linear_models = [m for m in model_order if m not in ("zero", "convlstm")]

    rows: list[dict] = []
    for h in horizons:
        for rmse_col, region_label in [
            ("rmse_full", "full"),
            ("rmse_vessel", "vessel"),
            ("rmse_nonvessel", "non_vessel"),
        ]:
            wide = _aligned(df, models, h, rmse_col=rmse_col)
            if wide.empty or "zero" not in wide.columns:
                continue

            zero_vals = wide["zero"].values.astype(float)
            for m in model_order:
                if m not in wide.columns:
                    continue
                vals = wide[m].values.astype(float)
                W, p = wilcoxon_test(vals, zero_vals)
                med, ci_lo, ci_hi = bootstrap_median_diff_ci(vals, zero_vals)
                rows.append(
                    {
                        "region": region_label,
                        "horizon": h,
                        "model_A": m,
                        "model_B": "zero",
                        "n_sessions": len(vals),
                        "W": W,
                        "p_value": p,
                        "median_diff": med,
                        "ci_low": ci_lo,
                        "ci_high": ci_hi,
                    }
                )

            linear_present = [m for m in linear_models if m in wide.columns]
            if linear_present and "convlstm" in wide.columns:
                median_rmse = {
                    m: float(np.median(wide[m].values.astype(float))) for m in linear_present
                }
                best_linear = min(median_rmse, key=median_rmse.get)
                a = wide["convlstm"].values.astype(float)
                b = wide[best_linear].values.astype(float)
                W, p = wilcoxon_test(a, b)
                med, ci_lo, ci_hi = bootstrap_median_diff_ci(a, b)
                rows.append(
                    {
                        "region": region_label,
                        "horizon": h,
                        "model_A": "convlstm",
                        "model_B": f"{best_linear} (best linear)",
                        "n_sessions": len(a),
                        "W": W,
                        "p_value": p,
                        "median_diff": med,
                        "ci_low": ci_lo,
                        "ci_high": ci_hi,
                    }
                )

    return pd.DataFrame(rows)


def residual_acf_latent(residual_latents: np.ndarray, max_lag: int) -> dict:
    """Compute per-component ACF of residuals in latent (e.g. PCA) space.

    Parameters
    ----------
    residual_latents : np.ndarray, shape (T, d)
        Residual time series for each of ``d`` latent components.
    max_lag : int
        Maximum lag to compute, in ``[0, T - 1]``.

    Returns
    -------
    dict
        ``acf``: ``(d, max_lag + 1)`` array of per-component autocorrelation.
        ``mean_abs_acf_by_lag``: ``(max_lag + 1,)`` mean absolute ACF across
        components, useful as a single residual-whiteness summary per lag.
    """
    x = np.asarray(residual_latents)
    if x.ndim != 2:
        raise ValueError("residual_latents must be (T, d)")
    T, d = x.shape
    max_lag = int(max_lag)
    if not (0 <= max_lag < T):
        raise ValueError(f"max_lag must be in [0, T-1]; got max_lag={max_lag}, T={T}")

    x_c = x - x.mean(axis=0, keepdims=True)
    denom = np.sum(x_c**2, axis=0)
    denom = np.where(denom == 0, np.nan, denom)

    acf = np.zeros((d, max_lag + 1), dtype=float)
    acf[:, 0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[:, lag] = np.sum(x_c[lag:] * x_c[:-lag], axis=0) / denom

    return {
        "acf": acf,
        "mean_abs_acf_by_lag": np.nanmean(np.abs(acf), axis=0),
    }


def ljung_box_test(residual_series: np.ndarray, lags: list[int]) -> list | None:
    """Run the Ljung-Box test on each column of a residual series.

    Requires ``statsmodels``. Returns ``None`` if not available, emitting a
    warning rather than raising, since this is a diagnostic check.

    Parameters
    ----------
    residual_series : np.ndarray, shape (T,) or (T, d)
        Residual time series. 1-D inputs are treated as a single column.
    lags : list of int
        Lags to test.

    Returns
    -------
    list of pd.DataFrame, or None
        One Ljung-Box result table per column of ``residual_series``.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        warnings.warn("statsmodels not available; skipping Ljung-Box test.")
        return None

    x = np.asarray(residual_series)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return [acorr_ljungbox(x[:, i], lags=lags, return_df=True) for i in range(x.shape[1])]
