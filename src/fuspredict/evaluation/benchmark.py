"""
benchmark.py
------------
Run and compare multiple Predictor models across multiple Session objects.

This module is the single source of truth for the evaluation contract: it
builds the (context, target) window pairs once per session/horizon and
reuses them for every model, so RMSE comparisons across models are
apples-to-apples. It contains no plotting, config loading, or path handling
beyond optional results checkpointing — it only orchestrates Session +
Predictor + metrics.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from fuspredict.data.session import Session
from fuspredict.evaluation.stats import rmse
from fuspredict.models.base import Predictor, split_frames
from fuspredict.models.convlstm import ConvLSTMPredictor, ConvLSTMVesselLoss, ConvLSTMVesselMaskedInput



# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def build_eval_windows(
    frames_test: np.ndarray,
    lag: int,
    horizon: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build the fixed set of (context, target) evaluation window pairs.

    Built once per session/horizon and reused across all models so that
    every predictor is scored on exactly the same windows.

    Parameters
    ----------
    frames_test : np.ndarray, shape (T, H, W)
        Z-scored test-portion frames for one session.
    lag : int
        Number of context frames preceding each prediction point.
    horizon : int
        Number of steps ahead to predict.

    Returns
    -------
    list of (context, target)
        ``context`` has shape ``(lag, H, W)``; ``target`` has shape ``(H, W)``
        and equals ``frames_test[t + horizon - 1]``, for every valid ``t``
        such that ``context = frames_test[t - lag:t]`` is fully in-bounds.
    """
    T = frames_test.shape[0]
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for t in range(lag, T - horizon + 1):
        context = frames_test[t - lag:t]
        target = frames_test[t + horizon - 1]
        windows.append((context, target))
    return windows


# ---------------------------------------------------------------------------
# RMSE helpers
# ---------------------------------------------------------------------------

def _rmse(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Compute RMSE between stacked predictions and targets, optionally masked.

    Parameters
    ----------
    preds, targets : np.ndarray, shape (N, H, W)
        Stacked predicted and true frames.
    mask : np.ndarray or None, shape (H, W)
        Boolean spatial mask. If provided, only pixels where ``mask`` is
        True are included. If the mask selects no pixels, returns NaN.

    Returns
    -------
    float
        RMSE, or NaN if there are no pixels to score.
    """
    if mask is not None:
        preds = preds[:, mask]
        targets = targets[:, mask]
        if preds.size == 0:
            return float("nan")
    return float(rmse(preds, targets))


# ---------------------------------------------------------------------------
# Per-predictor evaluation
# ---------------------------------------------------------------------------

def evaluate_predictor(
    predictor: Predictor,
    sessions: list[Session],
    lag: int,
    horizons: list[int],
    train_frac: float = 0.8,
    predictions_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Evaluate one predictor across all sessions and horizons.

    Each session gets its own freshly fitted model instance state (the
    predictor's ``fit`` is called once per session on that session's train
    split), matching a within-session evaluation methodology rather than
    cross-session pretraining. Sessions that fail during fit or prediction
    are skipped with a warning rather than aborting the whole run.

    Parameters
    ----------
    predictor : Predictor
        Model to evaluate.
    sessions : list of Session
        Sessions to evaluate on.
    lag : int
        Context window length passed to :func:`build_eval_windows`.
    horizons : list of int
        Prediction horizons to evaluate.
    train_frac : float
        Fraction of each session's frames used for training. Default 0.8.
    predictions_dir : Path or None
        If provided, raw ground-truth/prediction stacks for every
        ``(session, horizon)`` pair are saved as
        ``{predictions_dir}/{predictor.name}_{session.id}_h{horizon}.npz``
        with arrays ``gt`` and ``pred``, both shape ``(N, H, W)`` float32.
        The directory is created if it doesn't exist. If None, nothing is
        written to disk.

    Returns
    -------
    pd.DataFrame
        Long-form results with columns ``session_id``, ``model``,
        ``horizon``, ``rmse_full``, ``rmse_vessel``, ``rmse_nonvessel``,
        ``skill_vs_zero``.
    """
    if predictions_dir is not None:
        Path(predictions_dir).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for session in sessions:
        try:
            train, test = split_frames(session.frames, train_frac=train_frac)
            if isinstance(predictor, (ConvLSTMVesselLoss, ConvLSTMVesselMaskedInput)) and session.vessel_mask is not None:
                predictor.set_vessel_mask(session.vessel_mask)
            predictor.fit(train_frames=[train], horizons=horizons)

            for horizon in horizons:
                windows = build_eval_windows(test, lag=lag, horizon=horizon)
                if not windows:
                    continue

                targets = np.stack([target for _, target in windows])
                if isinstance(predictor, ConvLSTMPredictor):
                    contexts = np.stack([ctx for ctx, _ in windows])  # (N, lag, H, W)
                    preds = predictor.predict_batch(contexts, horizon)
                else:
                    preds = np.stack(
                        [predictor.predict(context, horizon) for context, _ in windows]
                    )
                zero_preds = np.zeros_like(targets)

                rmse_full = _rmse(preds, targets)
                rmse_zero = _rmse(zero_preds, targets)
                skill_vs_zero = (
                    1.0 - rmse_full / rmse_zero if rmse_zero != 0 else float("nan")
                )

                if predictions_dir is not None:
                    np.savez(
                        Path(predictions_dir) / f"{predictor.name}_{session.id}_h{horizon}.npz",
                        gt=targets.astype(np.float32),
                        pred=preds.astype(np.float32),
                    )

                rmse_vessel = float("nan")
                rmse_nonvessel = float("nan")
                if session.vessel_mask is not None:
                    rmse_vessel = _rmse(preds, targets, mask=session.vessel_mask)
                    rmse_nonvessel = _rmse(preds, targets, mask=~session.vessel_mask)

                rows.append(
                    {
                        "session_id": session.id,
                        "model": predictor.name,
                        "horizon": horizon,
                        "rmse_full": rmse_full,
                        "rmse_vessel": rmse_vessel,
                        "rmse_nonvessel": rmse_nonvessel,
                        "skill_vs_zero": skill_vs_zero,
                    }
                )

        except Exception as exc:
            warnings.warn(
                f"Skipping session {session.id} for model {predictor.name}: {exc}",
                stacklevel=2,
            )
            continue

    return pd.DataFrame(
        rows,
        columns=[
            "session_id",
            "model",
            "horizon",
            "rmse_full",
            "rmse_vessel",
            "rmse_nonvessel",
            "skill_vs_zero",
        ],
    )


# ---------------------------------------------------------------------------
# Multi-predictor benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    predictors: list[Predictor],
    sessions: list[Session],
    lag: int,
    horizons: list[int],
    train_frac: float = 0.8,
    results_path: Path | None = None,
    predictions_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Evaluate multiple predictors across multiple sessions and combine results.

    Supports resuming a long run: if ``results_path`` points to an existing
    file, already-computed ``(model, session_id)`` pairs are skipped, and
    results are appended and persisted to disk after each session so an
    interrupted run can resume without recomputation.

    Parameters
    ----------
    predictors : list of Predictor
        Models to evaluate.
    sessions : list of Session
        Sessions to evaluate on.
    lag : int
        Context window length.
    horizons : list of int
        Prediction horizons to evaluate.
    train_frac : float
        Fraction of each session's frames used for training. Default 0.8.
    results_path : Path or None
        If provided, used as a checkpoint file: loaded at the start (if it
        exists) to skip already-computed work, and rewritten after each
        session completes.
    predictions_dir : Path or None
        If provided, forwarded to :func:`evaluate_predictor` so raw
        ground-truth/prediction arrays are saved per ``(session, horizon)``.
        Opt-in; if None, no raw predictions are written to disk.

    Returns
    -------
    pd.DataFrame
        Combined long-form results across all predictors and sessions.
    """
    columns = [
        "session_id",
        "model",
        "horizon",
        "rmse_full",
        "rmse_vessel",
        "rmse_nonvessel",
        "skill_vs_zero",
    ]

    if results_path is not None and Path(results_path).exists():
        existing = pd.read_csv(results_path)
    else:
        existing = pd.DataFrame(columns=columns)

    done_pairs = set(zip(existing["model"], existing["session_id"]))
    all_results = [existing] if not existing.empty else []

    for i, predictor in enumerate(predictors):
        print(f"[{i+1}/{len(predictors)}] {predictor.name}")

        pending_sessions = [
            s for s in sessions if (predictor.name, s.id) not in done_pairs
        ]

        n_pending = len(pending_sessions)
        for s_idx, session in enumerate(pending_sessions):
            print(f"  [{s_idx+1}/{n_pending}] {session.id}", flush=True)
            session_df = evaluate_predictor(
                predictor,
                [session],
                lag=lag,
                horizons=horizons,
                train_frac=train_frac,
                predictions_dir=predictions_dir,
            )
            if not session_df.empty:
                all_results.append(session_df)

            if results_path is not None:
                combined_so_far = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(columns=columns)
                combined_so_far.to_csv(results_path, index=False)

    if not all_results:
        return pd.DataFrame(columns=columns)

    return pd.concat(all_results, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(per_session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-session results into per-(model, horizon) summary stats.

    Parameters
    ----------
    per_session_df : pd.DataFrame
        Long-form results as returned by :func:`run_benchmark` or
        :func:`evaluate_predictor`.

    Returns
    -------
    pd.DataFrame
        One row per ``(model, horizon)`` with columns ``model``, ``horizon``,
        ``rmse_mean``, ``rmse_std``, ``n_sessions``, ``skill_vs_zero_mean``.
        ``n_sessions`` reflects the actual number of sessions present for
        that group, so partial failures (e.g. 3 of 27 sessions skipped for
        a given model) are visible rather than hidden in the average.
    """
    grouped = per_session_df.groupby(["model", "horizon"], as_index=False).agg(
        rmse_mean=("rmse_full", "mean"),
        rmse_std=("rmse_full", "std"),
        n_sessions=("session_id", "nunique"),
        skill_vs_zero_mean=("skill_vs_zero", "mean"),
    )
    return grouped
