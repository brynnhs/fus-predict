# fus-predict

Functional ultrasound (fUS) is a neuroimaging modality that tracks brain activity through changes in cerebral blood volume, captured as a time series of 2D Power Doppler frames. Understanding how predictable those frames are — and which parts of the brain are most predictable — is a prerequisite for building stimulus-decoding or closed-loop models on top of fUS.

This repo is a frame-prediction benchmark: given a window of past Power Doppler frames from a resting-state (baseline) recording, predict the frame `N` steps ahead. It compares simple baselines (zero, rolling mean) against autoregressive models (per-pixel ridge regression, PCA-compressed ridge regression) and a ConvLSTM, with RMSE evaluated separately over vessel and parenchyma regions and Wilcoxon tests for statistical significance across sessions.

## 5-minute setup

```bash
git clone <repo-url> fus-predict
cd fus-predict
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Requires Python ≥3.11. Dependencies (numpy, scipy, xarray, netcdf4, matplotlib, pandas, pyyaml, scikit-learn, torch) are installed automatically via `pyproject.toml`.

## Pipeline at a glance

```
data/sourcedata/<subject>/Datas_*.mat, Label_pauses_*.mat   (raw .mat acquisitions)
        │
        ▼  scripts/preprocess.py
derivatives/preprocessing/<subject>/
    baseline_only_standardized/      ← z-scored (T, H, W) sessions, what models train on
    tissue_masks/                    ← vessel vs. parenchyma masks per session
        │
        ▼  scripts/run_benchmark.py
derivatives/modeling/benchmark/
    per_session_results.csv          ← RMSE per (model, session, horizon)
    aggregate_summary.csv
    predictions/*.npz                ← raw (gt, pred) arrays, if --save-predictions
        │
        ▼  scripts/report.py
derivatives/modeling/report/
    fig1..fig9_*.pdf, wilcoxon_stats.csv
```

A separate, independent script — `scripts/characterize.py` — runs signal-characterization diagnostics (autocorrelation, variance structure, R² ceiling) directly off the standardized sessions; it doesn't depend on the benchmark.

## Running it

All three pipeline stages read their parameters from [`config/config.yml`](config/config.yml) — no hardcoded paths or hyperparameters in scripts.

```bash
# 1. Raw .mat -> standardized, z-scored sessions + tissue masks
python scripts/preprocess.py

# 2. Fit and evaluate all models across all sessions
python scripts/run_benchmark.py --save-predictions

# 3. Generate comparison figures and Wilcoxon stats
python scripts/report.py

# Optional: signal-characterization diagnostics (independent of 2-3)
python scripts/characterize.py
```

Quick smoke test instead of a full run:

```bash
python scripts/run_benchmark.py --models zero,rolling_mean --n-sessions 2
python scripts/report.py --skip-spatial
```

## Code layout

```
src/fuspredict/
    project.py            repo-root discovery + config loading
    data/                 Session container + .nc loading (the only I/O for modeling)
    preprocessing/         .mat extraction → geometry → filtering → standardization → tissue masks
    models/                Predictor implementations (zero, rolling_mean, pixel_ar, pca_ar, convlstm)
                            + the Predictor protocol and train/test split (base.py)
    evaluation/
        benchmark.py        orchestrates fit/predict across sessions × models × horizons
        stats.py            pure statistics (Wilcoxon, bootstrap CIs, ACF)
        visualization.py    pure plotting, no fitting or path logic
    autocorrelation.py     pure ACF / spatial-correlation utilities (used by characterize.py)
scripts/                  the only place that touches argparse, config, and the filesystem
```

Every `Predictor` (in `models/`) implements `fit(train_frames, horizons)` / `predict(context, horizon)` (see `models/base.py`); `evaluation/benchmark.py` is the single source of truth for how train/test splits and evaluation windows are built, so RMSE comparisons across models are apples-to-apples.

## Where things live

- Models to compare: edit `ALL_MODEL_NAMES` and `build_predictor_factories` in `scripts/run_benchmark.py`.
- Hyperparameters, paths, excluded sessions: `config/config.yml`.
- Adding a new model: implement the `Predictor` protocol (`models/base.py`) in a new file under `models/`, register it in `build_predictor_factories`.
