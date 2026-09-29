"""Tests for the quantile LightGBM wrapper: parameter hygiene and basic
sanity on synthetic data (fast; not a performance test)."""

import numpy as np
import pandas as pd
from gridcast.models.lgbm import LGBM_PARAMS
from gridcast.models.ml_features import (
    CALENDAR_COLS,
    EXTERNAL_FORECAST_COL,
    TARGET_COL,
    WEATHER_COLS,
    build_design_matrix,
    daily_origins,
)
from gridcast.models.quantile_lgbm import fit_predict_fold_quantiles, quantile_params


def test_quantile_params_copy_and_do_not_mutate_phase2():
    before = dict(LGBM_PARAMS)
    p = quantile_params(0.9)
    assert p["objective"] == "quantile" and p["alpha"] == 0.9
    assert LGBM_PARAMS == before  # Phase 2 L1 model untouched
    assert LGBM_PARAMS["objective"] == "l1"


def _frame(days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="1h", tz="UTC", name="timestamp")
    df = pd.DataFrame(index=idx)
    for col in CALENDAR_COLS + WEATHER_COLS + [EXTERNAL_FORECAST_COL]:
        df[col] = 0.0
    df["hour"] = idx.hour
    hours = np.arange(len(idx))
    df[TARGET_COL] = 12000 + 1500 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 400, len(idx))
    return df


def test_fit_predict_fold_quantiles_shapes_and_order():
    df = _frame()
    matrix = build_design_matrix(df, daily_origins(df.index))
    origin = pd.Timestamp("2025-04-20", tz="UTC")
    preds, iters = fit_predict_fold_quantiles(matrix, origin)
    assert set(preds) == {"q10", "q50", "q90"}
    assert all(len(v) == 24 and np.isfinite(v).all() for v in preds.values())
    assert all(i > 0 for i in iters.values())
    # with N(0, 400) noise the quantiles must separate on average
    assert preds["q10"].mean() < preds["q50"].mean() < preds["q90"].mean()
