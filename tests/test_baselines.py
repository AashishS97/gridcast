"""Baselines must be exactly right — they're the control group."""

import numpy as np
import pandas as pd
import pytest
from gridcast.models.backtest import run_backtest
from gridcast.models.baselines import (
    make_external_benchmark,
    persistence,
    seasonal_naive_24,
    seasonal_naive_168,
)
from gridcast.models.metrics import mae, mape, rmse, summarize
from gridcast.models.splits import rolling_origin_folds


@pytest.fixture
def daily_series() -> pd.Series:
    """Perfectly 24h-periodic series: both seasonal naives must be exact."""
    idx = pd.date_range("2024-01-01", periods=600 * 24, freq="1h", tz="UTC")
    values = 10_000 + 2_000 * np.sin(2 * np.pi * idx.hour / 24)
    return pd.Series(values, index=idx)


@pytest.fixture
def weekly_series() -> pd.Series:
    """168h-periodic but NOT 24h-periodic: only weekly naive is exact."""
    idx = pd.date_range("2024-01-01", periods=600 * 24, freq="1h", tz="UTC")
    hour_of_week = idx.dayofweek * 24 + idx.hour
    values = 10_000 + 2_000 * np.sin(2 * np.pi * hour_of_week / 168)
    return pd.Series(values.to_numpy(), index=idx)


def _one_fold(series):
    fold = rolling_origin_folds(series.index)[0]
    history = series[series.index < fold.origin]
    test_index = series.index[fold.test_mask(series.index)]
    return history, test_index


def test_persistence_holds_last_value_flat(daily_series):
    history, test_index = _one_fold(daily_series)
    preds = persistence(history, test_index)
    assert np.allclose(preds, history.iloc[-1])
    assert len(preds) == 24


def test_seasonal_naive_24_exact_on_daily_pattern(daily_series):
    history, test_index = _one_fold(daily_series)
    preds = seasonal_naive_24(history, test_index)
    truth = daily_series.reindex(test_index).to_numpy()
    assert np.allclose(preds, truth)


def test_seasonal_naive_168_exact_on_weekly_pattern(weekly_series):
    history, test_index = _one_fold(weekly_series)
    preds = seasonal_naive_168(history, test_index)
    truth = weekly_series.reindex(test_index).to_numpy()
    assert np.allclose(preds, truth)


def test_seasonal_naive_24_wrong_on_weekly_pattern(weekly_series):
    history, test_index = _one_fold(weekly_series)
    preds = seasonal_naive_24(history, test_index)
    truth = weekly_series.reindex(test_index).to_numpy()
    assert not np.allclose(preds, truth)


def test_seasonal_naives_use_only_past_data(daily_series):
    history, test_index = _one_fold(daily_series)
    origin = test_index[0]
    assert (test_index - pd.Timedelta("24h") < origin).all()
    assert (test_index - pd.Timedelta("168h") < origin).all()
    assert history.index.max() < origin


def test_external_benchmark_alignment_and_missing():
    idx = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    external = pd.Series(np.arange(48, dtype=float), index=idx)
    fc = make_external_benchmark(external.drop(idx[30]))  # one missing hour
    test_index = idx[24:48]
    preds = fc(pd.Series(dtype=float), test_index)
    assert np.isnan(preds[6])  # idx[30] is position 6 in the window
    assert preds[0] == 24.0


def test_metrics_hand_computed():
    y = np.array([100.0, 200.0])
    p = np.array([110.0, 180.0])
    assert mae(y, p) == pytest.approx(15.0)
    assert rmse(y, p) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(y, p) == pytest.approx((0.10 + 0.10) / 2 * 100)


def test_mape_rejects_zeros():
    with pytest.raises(ValueError, match="zero"):
        mape(np.array([0.0, 1.0]), np.array([1.0, 1.0]))


def test_backtest_output_shape_and_columns(daily_series):
    folds = rolling_origin_folds(daily_series.index, max_folds=3)
    results = run_backtest(daily_series, folds, {"persistence": persistence})
    assert len(results) == 3 * 24
    assert set(results.columns) == {
        "fold",
        "model",
        "timestamp",
        "horizon",
        "y_true",
        "y_pred",
        "hour_local",
    }
    assert results["horizon"].min() == 1 and results["horizon"].max() == 24
    assert results["y_pred"].notna().all()


def test_summarize_counts_missing():
    df = pd.DataFrame(
        {
            "model": ["a"] * 4,
            "y_true": [100.0, 100.0, 100.0, 100.0],
            "y_pred": [90.0, 110.0, np.nan, 100.0],
        }
    )
    out = summarize(df)
    assert out.loc[0, "n"] == 3
    assert out.loc[0, "n_missing"] == 1
