"""Fourier-SARIMAX: regressor correctness and end-to-end sanity."""

import numpy as np
import pandas as pd
from gridcast.models.sarimax import DAILY_K, WEEKLY_K, fourier_terms, sarimax_forecaster


def test_fourier_shape_and_bounds():
    idx = pd.date_range("2024-01-01", periods=500, freq="1h", tz="UTC")
    ft = fourier_terms(idx)
    assert ft.shape == (500, 2 * (DAILY_K + WEEKLY_K))
    assert float(ft.abs().max().max()) <= 1.0 + 1e-12
    assert not ft.isna().any().any()


def test_fourier_periodicity():
    idx = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
    ft = fourier_terms(idx)
    daily = [c for c in ft.columns if c.endswith("1") and "_d" in c]
    weekly = [c for c in ft.columns if c.endswith("1") and "_w" in c]
    assert np.allclose(ft[daily].iloc[0], ft[daily].iloc[24], atol=1e-9)
    assert np.allclose(ft[weekly].iloc[0], ft[weekly].iloc[168], atol=1e-9)


def test_fourier_is_window_independent():
    """Same timestamp -> same regressors, however the window is cut (no leakage
    via phase drift)."""
    long = pd.date_range("2024-01-01", periods=1000, freq="1h", tz="UTC")
    short = long[500:600]
    a = fourier_terms(long).loc[short]
    b = fourier_terms(short)
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)


def test_sarimax_recovers_multiseasonal_signal():
    """On a clean two-seasonality signal + noise, forecasts should track the
    truth far better than a flat prediction would."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=70 * 24, freq="1h", tz="UTC")
    hour_of_week = idx.dayofweek * 24 + idx.hour
    signal = (
        12_000
        + 2_000 * np.sin(2 * np.pi * idx.hour / 24)
        + 800 * np.sin(2 * np.pi * hour_of_week / 168)
    )
    y = pd.Series(signal + rng.normal(0, 100, len(idx)), index=idx)

    history, test_index = y.iloc[:-24], y.index[-24:]
    preds = sarimax_forecaster(history, test_index)

    truth = y.iloc[-24:].to_numpy()
    mae = np.mean(np.abs(preds - truth))
    flat_mae = np.mean(np.abs(truth - history.iloc[-1]))
    assert len(preds) == 24
    assert np.all(np.isfinite(preds))
    assert mae < 300  # tracks a 2.8 GW-amplitude signal within noise scale
    assert mae < flat_mae / 3  # and demolishes persistence on this signal
