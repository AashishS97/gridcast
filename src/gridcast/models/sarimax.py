"""Dynamic harmonic regression: Fourier seasonal terms + ARMA errors.

Classical SARIMA with s=168 is computationally infeasible (state-space
dimension grows with s) and can only represent one seasonal period. The
standard alternative for multi-seasonal hourly data: encode the daily and
weekly cycles as Fourier (sin/cos) exogenous regressors and let a small
ARMA process model the remaining short-memory dynamics. Fourier columns
are pure functions of the timestamp — zero leakage risk by construction.

Deliberate trade-off: each fold trains on the trailing TRAIN_WINDOW
(8 weeks), not full history. ARMA coefficients describe *local* dynamics
and recent data reflects them best (built-in drift adaptation); the cost is
no annual seasonality, which at a 24h horizon is nearly a constant the
trailing window already sits at. Stated in the writeup, not hidden.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from gridcast.models.baselines import Forecaster

TRAIN_WINDOW = pd.Timedelta("56D")  # 8 weeks, multiple of 7 for weekday balance
DAILY_K = 8  # harmonics for the 24h cycle (sharp ramps need several)
WEEKLY_K = 6  # harmonics for the 168h cycle
ORDER = (2, 0, 1)  # small ARMA on the residual process; d=0 because the
# Fourier terms + constant absorb the deterministic structure


def fourier_terms(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic seasonal regressors: sin/cos pairs for 24h and 168h.

    Phase is anchored to absolute UTC time (hours since epoch), so the same
    timestamp always maps to the same regressor values regardless of window.
    """
    # Timedelta division is resolution-safe (works for ns/us/s-backed indexes);
    # never assume the int64 representation's unit.
    hours = np.asarray((index - pd.Timestamp(0, tz="UTC")) / pd.Timedelta("1h"))
    cols: dict[str, np.ndarray] = {}
    for period, k_max, tag in ((24.0, DAILY_K, "d"), (168.0, WEEKLY_K, "w")):
        for k in range(1, k_max + 1):
            angle = 2.0 * np.pi * k * hours / period
            cols[f"sin_{tag}{k}"] = np.sin(angle)
            cols[f"cos_{tag}{k}"] = np.cos(angle)
    return pd.DataFrame(cols, index=index)


def sarimax_forecaster(history: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    """Fit on the trailing window, forecast the 24h test window."""
    train = history.loc[history.index >= history.index[-1] - TRAIN_WINDOW + pd.Timedelta("1h")]
    exog_train = fourier_terms(train.index)
    exog_test = fourier_terms(test_index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels is chatty about convergence
        model = SARIMAX(
            train.to_numpy(dtype=float),
            exog=exog_train.to_numpy(),
            order=ORDER,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)
        forecast = fitted.forecast(steps=len(test_index), exog=exog_test.to_numpy())
    return np.asarray(forecast, dtype=float)


_ = Forecaster  # protocol conformity documented; sarimax_forecaster matches it
