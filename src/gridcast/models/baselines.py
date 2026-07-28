"""Baseline forecasters.

Every forecaster in this project implements the same protocol:

    forecaster(history: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray

where `history` is the target strictly before the fold origin, and
`test_index` is the 24 hourly timestamps to predict. The forecaster must use
nothing but `history` (or, for external benchmarks, data that was genuinely
available before the origin).

Baselines here:
- persistence: last observed value held flat. The floor.
- seasonal naive s=24: same hour yesterday. Daily cycle, wrong on
  weekday/weekend transitions.
- seasonal naive s=168: same hour last week. THE load benchmark; fails on
  holidays, weather swings, and trend — which is the feature wishlist for
  the ML model.
- entsoe_day_ahead: the TSO's published forecast, as an external benchmark.
  Note its information cutoff (~noon D-1) is earlier than our origins
  (midnight), which slightly favors our models; we state this rather than
  hide it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

Forecaster = Callable[[pd.Series, pd.DatetimeIndex], np.ndarray]


class _NamedForecaster(Protocol):
    def __call__(self, history: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray: ...


def persistence(history: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    """ŷ(t) = last observed value before the origin, for all 24 hours."""
    return np.full(len(test_index), history.iloc[-1], dtype=float)


def _seasonal_naive(lag_hours: int) -> Forecaster:
    lag = pd.Timedelta(hours=lag_hours)

    def forecast(history: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
        lookup = test_index - lag
        # For horizon <= lag, every lookup timestamp precedes the origin,
        # so this uses only past data. Guard anyway:
        origin = test_index[0]
        if (lookup >= origin).any():
            raise ValueError(
                f"seasonal naive s={lag_hours} would need future data for "
                f"this horizon — horizon must be <= {lag_hours}h"
            )
        return history.reindex(lookup).to_numpy(dtype=float)

    return forecast


seasonal_naive_24: Forecaster = _seasonal_naive(24)
seasonal_naive_168: Forecaster = _seasonal_naive(168)


def make_external_benchmark(external: pd.Series) -> Forecaster:
    """Wrap a pre-existing forecast series (e.g. ENTSO-E day-ahead) so it
    plugs into the same harness. Missing hours come back as NaN and are
    counted, not silently dropped."""

    def forecast(history: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
        return external.reindex(test_index).to_numpy(dtype=float)

    return forecast


def make_bias_corrected_benchmark(
    external: pd.Series,
    actual: pd.Series,
    window: str | pd.Timedelta = "28D",
) -> Forecaster:
    """External forecast plus its trailing mean error, per hour of day.

    The ENTSO-E NL day-ahead forecast is not uniformly biased low — the gap
    is hour-shaped (near zero overnight, several GW at midday: the forecast
    is amplitude-damped vs the published actuals). A single additive
    constant overcorrects the night and undercorrects the day, so we
    estimate 24 biases, one per hour, each from the trailing `window` of
    same-hour errors. Only pre-origin data is used, so this remains a
    legitimate forecast.
    """
    window = pd.Timedelta(window)

    def forecast(history: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
        origin = test_index[0]
        recent = pd.date_range(
            origin - window, origin - pd.Timedelta("1h"), freq="1h", tz=test_index.tz
        )
        errors = actual.reindex(recent) - external.reindex(recent)
        bias_by_hour = errors.groupby(recent.hour).mean()  # NaN-safe per hour
        bias = bias_by_hour.reindex(test_index.hour).fillna(0.0).to_numpy()
        return external.reindex(test_index).to_numpy(dtype=float) + bias

    return forecast
