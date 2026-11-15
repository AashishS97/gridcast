"""Shared synthetic data: an hourly table with the real feature schema."""

import numpy as np
import pandas as pd
import pytest


def make_frame(days: int = 300) -> pd.DataFrame:
    """Hourly from 2024-01-01 UTC: daily and weekly shape, temperature
    dependence, noise. 300 days ends 2024-10-26 23:00."""
    ts = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    n = len(ts)
    hour = ts.hour.to_numpy()
    doy = ts.dayofyear.to_numpy()
    weekend = np.asarray(ts.dayofweek >= 5)
    temp = 10 + 8 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 1.5, n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "hour": hour.astype("int8"),
            "dow": ts.dayofweek.to_numpy().astype("int8"),
            "is_weekend": weekend.astype("int8"),
            "month": ts.month.to_numpy().astype("int8"),
            "day_of_year": doy.astype("int16"),
            "is_holiday": np.zeros(n, dtype="int8"),
            "temperature_2m": temp,
            "shortwave_radiation": np.clip(600 * np.sin(np.pi * (hour - 6) / 12), 0, None),
            "wind_speed_10m": rng.gamma(2.0, 3.0, n),
        }
    )
    df["temp_mean_24h"] = df["temperature_2m"].rolling(24, min_periods=1).mean()
    load = (
        12000
        + 1500 * np.sin(2 * np.pi * (hour - 8) / 24)
        - 120 * (temp - 10)
        - 800 * weekend
        + rng.normal(0, 150, n)
    )
    df["load_mw"] = load
    df["da_forecast_mw"] = load * 0.8
    return df


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    return make_frame()
