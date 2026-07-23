"""Unit tests for the feature-table join. No files, no network."""

import pandas as pd
from gridcast.features.build import build_feature_table


def hourly(n=48, start="2025-01-01"):
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


def test_target_spine_is_kept_and_missing_covariates_stay_nan():
    ts = hourly(48)
    actual = pd.DataFrame({"timestamp": ts, "load_mw": 13_000.0})
    day_ahead = pd.DataFrame({"timestamp": ts, "load_mw": 12_900.0}).drop(index=[10])
    weather = pd.DataFrame(
        {
            "timestamp": ts[:43],
            "temperature_2m": 5.0,
            "shortwave_radiation": 0.0,
            "wind_speed_10m": 4.0,
            "temp_mean_24h": 5.0,
        }
    )
    out = build_feature_table(actual, day_ahead, weather)
    assert len(out) == 48  # spine never shrinks
    assert out["da_forecast_mw"].isna().sum() == 1
    assert out["temperature_2m"].isna().sum() == 5
    for col in ["hour", "dow", "is_weekend", "month", "day_of_year", "is_holiday"]:
        assert col in out.columns
