"""Unit tests for national weather aggregation. No network."""

import pandas as pd
import pytest
from gridcast.data import weather


def make_city(value: float, n: int = 48) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": value,
            "shortwave_radiation": value * 10,
            "wind_speed_10m": value / 2,
        }
    )


def all_cities(base: float = 10.0) -> dict[str, pd.DataFrame]:
    return {name: make_city(base) for name in weather.CITIES}


def test_weights_sum_to_one():
    assert abs(sum(w for _, _, w in weather.CITIES.values()) - 1.0) < 1e-9


def test_weighted_average_of_identical_cities_is_identity():
    national = weather.build_national(all_cities(10.0))
    assert national["temperature_2m"].round(9).eq(10.0).all()


def test_weighted_average_uses_weights():
    frames = all_cities(10.0)
    frames["amsterdam"] = make_city(20.0)  # weight 0.30
    national = weather.build_national(frames)
    expected = 20.0 * 0.30 + 10.0 * 0.70
    assert national["temperature_2m"].round(9).eq(round(expected, 9)).all()


def test_rolling_mean_is_trailing_only():
    frames = all_cities(10.0)
    national = weather.build_national(frames)
    # First 23 hours cannot have a full 24h window:
    assert national["temp_mean_24h"].iloc[:23].isna().all()
    assert national["temp_mean_24h"].iloc[23] == pytest.approx(10.0)
    # Perturb the LAST hour of the input; earlier rolling values must not move.
    frames["amsterdam"].loc[47, "temperature_2m"] = 99.0
    perturbed = weather.build_national(frames)
    pd.testing.assert_series_equal(
        national["temp_mean_24h"].iloc[:47], perturbed["temp_mean_24h"].iloc[:47]
    )
    assert perturbed["temp_mean_24h"].iloc[47] > national["temp_mean_24h"].iloc[47]
