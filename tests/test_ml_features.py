"""Design-matrix tests. The canary test is the important one: it FAILS if
any feature depends on target values at or after the origin."""

import numpy as np
import pandas as pd
import pytest
from gridcast.models.ml_features import (
    build_design_matrix,
    daily_origins,
    feature_columns,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """60 days of synthetic data where load encodes hours-since-start,
    so every lag value is exactly verifiable."""
    idx = pd.date_range("2024-01-01", periods=60 * 24, freq="1h", tz="UTC")
    hours_since_start = np.arange(len(idx), dtype=float)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "load_mw": 10_000.0 + hours_since_start,  # strictly increasing, unique
            "da_forecast_mw": 9_000.0 + hours_since_start,
            "temperature_2m": 10.0,
            "shortwave_radiation": 0.0,
            "wind_speed_10m": 5.0,
            "temp_mean_24h": 10.0,
            "hour": idx.hour,
            "dow": idx.dayofweek,
            "is_weekend": (idx.dayofweek >= 5).astype("int8"),
            "month": idx.month,
            "day_of_year": idx.dayofyear,
            "is_holiday": 0,
        }
    )
    return df


def _one_origin(frame, ts="2024-02-15 00:00", **kw):
    origins = pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC")])
    return build_design_matrix(frame, origins, **kw)


def test_shape_and_horizons(frame):
    m = _one_origin(frame)
    assert len(m) == 24
    assert list(m["horizon"]) == list(range(1, 25))
    assert (m["timestamp"] == m["origin"] + (m["horizon"] - 1) * pd.Timedelta("1h")).all()


def test_lag_values_exact(frame):
    """load encodes hours-since-start, so lag_L must equal y_true - L."""
    m = _one_origin(frame)
    for lag in (24, 48, 168, 336):
        assert np.allclose(m[f"lag_{lag}"], m["y_true"] - lag)


def test_short_lags_masked_beyond_availability(frame):
    m = _one_origin(frame, lags=(2, 24))
    lag2 = m.set_index("horizon")["lag_2"]
    assert lag2.loc[[1, 2]].notna().all()  # h <= L: observed
    assert lag2.loc[3:].isna().all()  # h > L: inside forecast window
    assert m["lag_24"].notna().all()


def test_origin_anchored_rolling_window(frame):
    """roll24_mean must cover [origin-24h, origin-1h] exactly."""
    m = _one_origin(frame)
    y = frame.set_index("timestamp")["load_mw"]
    origin = m["origin"].iloc[0]
    window = y.loc[origin - pd.Timedelta("24h") : origin - pd.Timedelta("1h")]
    assert len(window) == 24
    assert m["roll24_mean"].iloc[0] == pytest.approx(window.mean())
    assert m["last_obs"].iloc[0] == y.loc[origin - pd.Timedelta("1h")]
    assert (m["roll24_mean"] == m["roll24_mean"].iloc[0]).all()  # origin-level, shared


def test_leakage_canary(frame):
    """THE test. Corrupt every target value at/after the origin; every
    feature must be bit-identical. If this fails, a feature saw the future."""
    origin_ts = "2024-02-15 00:00"
    before = _one_origin(frame, origin_ts)

    corrupted = frame.copy()
    future = corrupted["timestamp"] >= pd.Timestamp(origin_ts, tz="UTC")
    corrupted.loc[future, "load_mw"] = -999_999.0
    after = _one_origin(corrupted, origin_ts)

    feats = feature_columns(before)
    pd.testing.assert_frame_equal(before[feats], after[feats])
    assert (after["y_true"] == -999_999.0).all()  # only the label changed


def test_builder_is_context_independent(frame):
    """Golden rule: rows for an origin are identical whether built alone
    (inference) or alongside 40 other origins (training replay)."""
    many = build_design_matrix(frame, daily_origins(pd.DatetimeIndex(frame["timestamp"]))[35:76])
    one = _one_origin(frame)
    origin = one["origin"].iloc[0]
    subset = many[many["origin"] == origin].reset_index(drop=True)
    pd.testing.assert_frame_equal(subset, one)


def test_inference_rows_have_nan_labels(frame):
    """A genuine future origin (next midnight after the data ends):
    pre-origin features exist, labels honestly NaN."""
    last_ts = frame["timestamp"].iloc[-1]
    origins = pd.DatetimeIndex([last_ts.floor("D") + pd.Timedelta("1D")])
    m = build_design_matrix(frame, origins)
    assert m["y_true"].isna().all()  # whole window is future
    assert m["lag_24"].notna().all()  # lags reach back into observed data
    assert m["last_obs"].notna().all()  # origin-anchored state exists
