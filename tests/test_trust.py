"""Tests for each trust-flag signal on hand-built cases, plus composition."""

import numpy as np
import pandas as pd
from gridcast.models.exclusions import Exclusion
from gridcast.models.trust import (
    FLAG_COLS,
    integrity_breach,
    irregular_calendar,
    recent_instability,
    temperature_percentile,
    trust_flags,
)

HOUR = pd.Timedelta("1h")


def _features(start="2025-04-01", days=90, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "temperature_2m": rng.uniform(5, 20, len(idx)),
            "is_holiday": 0,
            "load_mw": 12000.0,
        },
        index=idx,
    )


def _rows(origin: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": origin,
            "timestamp": origin + np.arange(24) * HOUR,
            "q50": 12000.0,
            "shift_q50": 0.0,
        }
    )


def test_cold_percentile_uses_only_history_before_origin():
    f = _features()
    origin = pd.Timestamp("2025-06-01", tz="UTC")
    f.loc[origin, "temperature_2m"] = -5.0  # colder than all history
    f.loc[origin + HOUR, "temperature_2m"] = 15.0
    pct = temperature_percentile(_rows(origin), f)
    assert pct[0] == 0.0
    assert 0.5 < pct[1] < 0.8  # uniform(5, 20): 15 C sits near the 67th percentile


def test_calendar_holiday_bridge_and_special_days():
    f = _features(start="2025-05-01", days=60)
    local_date = f.index.tz_convert("Europe/Amsterdam").date
    f.loc[local_date == pd.Timestamp("2025-05-29").date(), "is_holiday"] = 1  # Thursday
    ts = pd.DatetimeIndex(
        [
            "2025-05-29 10:00",  # holiday
            "2025-05-30 10:00",  # Friday bridge day
            "2025-06-03 10:00",  # ordinary Tuesday
            "2025-12-31 10:00",  # New Year's Eve
        ],
        tz="UTC",
    )
    assert irregular_calendar(ts, f).tolist() == [True, True, False, True]


def test_instability_threshold():
    flags = recent_instability(pd.Series([360.0, 120.0, -300.0, np.nan]), pd.Series([12000.0] * 4))
    assert flags.tolist() == [True, False, True, False]  # 3%, 1%, 2.5%, no shift


def test_integrity_recent_gap_or_exclusion_fires_old_gap_does_not():
    origin = pd.Timestamp("2025-06-01", tz="UTC")
    rows = _rows(origin)

    f = _features()
    assert not integrity_breach(rows, f, []).any()

    old = f.copy()
    old.loc[origin - 400 * HOUR, "load_mw"] = np.nan  # outside the 336h lookback
    assert not integrity_breach(rows, old, []).any()

    recent = f.copy()
    recent.loc[origin - 100 * HOUR, "load_mw"] = np.nan
    assert integrity_breach(rows, recent, []).all()

    ex = Exclusion(origin - 10 * HOUR, origin - 5 * HOUR, "t", "t")
    assert integrity_breach(rows, f, [ex]).all()


def test_trust_flags_composition():
    f = _features()
    origin = pd.Timestamp("2025-06-01", tz="UTC")
    f.loc[origin, "temperature_2m"] = -5.0
    rows = _rows(origin)
    rows.loc[5, "shift_q50"] = 500.0  # instability on one row only
    out = trust_flags(rows, f, [])
    assert set(FLAG_COLS) <= set(out.columns)
    assert out["trust_low"].tolist() == out[FLAG_COLS].any(axis=1).tolist()
    assert out.loc[0, "trust_reasons"] == "cold"
    assert out.loc[5, "trust_reasons"] == "instability"
    assert out.loc[10, "trust_reasons"] == ""
