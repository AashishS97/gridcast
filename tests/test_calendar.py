"""Unit tests for calendar features — DST boundary and holiday logic."""

import pandas as pd
import pytest
from gridcast.features import calendar


def test_hour_is_local_across_spring_forward():
    # 2025-03-30: NL jumps 02:00 -> 03:00. Consecutive UTC hours land on
    # local 01:00 then 03:00 — local 02:00 never exists.
    ts = pd.to_datetime(["2025-03-30 00:00", "2025-03-30 01:00"], utc=True)
    out = calendar.add_calendar_features(pd.DataFrame({"timestamp": ts}))
    assert list(out["hour"]) == [1, 3]


def test_hour_is_local_across_fall_back():
    # 2025-10-26: NL falls back 03:00 -> 02:00. Two UTC hours both land
    # on local hour 2 — it happens twice.
    ts = pd.to_datetime(["2025-10-26 00:00", "2025-10-26 01:00"], utc=True)
    out = calendar.add_calendar_features(pd.DataFrame({"timestamp": ts}))
    assert list(out["hour"]) == [2, 2]


def test_dutch_holidays_including_moved_kings_day():
    # 2025-04-27 (King's birthday) is a Sunday, so Koningsdag is observed
    # Saturday 2025-04-26. The Monday after is an ordinary day.
    ts = pd.to_datetime(["2024-12-25 12:00", "2025-04-26 12:00", "2025-04-28 12:00"], utc=True)
    out = calendar.add_calendar_features(pd.DataFrame({"timestamp": ts}))
    assert list(out["is_holiday"]) == [1, 1, 0]


def test_weekend_flag():
    ts = pd.to_datetime(["2025-01-03 12:00", "2025-01-04 12:00"], utc=True)  # Fri, Sat
    out = calendar.add_calendar_features(pd.DataFrame({"timestamp": ts}))
    assert list(out["is_weekend"]) == [0, 1]


def test_naive_timestamps_are_rejected():
    ts = pd.to_datetime(["2025-01-01 12:00"])  # naive on purpose
    with pytest.raises(ValueError, match="tz-naive"):
        calendar.add_calendar_features(pd.DataFrame({"timestamp": ts}))
