"""Unit tests for the cleaning module, on synthetic data."""

import pandas as pd
from gridcast.data import clean
from gridcast.data.base import LoadKind


def make_df(n: int = 192) -> pd.DataFrame:
    """Two days of clean, wiggly 15-min load on a full grid."""
    ts = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    load = 12_000.0 + (pd.Series(range(n)) % 40) * 10.0
    return pd.DataFrame({"timestamp": ts, "load_mw": load.values})


def test_sag_is_flagged_but_level_shift_is_not():
    df = make_df()
    df.loc[50, "load_mw"] -= 5_000.0  # down-and-back: a sag
    df.loc[100:, "load_mw"] += 5_000.0  # down... and STAY: real event
    s = clean.to_full_grid(df)
    flagged = clean.flag_single_point_outliers(s)
    assert len(flagged) == 1
    assert flagged[0] == df.loc[50, "timestamp"]


def test_interpolation_fills_short_runs_and_leaves_long_runs():
    df = make_df()
    s = clean.to_full_grid(df)
    short = s.index[10:13]  # 3 quarters
    long = s.index[50:70]  # 20 quarters
    s.loc[short] = float("nan")
    s.loc[long] = float("nan")
    out = clean.interpolate_short_gaps(s)
    assert out.loc[short].notna().all()
    assert out.loc[long].isna().all()  # untouched, not partially filled


def test_hourly_requires_all_four_quarters():
    df = make_df()
    s = clean.to_full_grid(df)
    s.iloc[4] = float("nan")  # break one quarter of hour 1
    hourly = clean.to_hourly(s)
    assert pd.notna(hourly.iloc[0])
    assert pd.isna(hourly.iloc[1])
    assert hourly.iloc[0] == s.iloc[0:4].mean()


def test_clean_series_end_to_end():
    df = make_df()
    df.loc[50, "load_mw"] -= 5_000.0  # sag -> repaired
    df = df.drop(index=[80]).reset_index(drop=True)  # 1-quarter gap -> filled
    out = clean.clean_series(df, LoadKind.ACTUAL)
    assert len(out) == 48
    assert out["load_mw"].notna().all()
    assert str(out["timestamp"].dt.tz) == "UTC"


def test_day_ahead_is_not_outlier_repaired():
    df = make_df()
    df.loc[50, "load_mw"] -= 5_000.0
    out = clean.clean_series(df, LoadKind.DAY_AHEAD)
    hour_of_sag = out.loc[12, "load_mw"]  # quarter 50 lives in hour 12
    assert hour_of_sag < 11_500.0  # the sag survives into the mean
