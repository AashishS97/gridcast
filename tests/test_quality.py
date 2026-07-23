"""Unit tests for quality checks, on synthetic data with planted defects."""

import pandas as pd
from gridcast.data import quality


def make_series(n: int = 192) -> pd.DataFrame:
    """Two days of clean, wiggly, in-bounds 15-min load."""
    ts = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    load = 12_000.0 + (pd.Series(range(n)) % 40) * 10.0
    return pd.DataFrame({"timestamp": ts, "load_mw": load.values})


def test_clean_series_has_no_findings():
    checks = {
        "gaps": quality.find_gaps(make_series()),
        "duplicates": quality.find_duplicates(make_series()),
        "out_of_bounds": quality.find_out_of_bounds(make_series()),
        "flatlines": quality.find_flatlines(make_series()),
        "spikes": quality.find_spikes(make_series()),
    }
    assert all(len(v) == 0 for v in checks.values())


def test_finds_gap_as_one_run():
    df = make_series().drop(index=[10, 11, 12]).reset_index(drop=True)
    gaps = quality.find_gaps(df)
    assert len(gaps) == 1
    assert gaps.loc[0, "n_missing"] == 3


def test_finds_both_copies_of_duplicate():
    df = make_series()
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True).sort_values("timestamp")
    assert len(quality.find_duplicates(df)) == 2


def test_zero_is_out_of_bounds():
    df = make_series()
    df.loc[20, "load_mw"] = 0.0
    assert len(quality.find_out_of_bounds(df)) == 1


def test_finds_flatline_run():
    df = make_series()
    df.loc[30:40, "load_mw"] = 13_337.0  # 11 identical points (loc is inclusive)
    flats = quality.find_flatlines(df)
    assert len(flats) == 1
    assert flats.loc[0, "n_points"] == 11


def test_spike_flags_jump_up_and_back_down():
    df = make_series()
    df.loc[50, "load_mw"] += 5_000.0
    assert len(quality.find_spikes(df)) == 2  # the jump up and the drop back
