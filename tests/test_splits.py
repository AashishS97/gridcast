"""Tests for chronological splitting — the anti-leakage tripwire."""

import pandas as pd
import pytest
from gridcast.models.splits import folds_summary, holdout_cutoff, rolling_origin_folds


@pytest.fixture
def index() -> pd.DatetimeIndex:
    """Two full years of complete hourly UTC data."""
    return pd.date_range("2024-01-01 00:00", "2025-12-31 23:00", freq="1h", tz="UTC")


def test_no_temporal_overlap(index):
    for fold in rolling_origin_folds(index):
        train_times = index[fold.train_mask(index)]
        test_times = index[fold.test_mask(index)]
        assert train_times.max() < test_times.min()


def test_test_window_is_24_rows(index):
    for fold in rolling_origin_folds(index):
        assert fold.test_mask(index).sum() == 24


def test_origins_at_midnight_weekly(index):
    folds = rolling_origin_folds(index)
    origins = [f.origin for f in folds]
    assert all(o.hour == 0 and o.minute == 0 for o in origins)
    spacings = {b - a for a, b in zip(origins, origins[1:], strict=False)}
    assert spacings == {pd.Timedelta("7D")}


def test_expanding_window_grows_from_data_start(index):
    folds = rolling_origin_folds(index, expanding=True)
    sizes = [f.train_mask(index).sum() for f in folds]
    assert all(f.train_start == index[0] for f in folds)
    assert sizes == sorted(sizes)
    assert sizes[0] >= 364 * 24


def test_sliding_window_constant_size(index):
    folds = rolling_origin_folds(index, expanding=False)
    sizes = {f.train_mask(index).sum() for f in folds}
    assert sizes == {364 * 24}


def test_max_folds_keeps_most_recent(index):
    all_folds = rolling_origin_folds(index)
    capped = rolling_origin_folds(index, max_folds=5)
    assert len(capped) == 5
    assert [f.origin for f in capped] == [f.origin for f in all_folds[-5:]]


def test_last_fold_fits_inside_data(index):
    folds = rolling_origin_folds(index)
    assert folds[-1].test_end - pd.Timedelta("1h") <= index[-1]


def test_holdout_cutoff_is_midnight_and_partitions(index):
    cutoff = holdout_cutoff(index, test_span="56D")
    assert cutoff.hour == 0 and cutoff.minute == 0
    test_rows = (index >= cutoff).sum()
    assert test_rows == 56 * 24
    assert (index < cutoff).sum() + test_rows == len(index)


def test_rejects_unsorted_and_duplicates(index):
    with pytest.raises(ValueError, match="sorted"):
        rolling_origin_folds(index[::-1])
    dup = index[:100].append(index[99:100])
    with pytest.raises(ValueError, match="duplicate"):
        rolling_origin_folds(dup.sort_values())


def test_insufficient_data_raises():
    short = pd.date_range("2024-01-01", periods=100 * 24, freq="1h", tz="UTC")
    with pytest.raises(ValueError, match="not enough data"):
        rolling_origin_folds(short)


def test_summary_row_counts_reflect_gaps(index):
    # knock 3 hours out of the data; summary should count actual rows
    gappy = index.delete([5000, 5001, 5002])
    folds = rolling_origin_folds(gappy)
    summary = folds_summary(folds, gappy)
    assert summary.loc[0, "train_rows"] == folds[0].train_mask(gappy).sum()
