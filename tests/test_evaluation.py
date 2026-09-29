"""Tests for the pinned evaluation windows. The repo config is the contract:
these tests tie it to Phase 2's generator, the exclusions, and each other."""

import inspect

import pandas as pd
import pytest
from gridcast.models.evaluation import backtest_folds, holdout_info, holdout_origins
from gridcast.models.exclusions import contaminated_origins, load_exclusions
from gridcast.models.splits import rolling_origin_folds

HOUR = pd.Timedelta("1h")


def test_backtest_is_the_phase2_fold_set():
    folds = backtest_folds()
    assert len(folds) == 140
    assert folds[0].origin == pd.Timestamp("2024-06-29", tz="UTC")
    assert folds[-1].origin == pd.Timestamp("2026-05-25", tz="UTC")
    assert all(f.test_end - f.origin == 24 * HOUR for f in folds)
    assert {f.train_start for f in folds} == {pd.Timestamp("2023-07-01", tz="UTC")}


def test_pinned_folds_match_phase2_generator():
    # a dev index ending inside (May 25, May 30) makes the Phase 2 generator
    # stop at the May 25 origin, as it did in Phase 2
    idx = pd.date_range("2023-07-01", "2026-05-28 23:00", freq="1h", tz="UTC")
    gen = rolling_origin_folds(idx, step="5D")
    pinned = backtest_folds()
    key = lambda fs: [(f.fold_id, f.train_start, f.origin, f.test_end) for f in fs]  # noqa: E731
    assert key(gen) == key(pinned)


def test_every_weekday_is_scored_evenly():
    dows = pd.Series([f.origin.dayofweek for f in backtest_folds()]).value_counts()
    assert len(dows) == 7
    assert dows.max() - dows.min() <= 1


def test_scored_windows_do_not_overlap():
    folds = backtest_folds()
    p2, p3 = holdout_info("phase2"), holdout_info("phase3")
    assert folds[-1].test_end <= p2.first_origin
    assert p2.last_origin + 24 * HOUR <= p3.first_origin


def test_scored_windows_avoid_exclusions():
    ex = load_exclusions()
    origins = pd.DatetimeIndex([f.origin for f in backtest_folds()])
    assert not contaminated_origins(origins, ex).any()
    p3 = holdout_info("phase3")
    p3_origins = pd.date_range(p3.first_origin, p3.last_origin, freq="1D")
    assert len(p3_origins) == 15
    assert not contaminated_origins(p3_origins, ex).any()


def test_sealed_and_spent_holdouts_need_one_shot():
    with pytest.raises(PermissionError):
        holdout_origins("phase3")
    with pytest.raises(PermissionError):
        holdout_origins("phase2")
    assert len(holdout_origins("phase3", one_shot=True)) == 15


def test_splitter_default_step_is_not_weekly():
    default = inspect.signature(rolling_origin_folds).parameters["step"].default
    assert pd.Timedelta(default) != pd.Timedelta("7D")


BASE = (
    '[backtest]\nfirst_origin = "2024-06-29T00:00:00Z"\nlast_origin = "{last}"\n'
    'step = "5D"\nhorizon_hours = 24\ntrain_start = "2023-07-01T00:00:00Z"\n'
    "expected_folds = {n}\n"
)


@pytest.mark.parametrize(
    "last,n",
    [("2024-07-05T00:00:00Z", 2), ("2024-07-04T00:00:00Z", 3)],
    ids=["last-origin-unreachable", "fold-count-mismatch"],
)
def test_inconsistent_backtest_config_rejected(tmp_path, last, n):
    path = tmp_path / "eval.toml"
    path.write_text(BASE.format(last=last, n=n), encoding="utf-8")
    with pytest.raises(ValueError):
        backtest_folds(path)
