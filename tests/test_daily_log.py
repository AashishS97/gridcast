"""Daily shadow log on the synthetic artifact: final-actuals gating, scoring
arithmetic, idempotent appends, and the stale-data alarm."""

import numpy as np
import pandas as pd
import pytest

from gridcast.models.exclusions import Exclusion
from gridcast.serving.daily_log import FORECAST_COLS, read_log, run, score_origin
from gridcast.serving.forecaster import Forecaster, load_artifact


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture(scope="module")
def artifact(synthetic_artifact_dir):
    return load_artifact(synthetic_artifact_dir)


def make(artifact, frame, exclusions=(), provisional_days=0):
    return Forecaster(artifact, frame, list(exclusions), provisional_days)


def test_final_actuals_waits_for_the_whole_target_day(artifact, synthetic_frame):
    # data ends 2024-10-26 23:00; 10 provisional days -> final up to 10-16 23:00
    fc = make(artifact, synthetic_frame, provisional_days=10)
    assert fc.provisional_from == utc("2024-10-16 23:00")
    assert fc.final_actuals(utc("2024-10-15")) is not None  # ends 10-16 00:00
    assert fc.final_actuals(utc("2024-10-16")) is None  # ends 10-17 00:00 > 10-16 23:00
    values = fc.final_actuals(utc("2024-10-10"))
    expected = synthetic_frame.set_index("timestamp")["load_mw"]
    expected = expected.loc[utc("2024-10-10") : utc("2024-10-10 23:00")].to_numpy()
    np.testing.assert_array_equal(values, expected)


def test_final_actuals_refuses_excluded_target_hours(artifact, synthetic_frame):
    ex = Exclusion(utc("2024-10-10 12:00"), utc("2024-10-10 13:00"), "test", "test")
    fc = make(artifact, synthetic_frame, exclusions=[ex])
    assert fc.final_actuals(utc("2024-10-10")) is None
    assert fc.final_actuals(utc("2024-10-11")) is not None


def test_score_origin_arithmetic():
    group = pd.DataFrame(
        {
            "horizon": range(1, 25),
            "p10": np.full(24, 90.0),
            "p50": np.full(24, 100.0),
            "p90": np.full(24, 110.0),
            "trust_low": [True] * 3 + [False] * 21,
        }
    )
    y = np.array([100.0] * 12 + [120.0] * 12)  # half inside, half above p90
    s = score_origin(group, y)
    assert s["mae_p50"] == pytest.approx(10.0)
    assert s["coverage_80"] == pytest.approx(0.5)
    assert s["width_mean"] == pytest.approx(20.0)
    assert s["pinball_p50"] == pytest.approx(0.5 * 10.0)
    # p90 = 110: zero loss below it, 0.9 * 10 above it on half the hours
    assert s["pinball_p90"] == pytest.approx(0.5 * 0.9 * 10.0 + 0.5 * 0.1 * 10.0)
    assert s["trust_low_hours"] == 3


def test_run_appends_once_and_is_idempotent(artifact, synthetic_frame, tmp_path):
    fc = make(artifact, synthetic_frame, provisional_days=10)
    now = utc("2024-10-27 06:00")
    assert run(fc, tmp_path, now) == 0
    forecasts = read_log(tmp_path / "forecasts.csv", FORECAST_COLS)
    n_origins = forecasts["origin"].nunique()
    assert n_origins == len(fc.servable_origins())
    assert (forecasts.groupby("origin").size() == 24).all()
    scores = pd.read_csv(tmp_path / "scores.csv")
    assert len(scores) == 15  # origins 10-01 .. 10-15 have final target days
    assert (scores["coverage_80"].between(0, 1)).all()

    assert run(fc, tmp_path, now + pd.Timedelta("1D")) == 0  # same data: nothing new
    assert read_log(tmp_path / "forecasts.csv", FORECAST_COLS)["origin"].nunique() == n_origins
    assert len(pd.read_csv(tmp_path / "scores.csv")) == 15
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert runs["new_forecasts"].tolist() == [n_origins, 0]
    assert runs["new_scores"].tolist() == [15, 0]
    assert (tmp_path / "README.md").read_text(encoding="utf-8").startswith("# GridCast")


def test_newly_final_days_are_scored_on_a_later_vintage(artifact, synthetic_frame, tmp_path):
    early = synthetic_frame[synthetic_frame["timestamp"] < utc("2024-10-20")]
    run(make(artifact, early, provisional_days=10), tmp_path, utc("2024-10-20 06:00"))
    first = len(pd.read_csv(tmp_path / "scores.csv"))
    run(make(artifact, synthetic_frame, provisional_days=10), tmp_path, utc("2024-10-27 06:00"))
    scores = pd.read_csv(tmp_path / "scores.csv")
    assert len(scores) > first
    assert not scores["origin"].duplicated().any()


def test_stale_data_fails_the_run_but_still_logs_it(artifact, synthetic_frame, tmp_path):
    fc = make(artifact, synthetic_frame, provisional_days=10)
    assert run(fc, tmp_path, utc("2024-11-15")) == 1
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert runs["status"].iloc[-1].startswith("stale_data")
