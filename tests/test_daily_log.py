"""Daily shadow log on the synthetic artifact: final-actuals gating, scoring
arithmetic, idempotent appends, and the stale-data alarm."""

import numpy as np
import pandas as pd
import pytest

from gridcast.models.exclusions import Exclusion
from gridcast.serving.daily_log import (
    FORECAST_COLS,
    HOURLY_COLS,
    INPUT_KEYS,
    RUN_COLS,
    VINTAGE_COLS,
    VINTAGE_DAYS,
    read_log,
    run,
    score_origin,
)
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


def test_inputs_and_vintages_are_logged(artifact, synthetic_frame, tmp_path):
    fc = make(artifact, synthetic_frame, provisional_days=10)
    run(fc, tmp_path, utc("2024-10-27 06:00"))
    x = read_log(tmp_path / "inputs.csv", INPUT_KEYS, extra_ok=True)
    assert list(x.columns) == INPUT_KEYS + [f for f in artifact.features if f != "horizon"]
    assert len(x) == 24 * len(fc.servable_origins())
    # what was logged is what the boosters predicted from
    origin = utc("2024-10-10")
    logged = x[x["origin"] == origin].sort_values("horizon")
    _, used = fc.forecast_with_inputs(origin)
    np.testing.assert_allclose(logged[artifact.features], used[artifact.features], rtol=1e-12)

    run(fc, tmp_path, utc("2024-10-28 06:00"))
    v = read_log(tmp_path / "vintages.csv", VINTAGE_COLS)
    assert len(v) == 2 * VINTAGE_DAYS  # one ledger per run, even with unchanged data
    assert v.groupby("run_at")["day"].max().eq(utc("2024-10-26")).all()
    assert len(read_log(tmp_path / "inputs.csv", INPUT_KEYS, extra_ok=True)) == len(x)


def test_runs_log_from_before_code_sha_still_reads(
    artifact, synthetic_frame, tmp_path, monkeypatch
):
    monkeypatch.delenv("GITHUB_SHA", raising=False)  # Actions sets it; outside CI it is absent
    old = (
        "run_at,artifact,data_as_of,provisional_from,n_servable,last_servable,"
        "new_forecasts,new_scores,status\n"
        "2024-10-20T06:00:00Z,synthetic,2024-10-19T23:00:00Z,2024-10-09T23:00:00Z,"
        "9,2024-10-09T00:00:00Z,9,0,ok\n"
    )
    (tmp_path / "runs.csv").write_text(old, encoding="utf-8")
    run(make(artifact, synthetic_frame, provisional_days=10), tmp_path, utc("2024-10-27 06:00"))
    runs = read_log(tmp_path / "runs.csv", RUN_COLS)
    assert len(runs) == 2
    assert pd.isna(runs["code_sha"].iloc[0])
    assert runs["code_sha"].iloc[1] == "local"


def test_code_sha_comes_from_github_sha(artifact, synthetic_frame, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef0123")
    run(make(artifact, synthetic_frame, provisional_days=10), tmp_path, utc("2024-10-27 06:00"))
    assert read_log(tmp_path / "runs.csv", RUN_COLS)["code_sha"].iloc[-1] == "0123456789ab"


def test_hourly_log_matches_scores_and_recovers_raw(artifact, synthetic_frame, tmp_path):
    fc = make(artifact, synthetic_frame, provisional_days=10)
    run(fc, tmp_path, utc("2024-10-27 06:00"))
    h = read_log(tmp_path / "hourly.csv", HOURLY_COLS)
    scores = pd.read_csv(tmp_path / "scores.csv")
    assert h["origin"].nunique() == len(scores) and len(h) == 24 * len(scores)
    # per-day MAE from the hourly rows equals the scored MAE
    mae = (h["y"] - h["p50"]).abs().groupby(h["origin"]).mean().round(2).to_numpy()
    # (hourly values are stored to 0.1 MW, so the two agree to within 0.05 MW)
    np.testing.assert_allclose(mae, scores["mae_p50"].to_numpy(), atol=0.06)
    # raw quantiles are logged at forecast time and survive into hourly
    _, _ = fc.forecast_with_inputs(utc("2024-10-10"))
    ref = fc.forecast(utc("2024-10-10"))
    day = h[h["origin"] == utc("2024-10-10")].sort_values("horizon")
    np.testing.assert_allclose(day["p50_raw"], ref["q50"], atol=0.05)
    # naive = load one week earlier
    load = synthetic_frame.set_index("timestamp")["load_mw"]
    np.testing.assert_allclose(
        day["naive_168"], load.reindex(day["timestamp"] - pd.Timedelta("168h")), atol=0.05
    )
    run(fc, tmp_path, utc("2024-10-28 06:00"))
    assert len(read_log(tmp_path / "hourly.csv", HOURLY_COLS)) == len(h)  # no duplicates


def test_raw_recovered_for_rows_logged_before_raw_columns(artifact, synthetic_frame, tmp_path):
    fc = make(artifact, synthetic_frame, provisional_days=10)
    run(fc, tmp_path, utc("2024-10-20 06:00"))  # forecasts logged, few scored
    f = pd.read_csv(tmp_path / "forecasts.csv").drop(columns=["p10_raw", "p50_raw", "p90_raw"])
    f.to_csv(tmp_path / "forecasts.csv", index=False)  # as written by the Phase 5 code
    for name in ("scores", "hourly"):
        (tmp_path / f"{name}.csv").unlink()
    run(fc, tmp_path, utc("2024-10-27 06:00"))
    h = read_log(tmp_path / "hourly.csv", HOURLY_COLS)
    assert h["p50_raw"].notna().all()
    np.testing.assert_allclose(h["p50"] - h["p50_raw"], artifact.shifts["q50"], atol=0.11)


def test_fetch_stats_are_recorded_per_run(artifact, synthetic_frame, tmp_path):
    fc = make(artifact, synthetic_frame, provisional_days=10)
    stats = {"entsoe_requests": 7, "entsoe_retries": 1, "months_cached": 76}
    run(fc, tmp_path, utc("2024-10-27 06:00"), stats)
    run(fc, tmp_path, utc("2024-10-28 06:00"))  # e.g. an --offline build: no stats
    runs = read_log(tmp_path / "runs.csv", RUN_COLS)
    assert runs[["entsoe_requests", "entsoe_retries", "months_cached"]].iloc[0].tolist() == [
        7,
        1,
        76,
    ]
    assert runs["entsoe_requests"].iloc[1] != runs["entsoe_requests"].iloc[1]  # NaN
