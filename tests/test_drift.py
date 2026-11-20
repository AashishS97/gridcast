"""Drift and data checks on constructed series and ledgers: the seasonal
anomaly is leave-one-year-out, an ordinary year stays green while a shifted
one does not, the support check counts out-of-range hours, and each data
check fires on exactly the defect it is for."""

import numpy as np
import pandas as pd
import pytest
from test_accuracy import hourly  # tests/ is on sys.path under pytest

from gridcast.monitoring.accuracy import MonitoringConfig
from gridcast.monitoring.drift import (
    DriftConfig,
    anomalies,
    data_checks,
    residual_drift,
    run,
    seasonal_drift,
)
from gridcast.serving.daily_log import VINTAGE_COLS, write_log

CFG = DriftConfig(
    window_days=28,
    clim_halfwidth_days=15,
    season_halfwidth_days=45,
    min_valid_frac=0.85,
    shift_central=0.90,
    support_amber=0.05,
    rel_tol=0.05,
    level_ratio_red=0.85,
    weather_lag_amber_days=10,
)
ACC = MonitoringConfig(
    window_origins=28,
    min_origins=14,
    n_boot=2000,
    seed=1,
    nominal_coverage=0.8,
    amber_level=0.95,
    red_level=0.999,
)
REF_END = pd.Timestamp("2026-01-01", tz="UTC")


def seasonal(offset_2026: float = 0.0, noise: float = 1.0, seed: int = 0) -> pd.Series:
    """2023-2026 hourly: annual + daily cycle + day-to-day weather noise, and an
    optional offset for 2026 (after the reference end)."""
    ts = pd.date_range("2023-01-01", "2026-06-30 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    doy, hour = ts.dayofyear.to_numpy(), ts.hour.to_numpy()
    day_noise = np.repeat(rng.normal(0, noise, len(ts) // 24 + 1), 24)[: len(ts)]
    x = 10 + 8 * np.sin(2 * np.pi * doy / 365.25) + 3 * np.sin(2 * np.pi * hour / 24) + day_noise
    x = x + np.where(ts.year == 2026, offset_2026, 0.0)
    return pd.Series(x, index=ts)


def test_anomalies_are_leave_one_year_out():
    # each reference year is a constant: 0, 3, 6. Leave-one-year-out against
    # the other two gives -4.5, 0, +4.5; including itself would give -3, 0, 3
    ts = pd.date_range("2023-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
    x = pd.Series(3.0 * (ts.year - 2023), index=ts, dtype=float)
    anom, lo, hi = anomalies(x, REF_END, 15)
    by_year = anom.groupby(anom.index.year).mean()
    assert by_year.to_numpy() == pytest.approx([-4.5, 0.0, 4.5])
    assert np.nanmin(lo) == 0.0 and np.nanmax(hi) == 6.0


def test_an_ordinary_season_is_green_and_a_shifted_one_is_not():
    end = pd.Timestamp("2026-03-01", tz="UTC")
    ok = seasonal_drift(seasonal(0.0), REF_END, end, CFG, "feature", "x")
    shift = {r["check"]: r for r in ok}
    assert shift["shift"]["zone"] == "green"
    assert shift["support"]["zone"] == "green"

    hot = {r["check"]: r for r in seasonal_drift(seasonal(4.0), REF_END, end, CFG, "feature", "x")}
    assert hot["shift"]["zone"].startswith("amber")
    assert hot["shift"]["value"] == pytest.approx(4.0, abs=1.0)
    assert hot["support"]["zone"] == "amber"  # +4 pushes many hours past the seasonal max


def test_drift_window_without_enough_data_is_insufficient():
    x = seasonal()
    x[x.index >= pd.Timestamp("2026-02-10", tz="UTC")] = np.nan
    r = seasonal_drift(x, REF_END, pd.Timestamp("2026-03-01", tz="UTC"), CFG, "feature", "x")
    assert r[0]["zone"] == "insufficient"


def ledger(runs: dict[str, dict[str, tuple[int, float]]], temp_until: str) -> pd.DataFrame:
    """runs: run date -> {day: (hours, mean)}; data_as_of = run date 06:00."""
    rows = []
    for run_day, days in runs.items():
        as_of = pd.Timestamp(run_day, tz="UTC") + pd.Timedelta("6h")
        for day, (hours, mean) in days.items():
            d = pd.Timestamp(day, tz="UTC")
            rows.append(
                {
                    "run_at": as_of + pd.Timedelta("1h"),
                    "data_as_of": as_of,
                    "day": d,
                    "load_hours": hours,
                    "load_mean_mw": mean,
                    "temp_mean_c": 10.0 if d <= pd.Timestamp(temp_until, tz="UTC") else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=VINTAGE_COLS)


def healthy_days(run_day: str, provisional_low: bool = True) -> dict[str, tuple[int, float]]:
    """60 days to run_day: final days 13000 MW, the last 29 days 25% low."""
    end = pd.Timestamp(run_day)
    out = {}
    for age in range(59, -1, -1):
        d = end - pd.Timedelta(days=age)
        low = provisional_low and age <= 29
        out[f"{d:%Y-%m-%d}"] = (24 if age else 8, 13000 * (0.75 if low else 1.0))
    return out


def by_check(rows):
    return {r["check"]: r for r in rows}


def test_healthy_ledger_is_green_and_shows_the_provisional_regime():
    v = ledger({"2026-11-20": healthy_days("2026-11-20")}, temp_until="2026-11-14")
    r = by_check(data_checks(v, 29, CFG))
    assert r["late_revision"]["zone"] == "insufficient"  # one vintage
    assert r["final_level"]["zone"] == "green"
    assert r["final_gaps"]["zone"] == "green"
    assert r["weather_lag"]["value"] == 6 and r["weather_lag"]["zone"] == "green"
    assert r["provisional"]["value"] == pytest.approx(0.75)


def test_revision_on_schedule_is_not_late_but_a_final_day_moving_is():
    day1 = healthy_days("2026-11-20")
    day2 = healthy_days("2026-11-21")  # day 2026-10-22 crosses age 29 -> 30 and is revised
    v = ledger({"2026-11-20": day1, "2026-11-21": day2}, "2026-11-14")
    r = by_check(data_checks(v, 29, CFG))
    assert r["late_revision"]["zone"] == "green"
    assert r["revisions_seen"]["lo"] == 29  # revised at age 29: on schedule

    late = dict(day2)
    late["2026-10-10"] = (24, 13000 * 1.2)  # age 41 at the first run: final, then moved
    r = by_check(
        data_checks(ledger({"2026-11-20": day1, "2026-11-21": late}, "2026-11-14"), 29, CFG)
    )
    assert r["late_revision"]["zone"] == "red"
    assert "2026-10-10" in r["late_revision"]["note"]


def test_missing_revision_and_gaps_and_weather_lag():
    days = healthy_days("2026-11-20")
    for age in range(30, 40):  # ten final days never revised
        d = pd.Timestamp("2026-11-20") - pd.Timedelta(days=age)
        days[f"{d:%Y-%m-%d}"] = (24 if age != 35 else 20, 13000 * 0.75)
    r = by_check(data_checks(ledger({"2026-11-20": days}, "2026-11-01"), 29, CFG))
    assert r["final_level"]["zone"] == "red"
    assert r["final_gaps"]["zone"] == "amber" and "(20h)" in r["final_gaps"]["note"]
    assert r["weather_lag"]["zone"] == "amber"


def test_residual_drift_tails_and_median():
    h = hourly(20, 24)  # every hour inside [p10, p90]; p50 = y - 100, raw p50 = y - 200
    r = residual_drift(h, {"q50": 200.0}, ACC)
    tails = [x for x in r if x["check"] == "tail_rate"]
    assert [t["value"] for t in tails] == [0.0, 0.0]
    assert all(t["zone"].startswith("red") for t in tails)  # too wide, decisively
    med = [x for x in r if x["check"] == "median_error"][0]
    assert med["value"] == pytest.approx(200.0)
    assert med["zone"] == "green"  # matches the pool's median error
    assert by_check(residual_drift(h, {"q50": 50.0}, ACC))["median_error"]["zone"] == "red"


def test_run_writes_report_history_and_exit_code(tmp_path):
    write_log(
        ledger({"2026-11-20": healthy_days("2026-11-20")}, "2026-11-14"), tmp_path / "vintages.csv"
    )
    manifest = {
        "train_cutoff": "2026-10-01T00:00:00+00:00",
        "calibration": {"shifts_mw": {"q10": 5.5, "q50": 109.9, "q90": 236.9}},
    }
    now = pd.Timestamp("2026-11-20 08:00", tz="UTC")
    assert run(tmp_path, None, manifest, 29, CFG, ACC, now) == 0
    text = (tmp_path / "DRIFT.md").read_text(encoding="utf-8")
    assert "## Data checks (vintage ledger)" in text
    hist = pd.read_csv(tmp_path / "drift.csv")
    assert set(hist["family"]) == {"concept", "data"}

    days = healthy_days("2026-11-20")
    for age in range(30, 44):
        d = pd.Timestamp("2026-11-20") - pd.Timedelta(days=age)
        days[f"{d:%Y-%m-%d}"] = (24, 13000 * 0.75)
    write_log(ledger({"2026-11-20": days}, "2026-11-14"), tmp_path / "vintages.csv")
    assert run(tmp_path, None, manifest, 29, CFG, ACC, now) == 2


def test_run_with_a_feature_table_checks_every_variable(tmp_path):
    from gridcast.monitoring.drift import DRIFT_VARIABLES

    f = pd.DataFrame({v: seasonal(seed=i) for i, v in enumerate(DRIFT_VARIABLES)})
    f.loc[f.index > pd.Timestamp("2026-06-24", tz="UTC"), "temperature_2m"] = np.nan  # archive lag
    f = f.rename_axis("timestamp").reset_index()
    manifest = {
        "train_cutoff": "2026-01-01T00:00:00+00:00",
        "calibration": {"shifts_mw": {"q10": 0.0, "q50": 0.0, "q90": 0.0}},
    }
    code = run(tmp_path, f, manifest, 29, CFG, ACC, pd.Timestamp("2026-07-01", tz="UTC"))
    assert code == 0
    hist = pd.read_csv(tmp_path / "drift.csv")
    shift = hist[hist["check"] == "shift"].set_index("subject")
    assert list(shift.index) == DRIFT_VARIABLES
    assert shift.loc["load_mw", "family"] == "target"
    assert (shift["zone"] == "green").all()
    # the load window ends at the provisional line, temperature at its own last day
    assert "to 2026-05-31" in shift.loc["load_mw", "note"]  # whole days before the line
    assert "to 2026-06-23" in shift.loc["temperature_2m", "note"]
