"""Accuracy tracking on hand-built hourly logs: per-day arithmetic, day-level
bootstrap bands, zone logic at its boundaries, the per-hour breakdown, the
report/zone files, and the red-zone exit code."""

import numpy as np
import pandas as pd
import pytest

from gridcast.monitoring.accuracy import (
    MonitoringConfig,
    band,
    by_horizon,
    coverage_zone,
    daily_table,
    rolling_table,
    run,
    skill_zone,
    window_summary,
)
from gridcast.serving.daily_log import HOURLY_COLS, write_log

CFG = MonitoringConfig(
    window_origins=28,
    min_origins=14,
    n_boot=2000,
    seed=1,
    nominal_coverage=0.8,
    amber_level=0.95,
    red_level=0.999,
)
MANIFEST = {
    "calibration": {
        "n_origins": 60,
        "first_origin": "2026-08-02T00:00:00+00:00",
        "last_origin": "2026-09-30T00:00:00+00:00",
        "pool_q50_mae_mw": 246.6,
        "pool_raw_interval_coverage": 0.642,
    }
}
GATE = {"fold_ids": list(range(14)), "metrics": {"mae_q50": 256.9, "interval_cov": 0.655}}


def hourly(n_days: int, hits_per_day, err: float = 100.0, naive_err: float = 200.0, seed=0):
    """n_days x 24 hours. Interval [y-50, y+50] around the truth, then the
    first `hits_per_day[d]` hours of day d are inside, the rest above p90."""
    rng = np.random.default_rng(seed)
    hits = np.resize(np.asarray(hits_per_day), n_days)  # cycle the pattern
    rows = []
    for d in range(n_days):
        origin = pd.Timestamp("2026-10-01", tz="UTC") + pd.Timedelta(days=d)
        ts = pd.date_range(origin, periods=24, freq="h")
        y = 12000 + rng.normal(0, 300, 24)
        p50 = y - err
        p10, p90 = y - 50.0, y + 50.0
        outside = np.arange(24) >= hits[d]
        p10 = np.where(outside, y - 400, p10)
        p90 = np.where(outside, y - 300, p90)
        rows.append(
            pd.DataFrame(
                {
                    "origin": origin,
                    "timestamp": ts,
                    "horizon": np.arange(1, 25),
                    "y": y,
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                    "p10_raw": p10 - 5,
                    "p50_raw": p50 - 100,
                    "p90_raw": p90 - 200,
                    "naive_168": y + naive_err,
                    "artifact": "synthetic",
                    "scored_at": pd.Timestamp("2026-11-20", tz="UTC"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)[HOURLY_COLS]


def test_daily_table_arithmetic():
    d = daily_table(hourly(3, [24, 12, 0]))
    assert d["coverage"].tolist() == [1.0, 0.5, 0.0]
    assert d["mae_p50"].to_numpy() == pytest.approx([100.0] * 3)
    assert d["naive_mae"].to_numpy() == pytest.approx([200.0] * 3)
    # raw p50 = served - 100 -> |y - (y - 200)| = 200
    assert d["raw_mae_p50"].to_numpy() == pytest.approx([200.0] * 3)


def test_band_is_deterministic_and_brackets_the_mean():
    s = window_summary(daily_table(hourly(20, [24, 18, 16, 20])), CFG)
    lo, hi = s["coverage_band"]
    assert lo <= s["coverage"] <= hi
    assert window_summary(daily_table(hourly(20, [24, 18, 16, 20])), CFG)["coverage_band"] == (
        lo,
        hi,
    )
    rlo, rhi = s["coverage_band_red"]
    assert rlo <= lo and hi <= rhi  # the stricter level is wider


def test_day_resampling_is_wider_than_an_hourly_binomial():
    # days that fail together: 24/24 or 12/24 -> strong within-day correlation
    d = daily_table(hourly(28, [24, 12]))
    s = window_summary(d, CFG)
    p, n_hours = s["coverage"], 28 * 24
    binomial_halfwidth = 1.96 * np.sqrt(p * (1 - p) / n_hours)
    lo, hi = s["coverage_band"]
    assert (hi - lo) / 2 > 1.5 * binomial_halfwidth


@pytest.mark.parametrize(
    ("point", "amber", "red", "zone"),
    [
        (0.82, (0.75, 0.89), (0.70, 0.92), "green"),
        (0.80, (0.80, 0.85), (0.78, 0.88), "green"),  # boundary inclusive
        (0.70, (0.65, 0.78), (0.60, 0.83), "amber_under"),
        (0.95, (0.90, 0.99), (0.79, 1.00), "amber_over"),
        (0.60, (0.55, 0.66), (0.50, 0.71), "red_under"),
    ],
)
def test_coverage_zone(point, amber, red, zone):
    assert coverage_zone(point, amber, red, 0.8) == zone


@pytest.mark.parametrize(
    ("amber", "red", "zone"),
    [
        ((0.10, 0.40), (0.02, 0.50), "green"),
        ((-0.05, 0.20), (-0.10, 0.30), "amber"),
        ((-0.30, -0.05), (-0.40, 0.01), "amber"),  # below 0, not yet decisive
        ((-0.30, -0.10), (-0.40, -0.02), "red"),
    ],
)
def test_skill_zone(amber, red, zone):
    assert skill_zone(amber, red) == zone


def test_window_takes_the_trailing_origins_and_zones_only_when_enough():
    d = daily_table(hourly(40, 24))
    s = window_summary(d, CFG)
    assert s["n_origins"] == 28
    assert s["window_first"] == pd.Timestamp("2026-10-13", tz="UTC")
    assert s["skill"] == pytest.approx(0.5)
    assert s["skill_zone"] == "green"
    small = window_summary(d.head(10), CFG)
    assert small["status"] == "insufficient"
    assert small["coverage_zone"] == "insufficient"


def test_rolling_table_starts_at_min_origins():
    r = rolling_table(daily_table(hourly(20, [24, 18])), CFG)
    assert len(r) == 20 - CFG.min_origins + 1
    assert r["n_origins"].tolist() == list(range(14, 21))


def test_by_horizon_shape_and_values():
    h = by_horizon(hourly(10, 12), CFG)  # hours 0-11 inside, 12-23 outside
    assert h["horizon"].tolist() == list(range(1, 25))
    assert h["coverage"].tolist() == [1.0] * 12 + [0.0] * 12
    assert (h["mae_lo"] <= h["mae_p50"]).all() and (h["mae_p50"] <= h["mae_hi"]).all()


def test_run_writes_report_and_zone_history(tmp_path):
    write_log(hourly(20, [24, 20, 16, 22]), tmp_path / "hourly.csv")
    now = pd.Timestamp("2026-11-20 08:00", tz="UTC")
    assert run(tmp_path, MANIFEST, GATE, CFG, now) == 0
    assert run(tmp_path, MANIFEST, GATE, CFG, now + pd.Timedelta("1D")) == 0
    text = (tmp_path / "ACCURACY.md").read_text(encoding="utf-8")
    assert text.startswith("# GridCast accuracy")
    assert "246.6" in text and "256.9" in text
    zones = pd.read_csv(tmp_path / "zones.csv")
    assert len(zones) == 2
    assert zones["n_origins"].tolist() == [20, 20]


def test_red_coverage_exits_2_after_writing(tmp_path):
    write_log(hourly(20, 8), tmp_path / "hourly.csv")  # 33% coverage every day
    assert run(tmp_path, MANIFEST, GATE, CFG, pd.Timestamp("2026-11-20", tz="UTC")) == 2
    assert pd.read_csv(tmp_path / "zones.csv")["coverage_zone"].iloc[-1] == "red_under"
    assert (tmp_path / "ACCURACY.md").exists()


def test_empty_log_is_not_an_error(tmp_path):
    assert run(tmp_path, MANIFEST, GATE, CFG, pd.Timestamp("2026-11-20", tz="UTC")) == 0
    assert "No scored origins" in (tmp_path / "ACCURACY.md").read_text(encoding="utf-8")


def test_band_percentiles():
    assert band(np.arange(1001, dtype=float), 0.9) == pytest.approx((50.0, 950.0))
