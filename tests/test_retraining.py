"""Retraining policy: what triggers a build and what blocks it, and the
champion/challenger decision on constructed paired logs (wait, promote,
reject, the end-of-trial tie rule, and that pre-cutoff origins never count)."""

import numpy as np
import pandas as pd
import pytest
from test_accuracy import hourly  # tests/ is on sys.path under pytest

from gridcast.monitoring.accuracy import MonitoringConfig
from gridcast.monitoring.retraining import RetrainingConfig, compare, daily_pinball, trigger

CFG = RetrainingConfig(
    max_unseen_days=28,
    min_new_days=7,
    persistent_amber_days=7,
    min_paired=14,
    max_paired=28,
    margin=0.02,
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
CHAMP = pd.Timestamp("2026-10-01", tz="UTC")


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def zones(days: int, coverage_zone="green", skill_zone="green", start="2026-11-01"):
    t = pd.date_range(utc(start) + pd.Timedelta("7h"), periods=days, freq="D")
    return pd.DataFrame({"run_at": t, "coverage_zone": coverage_zone, "skill_zone": skill_zone})


def drift(days: int, median_zone="green", data_zone="green", start="2026-11-01"):
    rows = []
    for t in pd.date_range(utc(start) + pd.Timedelta("7h"), periods=days, freq="D"):
        rows += [
            {
                "run_at": t,
                "family": "concept",
                "check": "median_error",
                "subject": "raw p50",
                "zone": median_zone,
            },
            {
                "run_at": t,
                "family": "concept",
                "check": "tail_rate",
                "subject": "below p10",
                "zone": "green",
            },
            {
                "run_at": t,
                "family": "data",
                "check": "late_revision",
                "subject": "load",
                "zone": data_zone,
            },
            {
                "run_at": t,
                "family": "feature",
                "check": "shift",
                "subject": "temperature_2m",
                "zone": "amber",
            },
        ]
    return pd.DataFrame(rows)


def test_quiet_and_young_champion_does_not_build():
    t = trigger(CHAMP, utc("2026-10-22 07:00"), zones(10), drift(10), CFG)
    assert not t.build
    assert t.unseen_days == 21 and t.candidate_cutoff == utc("2026-10-22")
    assert t.reasons == []  # feature-drift amber never triggers


def test_schedule_triggers():
    t = trigger(CHAMP, utc("2026-10-29 07:00"), zones(3), drift(3), CFG)
    assert t.build and t.reasons[0].startswith("schedule: 28")


def test_red_triggers_at_once_amber_only_when_persistent():
    t = trigger(CHAMP, utc("2026-10-15"), zones(1, coverage_zone="red_under"), drift(1), CFG)
    assert t.build and "accuracy coverage: red_under" in t.reasons

    six = trigger(CHAMP, utc("2026-10-15"), zones(6), drift(6, median_zone="amber"), CFG)
    assert not six.build
    seven = trigger(CHAMP, utc("2026-10-15"), zones(7), drift(7, median_zone="amber"), CFG)
    assert seven.build
    assert "concept raw p50: amber on 7 consecutive days" in seven.reasons


def test_manual_reruns_do_not_count_as_extra_days():
    d = drift(6, median_zone="amber")
    rerun = d.copy()
    rerun["run_at"] = rerun["run_at"] + pd.Timedelta("3h")  # second run the same days
    t = trigger(CHAMP, utc("2026-10-15"), zones(6), pd.concat([d, rerun]), CFG)
    assert not t.build  # still six distinct days


def test_red_data_check_and_too_little_new_data_block():
    t = trigger(
        CHAMP,
        utc("2026-11-05"),
        zones(1, coverage_zone="red_under"),
        drift(1, data_zone="red"),
        CFG,
    )
    assert not t.build and t.reasons and "late_revision" in t.blockers[0]
    t = trigger(CHAMP, utc("2026-10-04"), zones(1, coverage_zone="red_under"), drift(1), CFG)
    assert not t.build and "only 3 new final day(s)" in t.blockers[0]


def shifted(h: pd.DataFrame, scale: float, start=None) -> pd.DataFrame:
    """Copy whose quantile errors are scaled by `scale` (scale > 1: worse)."""
    out = h.copy()
    for c in ("p10", "p50", "p90"):
        out[c] = out["y"] + (out[c] - out["y"]) * scale
    if start is not None:
        out = out[out["origin"] >= start]
    return out


def test_daily_pinball_matches_hand_calculation():
    h = hourly(1, 24)  # p10 = y-50, p50 = y-100, p90 = y+50
    expected = (0.1 * 50 + 0.5 * 100 + 0.1 * 50) / 3
    assert daily_pinball(h)["pinball"].iloc[0] == pytest.approx(expected)


def test_compare_waits_then_promotes_a_non_inferior_challenger():
    champ = hourly(30, [24, 20, 18, 22], seed=0)
    chall = shifted(champ, 0.97)  # 3% better on every hour
    early = compare(champ, chall.head(10 * 24), utc("2026-10-01"), CFG, ACC)
    assert early.decision == "wait" and early.n_paired == 10
    # run daily, the trial decides on the day the 14th pair is scored
    c = compare(champ, chall.head(14 * 24), utc("2026-10-01"), CFG, ACC)
    assert c.decision == "promote" and c.n_paired == 14
    assert c.rel_diff == pytest.approx(-0.03, abs=0.005)


def test_compare_rejects_a_clearly_worse_challenger():
    champ = hourly(30, [24, 20, 18, 22], seed=0)
    c = compare(champ, shifted(champ, 1.2), utc("2026-10-01"), CFG, ACC)
    assert c.decision == "reject" and c.rel_band[0] > 0


def test_pre_cutoff_origins_never_count():
    champ = hourly(30, 20, seed=0)
    chall = shifted(champ, 0.97)
    c = compare(champ, chall, utc("2026-10-20"), CFG, ACC)  # only Oct 20-30 count
    assert c.decision == "wait" and c.n_paired == 11


def test_undecided_trial_ends_at_max_paired_with_the_tie_rule():
    rng = np.random.default_rng(5)
    champ = hourly(40, 20, seed=1)
    noisy = champ.copy()
    # equal on average, but noisy day to day: neither worse nor non-inferior
    day_scale = np.repeat(rng.choice([0.7, 1.3], size=40), 24)
    for c in ("p10", "p50", "p90"):
        noisy[c] = noisy["y"] + (noisy[c] - noisy["y"]) * day_scale
    mid = compare(champ, noisy.head(20 * 24), utc("2026-10-01"), CFG, ACC)
    assert mid.decision == "wait" and "undecided" in mid.reason
    end = compare(champ, noisy, utc("2026-10-01"), CFG, ACC)
    assert end.n_paired == 28 and end.decision in {"promote", "reject"}
    assert end.decision == ("promote" if end.rel_diff <= 0 else "reject")


def test_decide_force_and_trial_in_progress(tmp_path):
    from gridcast.monitoring.retraining import decide, read_challenger, register_challenger
    from gridcast.serving.daily_log import VINTAGE_COLS, write_log

    as_of = utc("2026-11-20 14:00")
    v = pd.DataFrame(
        [
            {
                "run_at": as_of,
                "data_as_of": as_of,
                "day": utc("2026-11-19"),
                "load_hours": 24,
                "load_mean_mw": 13000.0,
                "temp_mean_c": 8.0,
            }
        ],
        columns=VINTAGE_COLS,
    )
    write_log(v, tmp_path / "vintages.csv")
    quiet = decide(tmp_path, force=False, champion_cutoff=CHAMP)
    assert not quiet.build and quiet.candidate_cutoff == utc("2026-10-22")
    forced = decide(tmp_path, force=True, champion_cutoff=CHAMP)
    assert forced.build and forced.reasons == ["manual: forced from workflow_dispatch"]

    register_challenger(tmp_path, utc("2026-10-22"), "serving-2026-10-22", as_of)
    assert read_challenger(tmp_path)["status"] == "trial"
    again = decide(tmp_path, force=True, champion_cutoff=CHAMP)
    assert not again.build and again.blockers == ["trial in progress: serving-2026-10-22"]
