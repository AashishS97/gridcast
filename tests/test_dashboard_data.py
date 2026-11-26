"""Dashboard data layer: reading the log from a directory (missing files are
None, not errors), zone levels, the overview, drift reshaping, budget rows."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "dashboard"))
import data  # noqa: E402

NOW = pd.Timestamp("2026-11-26 12:00", tz="UTC")


def utc(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture
def log(tmp_path):
    pd.DataFrame(
        {
            "run_at": ["2026-11-25T07:30:00Z", "2026-11-26T07:30:00Z"],
            "artifact": ["lgbm_quantile_2026-10-01"] * 2,
            "data_as_of": ["2026-11-25T05:00:00Z", "2026-11-26T06:00:00Z"],
            "last_servable": ["2026-10-27T00:00:00Z", "2026-10-28T00:00:00Z"],
            "status": ["ok", "ok"],
            "entsoe_requests": [None, 6],
            "entsoe_retries": [None, 1],
            "months_cached": [None, 76],
        }
    ).to_csv(tmp_path / "runs.csv", index=False)
    pd.DataFrame(
        {
            "run_at": ["2026-11-26T07:31:00Z"],
            "coverage": [0.796],
            "coverage_zone": ["green"],
            "skill": [0.52],
            "skill_zone": ["green"],
        }
    ).to_csv(tmp_path / "zones.csv", index=False)
    rows = []
    for t, zones in (
        ("2026-11-25T07:32:00Z", ["green", "amber", "red"]),
        ("2026-11-26T07:32:00Z", ["amber_unprecedented", "green", "info"]),
    ):
        for i, z in enumerate(zones):
            rows.append(
                {
                    "run_at": t,
                    "family": "data",
                    "check": f"c{i}",
                    "subject": "load",
                    "value": 0.84 + i,
                    "lo": None,
                    "hi": None,
                    "zone": z,
                    "note": "",
                }
            )
    rows.append(
        {
            "run_at": "2026-11-26T07:32:00Z",
            "family": "data",
            "check": "provisional",
            "subject": "load",
            "value": 0.84,
            "lo": None,
            "hi": None,
            "zone": "info",
            "note": "",
        }
    )
    pd.DataFrame(rows).to_csv(tmp_path / "drift.csv", index=False)
    (tmp_path / "challenger.json").write_text(
        json.dumps({"tag": "serving-2026-10-27", "status": "trial"})
    )
    pd.DataFrame(
        {
            "run_at": ["2026-11-26T07:33:00Z"],
            "tag": ["serving-2026-10-27"],
            "n_paired": [1],
            "rel_diff": [None],
            "decision": ["wait"],
            "reason": ["1 paired origin(s), need 14"],
        }
    ).to_csv(tmp_path / "trial_progress.csv", index=False)
    return str(tmp_path)


def test_reading_and_missing_files(log):
    runs = data.read_csv("runs.csv", log)
    assert isinstance(runs["run_at"].dtype, pd.DatetimeTZDtype)
    assert data.read_csv("nope.csv", log) is None
    assert data.read_json("challenger.json", log)["tag"] == "serving-2026-10-27"


@pytest.mark.parametrize(
    ("zone", "level"),
    [
        ("green", "green"),
        ("amber_under", "amber"),
        ("amber_unprecedented", "amber"),
        ("red_over", "red"),
        ("info", "other"),
        ("insufficient", "other"),
        (float("nan"), "other"),
    ],
)
def test_zone_level(zone, level):
    assert data.zone_level(zone) == level


def test_overview(log):
    ov = data.overview(
        *(data.read_csv(f, log) for f in ("runs.csv", "zones.csv", "drift.csv")),
        data.read_json("challenger.json", log),
        data.read_csv("trial_progress.csv", log),
        NOW,
    )
    assert ov["data_as_of"] == utc("2026-11-26 06:00") and ov["data_age_h"] == 6.0
    assert (ov["coverage_zone"], ov["skill"]) == ("green", 0.52)
    assert (ov["drift_red"], ov["drift_amber"]) == (0, 1)  # latest run only
    assert (ov["trial_paired"], ov["trial_decision"]) == (1, "wait")


def test_drift_reshaping(log):
    dr = data.read_csv("drift.csv", log)
    assert len(data.drift_latest(dr)) == 4
    hist = data.drift_history(dr)
    assert hist.set_index(["day", "level"])["checks"].to_dict() == {
        (utc("2026-11-25"), "amber"): 1,
        (utc("2026-11-25"), "red"): 1,
        (utc("2026-11-26"), "amber"): 1,
    }
    assert data.check_series(dr, "provisional")["value"].tolist() == [0.84]


def test_budget(log):
    runs = data.read_csv("runs.csv", log)
    actions = pd.DataFrame(
        {
            "workflow": ["CI", "CI"],
            "minutes": [4.0, 5.0],
            "created_at": [utc("2026-10-30"), utc("2026-11-02")],
        }
    )
    b = data.budget(actions, runs, NOW).set_index("resource")
    assert b.loc["GitHub Actions (standard runners)", "used"].startswith("5 min in 1 runs")
    assert "at least 30 h" in b.loc["Render free web service", "headroom"]  # 750 - 24 x 30
    assert (
        "6 requests in the last run (1 retries, 76 months cached)" in b.loc["ENTSO-E API", "used"]
    )
