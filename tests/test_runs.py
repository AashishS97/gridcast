"""Run health from a synthetic Actions run list: parsing, merging by run id,
schedule delay against the cron, and days with no scheduled run."""

import re
from pathlib import Path

import pandas as pd
import pytest

from gridcast.monitoring.runs import (
    DAILY_CRON_UTC,
    DAILY_WORKFLOW,
    daily_health,
    merge_runs,
    parse_runs,
)


def run(id_, name, event, start, minutes, conclusion="success", status="completed"):
    t0 = pd.Timestamp(start, tz="UTC")
    return {
        "id": id_,
        "name": name,
        "event": event,
        "status": status,
        "conclusion": conclusion,
        "created_at": t0.isoformat(),
        "run_started_at": t0.isoformat(),
        "updated_at": (t0 + pd.Timedelta(minutes=minutes)).isoformat(),
        "run_attempt": 1,
        "head_sha": "abcdef1234567890",
    }


PAYLOAD = {
    "workflow_runs": [
        run(1, DAILY_WORKFLOW, "schedule", "2026-11-24 07:40", 12),  # 23 min late
        run(2, "Retrain challenger", "workflow_run", "2026-11-24 07:53", 0.3),
        run(3, "CI", "push", "2026-11-24 10:00", 4, conclusion="failure"),
        # 2026-11-25: no scheduled run at all, only a manual one
        run(4, DAILY_WORKFLOW, "workflow_dispatch", "2026-11-25 14:00", 25),
        run(
            5,
            DAILY_WORKFLOW,
            "schedule",
            "2026-11-26 15:40",
            3,
            conclusion=None,
            status="in_progress",
        ),
    ]
}


def test_parse_runs():
    r = parse_runs(PAYLOAD)
    assert len(r) == 5 and r["head_sha"].iloc[0] == "abcdef123456"
    assert r.loc[r["id"] == 1, "minutes"].item() == 12.0
    assert r.loc[r["id"] == 5, "conclusion"].item() == ""


def test_merge_keeps_history_and_latest_state():
    old = parse_runs(PAYLOAD)
    later = run(5, DAILY_WORKFLOW, "schedule", "2026-11-26 15:40", 14)
    new = parse_runs({"workflow_runs": [later, run(6, "CI", "push", "2026-11-27 09:00", 5)]})
    m = merge_runs(old, new)
    assert m["id"].tolist() == [1, 2, 3, 4, 5, 6]  # run 1-4 kept beyond the API's window
    assert m.loc[m["id"] == 5, "conclusion"].item() == "success"
    assert m.loc[m["id"] == 5, "minutes"].item() == 14.0


def test_daily_health():
    h = daily_health(parse_runs(PAYLOAD)).set_index("day")
    d24, d25, d26 = (pd.Timestamp(f"2026-11-{d}", tz="UTC") for d in (24, 25, 26))
    assert h.loc[d24, "scheduled_daily"] and h.loc[d24, "schedule_delay_min"] == 23.0
    assert h.loc[d24, ["retrain_runs", "ci_runs"]].tolist() == [1, 1]
    assert h.loc[d24, "runner_minutes"] == pytest.approx(16.3)
    assert not h.loc[d25, "scheduled_daily"] and pd.isna(h.loc[d25, "schedule_delay_min"])
    assert h.loc[d25, "daily_ok"]  # the manual run still produced the day's log
    assert h.loc[d26, "schedule_delay_min"] == 503.0  # 8 h 23 min late
    assert not h.loc[d26, "daily_ok"]  # still running


def test_cron_constant_matches_the_workflow():
    text = (Path(__file__).parents[1] / ".github" / "workflows" / "daily.yml").read_text()
    minute, hour = re.search(r'cron: "(\d+) (\d+) \* \* \*"', text).groups()
    assert (int(hour), int(minute)) == DAILY_CRON_UTC


def test_health_starts_when_the_daily_workflow_first_ran():
    early_ci = run(9, "CI", "push", "2026-11-20 10:00", 3)
    h = daily_health(parse_runs({"workflow_runs": [early_ci, *PAYLOAD["workflow_runs"]]}))
    assert h["day"].min() == pd.Timestamp("2026-11-24", tz="UTC")
