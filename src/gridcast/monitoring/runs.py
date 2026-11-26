"""Run health: did the pipeline run when it should, how long did it take,
and what did it cost? From the GitHub Actions run history, which the daily
job fetches with its own token (gh api .../actions/runs) and hands to this
module, so the dashboard never calls an API itself.

  actions_runs.csv  every workflow run seen so far (the API returns the last
                    100; merging keeps the history beyond that), one row per
                    run: workflow, trigger, result, start, end, minutes.
  run_health.csv    per UTC day: did the SCHEDULED daily run happen, how late
                    did it start against its cron, did any daily run succeed,
                    how many runs and runner-minutes each workflow used.

Why "late" matters: GitHub runs schedules best-effort -- under load a
scheduled run starts late or is dropped. Nothing fails when that happens;
the log simply gets no new day. Only a record of when runs actually started
makes a silent skip visible.

Minutes are wall time from start to last update, a close upper bound on
runner time for these single-job-at-a-time workflows. Standard runners are
free on public repositories, so minutes are budget headroom, not money.

Run:  gh api "repos/OWNER/REPO/actions/runs?per_page=100" > runs.json
      uv run python -m gridcast.monitoring.runs --json runs.json --log-dir log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DAILY_WORKFLOW = "Daily shadow forecast"
DAILY_CRON_UTC = (7, 17)  # .github/workflows/daily.yml: cron "17 7 * * *"
RUN_COLS = [
    "id",
    "workflow",
    "event",
    "status",
    "conclusion",
    "created_at",
    "run_started_at",
    "updated_at",
    "attempt",
    "head_sha",
    "minutes",
]
HEALTH_COLS = [
    "day",
    "scheduled_daily",
    "schedule_delay_min",
    "daily_runs",
    "daily_ok",
    "retrain_runs",
    "ci_runs",
    "runner_minutes",
]
_TIMES = ["created_at", "run_started_at", "updated_at"]


def parse_runs(payload: dict) -> pd.DataFrame:
    """The REST API's workflow_runs list, reduced to what run health needs."""
    rows = [
        {
            "id": r["id"],
            "workflow": r.get("name", ""),
            "event": r.get("event", ""),
            "status": r.get("status", ""),
            "conclusion": r.get("conclusion") or "",
            "created_at": r.get("created_at"),
            "run_started_at": r.get("run_started_at") or r.get("created_at"),
            "updated_at": r.get("updated_at"),
            "attempt": r.get("run_attempt", 1),
            "head_sha": (r.get("head_sha") or "")[:12],
        }
        for r in payload.get("workflow_runs", [])
    ]
    df = pd.DataFrame(rows, columns=RUN_COLS[:-1])
    for c in _TIMES:
        df[c] = pd.to_datetime(df[c], utc=True)
    df["minutes"] = ((df["updated_at"] - df["run_started_at"]).dt.total_seconds() / 60).round(1)
    return df[RUN_COLS]


def merge_runs(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union by run id; a run seen again (finished since, re-run) keeps its
    latest state."""
    both = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
    both = both.sort_values("updated_at").drop_duplicates("id", keep="last")
    return both.sort_values("created_at").reset_index(drop=True)


def daily_health(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per UTC day from the first run to the last."""
    if runs.empty:
        return pd.DataFrame(columns=HEALTH_COLS)
    r = runs.copy()
    r["day"] = r["created_at"].dt.floor("D")
    # from the first day the daily workflow ran at all: earlier days (CI only)
    # would count as "missed" runs of a workflow that did not exist yet
    daily_days = r.loc[r["workflow"] == DAILY_WORKFLOW, "day"]
    first = daily_days.min() if len(daily_days) else r["day"].min()
    days = pd.date_range(first, r["day"].max(), freq="D")
    out = []
    for day in days:
        d = r[r["day"] == day]
        daily = d[d["workflow"] == DAILY_WORKFLOW]
        sched = daily[daily["event"] == "schedule"]
        due = day + pd.Timedelta(hours=DAILY_CRON_UTC[0], minutes=DAILY_CRON_UTC[1])
        delay = (
            round((sched["run_started_at"].min() - due).total_seconds() / 60, 1)
            if len(sched)
            else float("nan")
        )
        out.append(
            {
                "day": day,
                "scheduled_daily": bool(len(sched)),
                "schedule_delay_min": delay,
                "daily_runs": len(daily),
                "daily_ok": bool((daily["conclusion"] == "success").any()),
                "retrain_runs": int((d["workflow"] == "Retrain challenger").sum()),
                "ci_runs": int((d["workflow"] == "CI").sum()),
                "runner_minutes": round(float(d["minutes"].sum()), 1),
            }
        )
    return pd.DataFrame(out, columns=HEALTH_COLS)


def read_runs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RUN_COLS)
    df = pd.read_csv(path, dtype={"head_sha": str, "conclusion": str})
    df["conclusion"] = df["conclusion"].fillna("")
    for c in _TIMES:
        df[c] = pd.to_datetime(df[c], utc=True)
    return df[RUN_COLS]


def _write(df: pd.DataFrame, path: Path, time_cols: list[str]) -> None:
    out = df.copy()
    for c in time_cols:
        out[c] = pd.to_datetime(out[c], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="run history and run health")
    ap.add_argument("--json", type=Path, required=True, help="gh api .../actions/runs output")
    ap.add_argument("--log-dir", type=Path, required=True)
    args = ap.parse_args()
    new = parse_runs(json.loads(args.json.read_text(encoding="utf-8")))
    runs = merge_runs(read_runs(args.log_dir / "actions_runs.csv"), new)
    _write(runs, args.log_dir / "actions_runs.csv", _TIMES)
    health = daily_health(runs)
    _write(health, args.log_dir / "run_health.csv", ["day"])
    print(health.tail(14).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
