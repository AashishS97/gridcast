"""Data layer of the GridCast monitoring dashboard: read the log branch and
shape it for display. Pure pandas, no Streamlit, so it is unit-tested with the
rest of the project.

The dashboard computes nothing that the pipeline already decides: zones,
bands, trial verdicts and run health come precomputed from the daily job
(gridcast.monitoring.*). Here they are only selected and reshaped. One
definition of every metric, in the package; the dashboard cannot drift from
what the escalation logic sees.

Source: the public forecast-log branch over raw.githubusercontent.com, or a
local directory (GRIDCAST_LOG=path) for development and tests.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request

import pandas as pd

RAW_BASE = "https://raw.githubusercontent.com/AashishS97/gridcast/forecast-log/"
TIME_COLS = {
    "origin",
    "run_at",
    "data_as_of",
    "provisional_from",
    "last_servable",
    "day",
    "window_first",
    "window_last",
    "created_at",
    "run_started_at",
    "updated_at",
    "registered_at",
    "closed_at",
}

# Free-tier limits, as documented by each provider (checked 2026-11-26)
RENDER_FREE_HOURS = 750  # per workspace per calendar month
ENTSOE_REQUESTS_PER_MIN = 400  # per API token
STATUS_OK = {"green", "info"}


def source() -> str:
    return os.environ.get("GRIDCAST_LOG", RAW_BASE)


def read_text(name: str, src: str | None = None) -> str | None:
    """A file from the log, or None if it does not exist (yet)."""
    src = src or source()
    if src.startswith("http"):
        try:
            with urllib.request.urlopen(src + name, timeout=20) as resp:
                return resp.read().decode("utf-8-sig")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
    path = os.path.join(src, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


def read_csv(name: str, src: str | None = None) -> pd.DataFrame | None:
    text = read_text(name, src)
    if text is None:
        return None
    df = pd.read_csv(io.StringIO(text))
    for col in TIME_COLS & set(df.columns):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def read_json(name: str, src: str | None = None) -> dict | None:
    text = read_text(name, src)
    return json.loads(text) if text else None


# --- overview ---------------------------------------------------------------


def zone_level(zone: object) -> str:
    """green / amber / red / other, from the zone strings the pipeline writes
    (e.g. amber_under, red_over, amber_unprecedented)."""
    z = str(zone)
    for level in ("red", "amber", "green"):
        if z.startswith(level):
            return level
    return "other"


def overview(
    runs: pd.DataFrame | None,
    zones: pd.DataFrame | None,
    drift: pd.DataFrame | None,
    challenger: dict | None,
    trial_progress: pd.DataFrame | None,
    now: pd.Timestamp,
) -> dict:
    out: dict = {}
    if runs is not None and len(runs):
        last = runs.sort_values("run_at").iloc[-1]
        out["artifact"] = last["artifact"]
        out["data_as_of"] = last["data_as_of"]
        out["data_age_h"] = round((now - last["data_as_of"]).total_seconds() / 3600, 1)
        out["last_run_at"] = last["run_at"]
        out["last_run_status"] = last["status"]
        out["last_servable"] = last["last_servable"]
    if zones is not None and len(zones):
        z = zones.sort_values("run_at").iloc[-1]
        out["coverage"] = z["coverage"]
        out["coverage_zone"] = z["coverage_zone"]
        out["skill"] = z["skill"]
        out["skill_zone"] = z["skill_zone"]
    if drift is not None and len(drift):
        latest = drift[drift["run_at"] == drift["run_at"].max()]
        levels = latest["zone"].map(zone_level)
        out["drift_red"] = int((levels == "red").sum())
        out["drift_amber"] = int((levels == "amber").sum())
    if challenger:
        out["trial_tag"] = challenger.get("tag")
        out["trial_status"] = challenger.get("status")
        if trial_progress is not None and len(trial_progress):
            mine = trial_progress[trial_progress["tag"] == challenger.get("tag")]
            if len(mine):
                last = mine.iloc[-1]
                out["trial_paired"] = int(last["n_paired"])
                out["trial_decision"] = last["decision"]
    return out


# --- drift ------------------------------------------------------------------


def drift_latest(drift: pd.DataFrame) -> pd.DataFrame:
    latest = drift[drift["run_at"] == drift["run_at"].max()].copy()
    latest["level"] = latest["zone"].map(zone_level)
    return latest[["family", "check", "subject", "value", "lo", "hi", "zone", "level", "note"]]


def drift_history(drift: pd.DataFrame) -> pd.DataFrame:
    """Per day (its last run): how many checks were amber and red."""
    d = drift.copy()
    d["day"] = d["run_at"].dt.floor("D")
    last = d.groupby("day")["run_at"].transform("max")
    d = d[d["run_at"] == last]
    d["level"] = d["zone"].map(zone_level)
    d = d[d["level"].isin(["amber", "red"])]
    if d.empty:
        return pd.DataFrame(columns=["day", "level", "checks"])
    return d.groupby(["day", "level"]).size().rename("checks").reset_index()


def check_series(drift: pd.DataFrame, check: str, subject: str | None = None) -> pd.DataFrame:
    """One check's value over time, one point per run."""
    d = drift[drift["check"] == check]
    if subject is not None:
        d = d[d["subject"] == subject]
    return d[["run_at", "subject", "value", "lo", "hi", "zone"]].sort_values("run_at")


# --- run health and budget ----------------------------------------------------


def month_minutes(actions_runs: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Runner minutes per workflow, month to date."""
    start = now.floor("D") - pd.Timedelta(days=now.day - 1)
    r = actions_runs[actions_runs["created_at"] >= start]
    return r.groupby("workflow")["minutes"].agg(["count", "sum"]).reset_index()


def budget(
    actions_runs: pd.DataFrame | None, runs: pd.DataFrame | None, now: pd.Timestamp
) -> pd.DataFrame:
    """Free-tier headroom: what each resource allows, what we use."""
    days_in_month = pd.Period(now.strftime("%Y-%m")).days_in_month
    rows = []
    if actions_runs is not None and len(actions_runs):
        mm = month_minutes(actions_runs, now)
        used = f"{mm['sum'].sum():,.0f} min in {int(mm['count'].sum())} runs this month"
    else:
        used = "no run history yet"
    rows.append(
        {
            "resource": "GitHub Actions (standard runners)",
            "limit": "free and unlimited for public repositories",
            "used": used,
            "headroom": "unlimited while the repository is public",
        }
    )
    rows.append(
        {
            "resource": "Render free web service",
            "limit": f"{RENDER_FREE_HOURS} instance hours / month",
            "used": "not measured (needs a Render API key)",
            "headroom": f"at least {RENDER_FREE_HOURS - 24 * days_in_month} h even if never asleep "
            f"({24 * days_in_month} h in this month)",
        }
    )
    if runs is not None and "entsoe_requests" in runs and runs["entsoe_requests"].notna().any():
        r = runs.dropna(subset=["entsoe_requests"]).sort_values("run_at").iloc[-1]
        rows.append(
            {
                "resource": "ENTSO-E API",
                "limit": f"{ENTSOE_REQUESTS_PER_MIN} requests / minute per token",
                "used": f"{int(r['entsoe_requests'])} requests in the last run "
                f"({int(r['entsoe_retries'])} retries, {int(r['months_cached'])} months cached)",
                "headroom": "a whole run stays far below one minute's allowance",
            }
        )
    rows.append(
        {
            "resource": "Streamlit Community Cloud",
            "limit": "about 1 GB memory per app; sleeps after 12 h without traffic",
            "used": "this dashboard: a few small CSVs",
            "headroom": "first visit after a sleep waits for a restart",
        }
    )
    return pd.DataFrame(rows)
