"""GridCast monitoring dashboard (Streamlit).

Reads the forecast-log branch (see data.py) and shows what the daily job
already decided: accuracy and coverage with their uncertainty bands, drift
and data checks, the champion/challenger trial, run health and free-tier
budget. It computes no metric of its own.

Run locally:  uv run --with streamlit streamlit run dashboard/app.py
              (GRIDCAST_LOG=path/to/log reads a local checkout instead)
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402

# reference palette (dataviz skill): categorical slots 1-3, status colours,
# secondary ink for reference lines; status never carries meaning alone
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
BAND = "#519ced"  # blue step 350: a band that reads on light and dark surfaces
INK2 = "#9c9c9c"  # reference lines: secondary ink that stays visible on dark
STATUS = {"green": "#0aac23", "amber": "#fbb315", "red": "#dc3c3d"}
ICON = {"green": "🟢", "amber": "🟠", "red": "🔴", "other": "⚪"}

st.set_page_config(page_title="GridCast monitoring", page_icon="⚡", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load() -> dict:
    csvs = [
        "runs",
        "zones",
        "rolling",
        "by_horizon",
        "drift",
        "trials",
        "trial_progress",
        "run_health",
        "actions_runs",
    ]
    out = {name: data.read_csv(f"{name}.csv") for name in csvs}
    out["challenger"] = data.read_json("challenger.json")
    out["loaded_at"] = pd.Timestamp.now(tz="UTC")
    return out


def day_axis(field: str) -> alt.X:
    """Daily data on a time axis: label days, not Vega's default hour ticks."""
    return alt.X(field, title="", axis=alt.Axis(format="%b %d", labelAngle=0, tickCount="day"))


def rule(y: float, dash: bool = True) -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"y": [y]}))
        .mark_rule(color=INK2, strokeDash=[4, 4] if dash else [], strokeWidth=1)
        .encode(y="y:Q")
    )


def band_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    lo: str,
    hi: str,
    title: str,
    fmt: str = ",.1f",
    ref: float | None = None,
) -> alt.Chart:
    base = alt.Chart(df).encode(x=day_axis(f"{x}:T"))
    tip = [
        alt.Tooltip(f"{x}:T", title="window ends"),
        alt.Tooltip(f"{y}:Q", format=fmt),
        alt.Tooltip(f"{lo}:Q", format=fmt, title="band low"),
        alt.Tooltip(f"{hi}:Q", format=fmt, title="band high"),
    ]
    area = base.mark_area(color=BAND, opacity=0.25).encode(
        y=alt.Y(f"{lo}:Q", title=title, axis=alt.Axis(format=fmt), scale=alt.Scale(zero=False)),
        y2=f"{hi}:Q",
    )
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(y=f"{y}:Q")
    pts = base.mark_point(color=BLUE, size=60, filled=True, opacity=0).encode(
        y=f"{y}:Q", tooltip=tip
    )
    layers = [area, line, pts] + ([rule(ref)] if ref is not None else [])
    return alt.layer(*layers).properties(height=260)


d = load()
now = pd.Timestamp.now(tz="UTC")
ov = data.overview(d["runs"], d["zones"], d["drift"], d["challenger"], d["trial_progress"], now)

st.title("GridCast monitoring")
st.caption(
    "Day-ahead probabilistic forecasts of Dutch electricity load, scored in shadow "
    "against ENTSO-E actuals once they are final (~30 days later). Everything here is "
    "read from the `forecast-log` branch the daily job writes; zones and verdicts are "
    f"decided there, not in this page. Loaded {d['loaded_at']:%Y-%m-%d %H:%M} UTC."
)

# --- overview ---------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Champion", str(ov.get("artifact", "-")).replace("lgbm_quantile_", "cutoff "))
c2.metric(
    "Data as of",
    f"{ov['data_as_of']:%Y-%m-%d %H:%M}" if "data_as_of" in ov else "-",
    help="Newest actual load published by ENTSO-E at the last run",
)
if "coverage" in ov:
    lvl = data.zone_level(ov["coverage_zone"])
    c3.metric(
        "80% coverage (trailing)",
        f"{ov['coverage']:.1%}",
        help="Share of hours inside the served p10-p90 interval, trailing 28 origins",
    )
    c3.caption(f"{ICON[lvl]} zone: {ov['coverage_zone']}")
    lvl = data.zone_level(ov["skill_zone"])
    c4.metric(
        "Skill vs seasonal naive",
        f"{ov['skill']:.2f}",
        help="1 - MAE(p50) / MAE(load one week earlier); above 0 beats naive",
    )
    c4.caption(f"{ICON[lvl]} zone: {ov['skill_zone']}")
if "trial_tag" in ov:
    paired = ov.get("trial_paired", 0)
    c5.metric(
        "Challenger trial",
        f"{paired} / 14 pairs" if ov["trial_status"] == "trial" else ov["trial_status"],
    )
    c5.caption(f"{ov['trial_tag']}: {ov.get('trial_decision', ov['trial_status'])}")
if "last_run_at" in ov:
    st.caption(
        f"Last daily run {ov['last_run_at']:%Y-%m-%d %H:%M} UTC, status `{ov['last_run_status']}`; "
        f"drift and data checks on that run: {ov.get('drift_red', 0)} red, "
        f"{ov.get('drift_amber', 0)} amber. Newest servable origin "
        f"{ov['last_servable']:%Y-%m-%d}."
    )

tabs = st.tabs(
    ["Accuracy", "Coverage by hour", "Drift & data", "Retraining", "Run health & budget"]
)

# --- accuracy ----------------------------------------------------------------
with tabs[0]:
    r = d["rolling"]
    if r is None or r.empty:
        st.info("Not enough scored origins for a trailing window yet.")
    else:
        st.markdown(
            "Each point is the trailing 28-origin window ending on that origin. Shaded: 95% "
            "band from resampling whole days (a forecast's 24 hours fail together, so hours "
            "are not independent trials). Zones follow the Basel traffic-light idea: green "
            "while the band is consistent with the target."
        )
        a, b = st.columns(2)
        with a:
            st.subheader("80% interval coverage")
            st.altair_chart(
                band_line(
                    r,
                    "origin",
                    "coverage",
                    "coverage_lo",
                    "coverage_hi",
                    "coverage",
                    ".0%",
                    ref=0.8,
                ),
                width="stretch",
            )
            st.caption("Dashed: the 80% target.")
        with b:
            st.subheader("p50 MAE (MW)")
            st.altair_chart(
                band_line(r, "origin", "mae_p50", "mae_lo", "mae_hi", "MW"),
                width="stretch",
            )
        st.subheader("Skill against the seasonal naive (load one week earlier)")
        st.altair_chart(
            band_line(r, "origin", "skill", "skill_lo", "skill_hi", "skill", ".2f", ref=0.0),
            width="stretch",
        )
        st.caption(
            "Dashed: 0, no better than naive. The model sees observed weather (ERA5), the "
            "naive sees none, so this skill is an upper bound on what forecast weather allows."
        )
        with st.expander("Table"):
            st.dataframe(r, width="stretch", hide_index=True)

# --- coverage by hour ----------------------------------------------------------
with tabs[1]:
    h = d["by_horizon"]
    if h is None or h.empty:
        st.info("No scored origins yet.")
    else:
        n = int(h["n_origins"].iloc[0])
        st.markdown(
            f"All {n} scored origins. Origins are midnight UTC, so horizon *h* is UTC hour "
            "*h*-1: hour-of-day and horizon cannot be told apart here. The single conformal "
            "shift per quantile calibrates the total, not each hour: in daylight hours "
            "(rooftop solar) coverage sits below 80%, at night above."
        )
        base = alt.Chart(h).encode(x=alt.X("utc_hour:O", title="UTC hour"))
        tip = [
            alt.Tooltip("utc_hour:O", title="UTC hour"),
            alt.Tooltip("coverage:Q", format=".0%"),
            alt.Tooltip("coverage_lo:Q", format=".0%", title="band low"),
            alt.Tooltip("coverage_hi:Q", format=".0%", title="band high"),
        ]
        cov = alt.layer(
            base.mark_rule(color=BLUE, strokeWidth=2).encode(
                y=alt.Y("coverage_lo:Q", title="coverage", axis=alt.Axis(format=".0%")),
                y2="coverage_hi:Q",
            ),
            base.mark_point(color=BLUE, filled=True, size=70).encode(y="coverage:Q", tooltip=tip),
            rule(0.8),
        ).properties(height=280)
        st.subheader("80% coverage per hour (95% band)")
        st.altair_chart(cov, width="stretch")
        tip = [
            alt.Tooltip("utc_hour:O", title="UTC hour"),
            alt.Tooltip("mae_p50:Q", format=",.0f", title="MAE"),
            alt.Tooltip("mae_lo:Q", format=",.0f", title="band low"),
            alt.Tooltip("mae_hi:Q", format=",.0f", title="band high"),
        ]
        mae = alt.layer(
            base.mark_rule(color=BLUE, strokeWidth=2).encode(
                y=alt.Y("mae_lo:Q", title="MW"), y2="mae_hi:Q"
            ),
            base.mark_point(color=BLUE, filled=True, size=70).encode(y="mae_p50:Q", tooltip=tip),
        ).properties(height=280)
        st.subheader("p50 MAE per hour (MW, 95% band)")
        st.altair_chart(mae, width="stretch")

# --- drift --------------------------------------------------------------------
with tabs[2]:
    dr = d["drift"]
    if dr is None or dr.empty:
        st.info("No drift checks logged yet.")
    else:
        st.markdown(
            "Feature and target drift compare the recent window with the **same season** in "
            "the training years (as anomalies from the seasonal average), and are amber at "
            "most: unusual inputs are a reason to look, not proof of worse forecasts. Concept "
            "drift shows only in the residuals. Data checks read the daily ledger of what "
            "ENTSO-E published, to catch a late or missing revision."
        )
        latest = data.drift_latest(dr)
        latest.insert(0, "", latest["level"].map(ICON))
        st.subheader(f"Latest checks ({dr['run_at'].max():%Y-%m-%d %H:%M} UTC)")
        st.dataframe(latest.drop(columns="level"), width="stretch", hide_index=True)

        hist = data.drift_history(dr)
        if len(hist):
            st.subheader("Amber and red checks per day")
            chart = (
                alt.Chart(hist)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=day_axis("day:T"),
                    y=alt.Y("checks:Q", title="checks", stack=True),
                    color=alt.Color(
                        "level:N",
                        title="zone",
                        scale=alt.Scale(
                            domain=["amber", "red"], range=[STATUS["amber"], STATUS["red"]]
                        ),
                    ),
                    tooltip=["day:T", "level:N", "checks:Q"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, width="stretch")

        prov = data.check_series(dr, "provisional")
        if len(prov):
            st.subheader("Provisional under-publication")
            chart = alt.layer(
                alt.Chart(prov)
                .mark_line(color=BLUE, strokeWidth=2, point=True)
                .encode(
                    x=day_axis("run_at:T"),
                    y=alt.Y(
                        "value:Q",
                        title="young days / first final days",
                        scale=alt.Scale(zero=False),
                    ),
                    tooltip=[
                        alt.Tooltip("run_at:T", title="run"),
                        alt.Tooltip("value:Q", format=".2f"),
                    ],
                ),
                rule(1.0),
            ).properties(height=220)
            st.altair_chart(chart, width="stretch")
            st.caption(
                "Mean load of days aged 1-29 over the first final days. Well below 1: ENTSO-E "
                "is publishing provisional values low, as the 29-day gate assumes."
            )

# --- retraining ------------------------------------------------------------------
with tabs[3]:
    ch = d["challenger"]
    tp = d["trial_progress"]
    st.markdown(
        "A challenger is the same recipe at a newer cutoff. None of its out-of-sample days "
        "is final when it is built, so it is tried **prospectively**: both models forecast "
        "every new origin and each pair is scored once final. Promote when the band on "
        "the relative pinball difference ends below +2% (non-inferior); reject when it lies "
        "above 0. Promotion is a pull request a person merges."
    )
    if ch:
        st.write(
            f"**Current challenger:** `{ch['tag']}` (cutoff {ch['cutoff'][:10]}), "
            f"status **{ch['status']}**" + (f": {ch['reason']}" if ch.get("reason") else "")
        )
    if tp is not None and len(tp) and ch:
        mine = tp[(tp["tag"] == ch["tag"]) & tp["rel_diff"].notna()]
        if len(mine):
            st.subheader("Challenger vs champion, mean pinball")
            base = alt.Chart(mine).encode(x=alt.X("n_paired:Q", title="paired origins"))
            tip = [
                alt.Tooltip("n_paired:Q", title="pairs"),
                alt.Tooltip("rel_diff:Q", format="+.1%", title="relative difference"),
                alt.Tooltip("rel_lo:Q", format="+.1%", title="band low"),
                alt.Tooltip("rel_hi:Q", format="+.1%", title="band high"),
                alt.Tooltip("decision:N"),
            ]
            chart = alt.layer(
                base.mark_area(color=BAND, opacity=0.25).encode(
                    y=alt.Y(
                        "rel_lo:Q", title="challenger / champion - 1", axis=alt.Axis(format="+.0%")
                    ),
                    y2="rel_hi:Q",
                ),
                base.mark_line(color=BLUE, strokeWidth=2, point=True).encode(
                    y="rel_diff:Q", tooltip=tip
                ),
                rule(0.0, dash=False),
                rule(0.02),
            ).properties(height=260)
            st.altair_chart(chart, width="stretch")
            st.caption("Solid: 0 (equal). Dashed: +2% margin; promote once the band ends below it.")
        else:
            last = tp[tp["tag"] == ch["tag"]].tail(1)
            if len(last):
                st.info(f"{last['reason'].item()}")
    trials = d["trials"]
    if trials is not None and len(trials):
        st.subheader("Trial history")
        shown = trials.copy()
        for col in ("cutoff", "registered_at", "closed_at"):
            if col in shown:
                shown[col] = pd.to_datetime(shown[col], utc=True).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(shown, width="stretch", hide_index=True)

# --- run health & budget ------------------------------------------------------------
with tabs[4]:
    rh, ar = d["run_health"], d["actions_runs"]
    st.markdown(
        "GitHub runs schedules best-effort: a scheduled run can start late or not at all, "
        "and nothing fails when that happens. The daily job is due at 07:17 UTC."
    )
    if rh is not None and len(rh):
        # today is still open: its scheduled run may simply not have started yet
        past = rh[rh["day"] < now.floor("D")]
        st.write(
            f"**Completed days without a scheduled daily run:** "
            f"{int((~past['scheduled_daily']).sum())} of {len(past)}; with no successful "
            f"daily run: {int((~past['daily_ok']).sum())}. Today is excluded until it ends."
        )
        sched = rh.dropna(subset=["schedule_delay_min"])
        if len(sched):
            st.subheader("Start delay of the scheduled daily run (minutes)")
            chart = (
                alt.Chart(sched)
                .mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=day_axis("day:T"),
                    y=alt.Y("schedule_delay_min:Q", title="minutes late"),
                    tooltip=[
                        alt.Tooltip("day:T"),
                        alt.Tooltip("schedule_delay_min:Q", title="min"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, width="stretch")
    if ar is not None and len(ar):
        st.subheader("Runner minutes per day")
        ar = ar.assign(day=ar["created_at"].dt.floor("D"))
        per = ar.groupby(["day", "workflow"])["minutes"].sum().reset_index()
        order = ["Daily shadow forecast", "Retrain challenger", "CI"]
        chart = (
            alt.Chart(per)
            .mark_bar()
            .encode(
                x=day_axis("day:T"),
                y=alt.Y("minutes:Q", title="minutes", stack=True),
                color=alt.Color(
                    "workflow:N",
                    scale=alt.Scale(domain=order, range=[BLUE, ORANGE, AQUA]),
                    title="workflow",
                ),
                tooltip=["day:T", "workflow:N", alt.Tooltip("minutes:Q", format=",.1f")],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")
    runs = d["runs"]
    if runs is not None and "entsoe_requests" in runs and runs["entsoe_requests"].notna().any():
        st.subheader("ENTSO-E requests per daily run")
        rr = runs.dropna(subset=["entsoe_requests"])
        chart = (
            alt.Chart(rr)
            .mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=day_axis("run_at:T"),
                y=alt.Y("entsoe_requests:Q", title="requests"),
                tooltip=[
                    alt.Tooltip("run_at:T", title="run"),
                    "entsoe_requests:Q",
                    "entsoe_retries:Q",
                    "months_cached:Q",
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Since the raw-data cache: only months that can still change are fetched.")
    st.subheader("Free-tier budget")
    st.dataframe(data.budget(ar, runs, now), width="stretch", hide_index=True)
