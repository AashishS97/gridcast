"""Phase 4 step 0: has ENTSO-E revised the corrupt NL actual load?

READ-ONLY: fetches into memory; never writes data/raw or features.parquet
(Phase 3's numbers were produced from the current cache and must stay
reproducible). June and August are controls: they were clean, so they
should show no revision and a YoY ratio near 1.

Check 1 (revision): fresh values vs the cached month, same timestamps.
Check 2 (level): fresh daily mean vs the 2023-25 mean for the same ISO
week + weekday -- the YoY test from phase3_data_state.py, which flagged
July (~0.65) and September (~0.73).

Run:  uv run python notebooks/phase4_recheck_entsoe.py
"""

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from gridcast.data.base import LoadKind, NoDataError
from gridcast.data.entsoe import EntsoeClient
from gridcast.data.fetch import RAW_DIR, load_raw

MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]
CONTROLS = {"2026-06", "2026-08"}
MIN_QUARTERS_PER_DAY = 88  # of 96 (92/100 on DST days): skip partial days
LOW, HIGH = 0.85, 1.15


def daily_mean(df: pd.DataFrame) -> pd.DataFrame:
    s = df.drop_duplicates("timestamp").set_index("timestamp")["load_mw"].astype(float)
    g = s.resample("1D")
    out = pd.DataFrame({"load_mw": g.mean(), "n": g.count()})
    return out[out["n"] >= MIN_QUARTERS_PER_DAY]


def with_week_keys(d: pd.DataFrame) -> pd.DataFrame:
    iso = d.index.isocalendar()
    return d.assign(week=iso["week"].astype(int).values, dow=iso["day"].astype(int).values)


def main() -> None:
    load_dotenv()
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise SystemExit("ENTSOE_API_KEY not set")
    client = EntsoeClient(api_key=key)
    now = pd.Timestamp.now(tz="UTC").floor("h")

    hist = with_week_keys(daily_mean(load_raw(LoadKind.ACTUAL)))
    hist = hist[hist.index.year <= 2025]
    ref = hist.groupby(["week", "dow"])["load_mw"].mean().rename("ref_mw")
    print(f"reference: {len(hist)} full days 2023-2025, {len(ref)} week/weekday cells\n")

    fresh_all: list[pd.DataFrame] = []
    print("=== check 1: revision vs cache ===")
    for m in MONTHS:
        start = pd.Timestamp(f"{m}-01", tz="UTC")
        end = min(start + pd.offsets.MonthBegin(1), now)
        tag = " (control)" if m in CONTROLS else ""
        try:
            fresh = client.fetch_load(LoadKind.ACTUAL, start.to_pydatetime(), end.to_pydatetime())
        except NoDataError as exc:
            print(f"{m}{tag}: no data from ENTSO-E ({exc})")
            continue
        fresh_all.append(fresh)
        path = RAW_DIR / LoadKind.ACTUAL.value / f"{m}.parquet"
        if not path.exists():
            print(f"{m}{tag}: {len(fresh)} fresh rows, nothing cached to compare")
            continue
        cached = pd.read_parquet(path)
        both = (
            cached.drop_duplicates("timestamp")
            .set_index("timestamp")[["load_mw"]]
            .join(
                fresh.drop_duplicates("timestamp").set_index("timestamp")[["load_mw"]],
                lsuffix="_old",
                rsuffix="_new",
                how="inner",
            )
            .dropna()
        )
        diff = (both["load_mw_new"] - both["load_mw_old"]).abs()
        ratio = both["load_mw_new"].mean() / both["load_mw_old"].mean()
        print(
            f"{m}{tag}: cached {len(cached)} rows, fresh {len(fresh)}, common {len(both)}, "
            f"changed >1 MW: {int((diff > 1).sum())}, max change {diff.max():,.1f} MW, "
            f"mean new/old {ratio:.3f}"
        )

    if not fresh_all:
        raise SystemExit("no fresh data fetched")
    fresh = pd.concat(fresh_all, ignore_index=True)
    d = with_week_keys(daily_mean(fresh)).join(ref, on=["week", "dow"])
    d["yoy"] = d["load_mw"] / d["ref_mw"]
    d["month"] = d.index.strftime("%Y-%m")

    print("\n=== check 2: YoY level (fresh vs 2023-25 same ISO week + weekday) ===")
    summ = d.groupby("month")["yoy"].agg(days="count", median="median", min="min", max="max")
    print(summ.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\nweekly median YoY:")
    print(d["yoy"].resample("W-SUN").median().to_string(float_format=lambda v: f"{v:.2f}"))

    for label, flag in [
        (f"LOW  (yoy < {LOW})", d["yoy"] < LOW),
        (f"HIGH (yoy > {HIGH})", d["yoy"] > HIGH),
    ]:
        print(f"\nruns flagged {label}:")
        hits = d[flag]
        if hits.empty:
            print("  none")
            continue
        runs = (flag != flag.shift()).cumsum()
        for _, g in hits.groupby(runs[flag]):
            print(
                f"  {g.index.min().date()} -> {g.index.max().date()}  "
                f"({len(g)} days, median yoy {g['yoy'].median():.2f})"
            )
    print(f"\nlast fresh timestamp: {fresh['timestamp'].max()}")


if __name__ == "__main__":
    main()
