"""Data quality checks for raw load series.

Checks DETECT problems and report them; they never modify data. Repair
happens explicitly in the cleaning module. All checks run on the UTC
index: every UTC day has 24 hours / 96 quarters, so DST cannot produce
false alarms here.
"""

from __future__ import annotations

import logging

import pandas as pd

from gridcast.data.base import LoadKind
from gridcast.data.fetch import load_raw

logger = logging.getLogger(__name__)

FREQ_15MIN = pd.Timedelta(minutes=15)

# Physical bounds for NL system load, MW.
#
# History lesson (found in this project's own data, 2026-07): an earlier
# 4 GW lower bound flagged 368 points that turned out to be REAL — midday
# net load collapsing below 4 GW (min observed: 327 MW) on summer 2026
# days, driven by rooftop solar. The fingerprint that proved it: long
# smooth runs, local hours 10-17, summer only, deepening year over year.
# Telemetry failures look nothing like that (random hours, abrupt, short).
# So the bound encodes only what remains physically impossible: load must
# be positive (a hard ~0 means telemetry death) and cannot exceed ~25 GW
# (a x1000 unit mix-up would). Known limitation: a night-time sag to a few
# hundred MW would pass this check; the spike and flatline checks are the
# backstop. A season/hour-conditional bound is the rigorous upgrade if
# ever needed.
LOAD_MW_LOW = 100.0
LOAD_MW_HIGH = 25_000.0

# Largest plausible change between consecutive 15-min values, MW.
MAX_STEP_MW = 2_000.0

FLATLINE_MIN_RUN = 8  # 8 x 15min = 2h of literally identical values


def find_gaps(df: pd.DataFrame, freq: pd.Timedelta = FREQ_15MIN) -> pd.DataFrame:
    """Runs of missing timestamps in an otherwise regular UTC grid.

    Returns columns: gap_start, gap_end, n_missing.
    """
    idx = pd.DatetimeIndex(df["timestamp"])
    full = pd.date_range(idx.min(), idx.max(), freq=freq)
    missing = full.difference(idx)
    if missing.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "n_missing"])
    s = pd.Series(missing)
    run_id = (s.diff() != freq).cumsum()
    out = s.groupby(run_id).agg(["min", "max", "size"])
    out.columns = ["gap_start", "gap_end", "n_missing"]
    return out.reset_index(drop=True)


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose timestamp occurs more than once (all copies returned)."""
    mask = df["timestamp"].duplicated(keep=False)
    return df.loc[mask].sort_values("timestamp")


def find_out_of_bounds(
    df: pd.DataFrame, low: float = LOAD_MW_LOW, high: float = LOAD_MW_HIGH
) -> pd.DataFrame:
    """Values outside physical bounds. NaN counts as out of bounds."""
    mask = ~df["load_mw"].between(low, high)
    return df.loc[mask]


def find_flatlines(df: pd.DataFrame, min_run: int = FLATLINE_MIN_RUN) -> pd.DataFrame:
    """Runs of >= min_run consecutive IDENTICAL values.

    Night load is stable but never bit-identical for hours; a long run of
    the exact same MW value means a stuck meter or upstream fill-forward.
    Returns: run_start, run_end, value, n_points.
    """
    run_id = (df["load_mw"].diff() != 0).cumsum()
    g = df.assign(run=run_id).groupby("run")
    out = g.agg(
        run_start=("timestamp", "min"),
        run_end=("timestamp", "max"),
        value=("load_mw", "first"),
        n_points=("timestamp", "size"),
    )
    return out[out["n_points"] >= min_run].reset_index(drop=True)


def find_spikes(df: pd.DataFrame, max_step: float = MAX_STEP_MW) -> pd.DataFrame:
    """Jumps larger than max_step between CONSECUTIVE timestamps.

    The contiguity guard matters: the difference across a gap is
    meaningless and would otherwise produce phantom spikes.
    """
    step = df["load_mw"].diff().abs()
    contiguous = df["timestamp"].diff() == FREQ_15MIN
    mask = (step > max_step) & contiguous
    out = df.loc[mask].copy()
    out["step_mw"] = step[mask]
    return out


def quality_report(df: pd.DataFrame, name: str) -> dict[str, pd.DataFrame]:
    """Run all checks, log a summary, return the findings per check."""
    checks = {
        "gaps": find_gaps(df),
        "duplicates": find_duplicates(df),
        "out_of_bounds": find_out_of_bounds(df),
        "flatlines": find_flatlines(df),
        "spikes": find_spikes(df),
    }
    logger.info("=== quality report: %s ===", name)
    logger.info(
        "rows %d | %s -> %s",
        len(df),
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    for check_name, result in checks.items():
        logger.info("%-14s %d finding(s)", check_name, len(result))
        if 0 < len(result) <= 12:
            logger.info("\n%s\n", result.to_string())
    return checks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for kind in LoadKind:
        quality_report(load_raw(kind), kind.value)


if __name__ == "__main__":
    main()
