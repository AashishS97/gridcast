"""Cleaning: raw 15-min series -> regular hourly UTC series.

The rules below are the spec written by the quality investigation
(2026-07), not generic defaults:

1. Deduplicate, reindex to the full 15-min UTC grid (gaps become NaN).
2. ACTUAL only: single-point sags -> NaN. A point qualifies iff both
   neighbouring steps exceed MAX_STEP_MW with opposite signs (down-and-
   back). Persistent steps (down-and-stay, e.g. 2026-06-25 17:45) are
   real events and are kept.
3. Interpolate NaN runs up to MAX_INTERP_QUARTERS (2h), time-weighted.
   Longer runs stay NaN — we bridge telemetry blips, we don't invent
   half a day of load. NB: pandas interpolate(limit=n) would instead
   fill the FIRST n points of arbitrarily long gaps, hence the mask.
4. Hourly = mean of the 4 quarters (MW is average power; mean is the
   physically correct aggregate), and all 4 quarters are required —
   resample().mean() would otherwise silently average partial hours.
5. DAY_AHEAD is a covariate: published values are taken as-is (no
   outlier repair), because the model must train on exactly what it
   will receive at prediction time. Steps 1, 3, 4 only.
6. Every repair is logged with timestamps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gridcast.data.base import LoadKind
from gridcast.data.fetch import load_raw
from gridcast.data.quality import FREQ_15MIN, MAX_STEP_MW

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MAX_INTERP_QUARTERS = 8  # 2 hours


def to_full_grid(df: pd.DataFrame) -> pd.Series:
    """Dedupe and reindex onto the complete 15-min UTC grid."""
    s = df.drop_duplicates("timestamp", keep="last").set_index("timestamp")["load_mw"].sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq=FREQ_15MIN)
    return s.reindex(full)


def flag_single_point_outliers(s: pd.Series, max_step: float = MAX_STEP_MW) -> pd.DatetimeIndex:
    """Indices of down-and-back (or up-and-back) single-point excursions.

    Both neighbouring steps must exceed max_step AND have opposite signs.
    A persistent level shift fails the second-step condition and is kept.
    NaN neighbours (gap edges) disqualify a point — we never diagnose
    across missing data.
    """
    prev_step = s.diff()
    next_step = s.shift(-1) - s
    mask = (prev_step.abs() > max_step) & (next_step.abs() > max_step) & (prev_step * next_step < 0)
    return s.index[mask.fillna(False)]


def interpolate_short_gaps(s: pd.Series, max_run: int = MAX_INTERP_QUARTERS) -> pd.Series:
    """Time-interpolate NaN runs of length <= max_run; leave longer runs."""
    na = s.isna()
    run_id = (na != na.shift()).cumsum()
    run_size = na.groupby(run_id).transform("size")
    fillable = na & (run_size <= max_run)
    filled = s.interpolate(method="time", limit_area="inside")
    return s.where(~fillable, filled)


def to_hourly(s: pd.Series) -> pd.Series:
    """Hourly mean requiring all 4 quarters; else the hour is NaN."""
    grouped = s.resample("1h")
    hourly = grouped.mean()
    hourly[grouped.count() < 4] = float("nan")
    return hourly


def clean_series(df: pd.DataFrame, kind: LoadKind) -> pd.DataFrame:
    """Full pipeline for one series: raw rows -> hourly UTC DataFrame."""
    s = to_full_grid(df)
    n_missing = int(s.isna().sum())

    if kind is LoadKind.ACTUAL:
        outliers = flag_single_point_outliers(s)
        if len(outliers):
            logger.info(
                "%s: repairing %d single-point outlier(s) at %s",
                kind.value,
                len(outliers),
                [str(t) for t in outliers],
            )
        s.loc[outliers] = float("nan")

    s = interpolate_short_gaps(s)
    hourly = to_hourly(s)
    hourly = hourly.loc[: hourly.last_valid_index()]  # drop trailing partial hour

    n_nan_hours = int(hourly.isna().sum())
    logger.info(
        "%s: %d grid quarters (%d were missing) -> %d hourly rows, %d NaN hour(s) remain",
        kind.value,
        len(s),
        n_missing,
        len(hourly),
        n_nan_hours,
    )
    return hourly.rename("load_mw").rename_axis("timestamp").reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for kind in LoadKind:
        hourly = clean_series(load_raw(kind), kind)
        path = PROCESSED_DIR / f"{kind.value}_hourly.parquet"
        hourly.to_parquet(path, index=False)
        logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
