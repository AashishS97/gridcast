"""Calendar features for load forecasting.

All features derive from LOCAL time (Europe/Amsterdam) because load is
driven by human routine on the local clock, while the index stays UTC.
This is the only place in the pipeline where local time appears, and it
appears transiently — tz_convert on a column, never a re-index.

Raw integer hour/dow (no sin/cos): cyclical encoding helps linear models
see that hour 23 neighbours hour 0; trees split on thresholds and don't
need it. School vacations are a known omission (NL staggers them across
three regions) — documented limitation, not an oversight.
"""

from __future__ import annotations

import logging

import holidays
import pandas as pd

logger = logging.getLogger(__name__)

LOCAL_TZ = "Europe/Amsterdam"


def add_calendar_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Append calendar feature columns derived from the UTC timestamp column."""
    if df[ts_col].dt.tz is None:
        raise ValueError(f"{ts_col!r} is tz-naive — GridCast rule: tz-aware UTC only")

    out = df.copy()
    local = out[ts_col].dt.tz_convert(LOCAL_TZ)

    out["hour"] = local.dt.hour.astype("int8")
    out["dow"] = local.dt.dayofweek.astype("int8")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["month"] = local.dt.month.astype("int8")
    out["day_of_year"] = local.dt.dayofyear.astype("int16")

    # +2 years: the day-ahead series reaches into tomorrow, which near
    # New Year's Eve is next year.
    years = list(range(int(local.dt.year.min()), int(local.dt.year.max()) + 2))
    nl_holidays = holidays.country_holidays("NL", years=years)
    out["is_holiday"] = local.dt.date.isin(set(nl_holidays.keys())).astype("int8")

    return out


def main() -> None:
    """Smoke run on real cleaned data: the evening peak must sit at local 18h."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_parquet("data/processed/actual_hourly.parquet")
    feat = add_calendar_features(df)
    logger.info("columns: %s", list(feat.columns))
    logger.info(
        "\nmean load (MW) by local hour:\n%s",
        feat.groupby("hour")["load_mw"].mean().round(0).to_string(),
    )
    logger.info(
        "\nmean load by day of week (0=Mon):\n%s",
        feat.groupby("dow")["load_mw"].mean().round(0).to_string(),
    )
    logger.info(
        "\nmean load, holiday vs not:\n%s",
        feat.groupby("is_holiday")["load_mw"].mean().round(0).to_string(),
    )


if __name__ == "__main__":
    main()
