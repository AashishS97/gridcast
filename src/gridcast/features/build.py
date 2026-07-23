"""Join target, covariates, and calendar into one feature table.

The TARGET (actual hourly load) defines the row spine; day-ahead and
weather LEFT-join onto it. Missing covariates stay as visible NaN —
whether to drop such rows is a modelling decision, not a pipeline one,
so the pipeline reports coverage instead of silently shrinking the data.
"""

from __future__ import annotations

import logging

import pandas as pd

from gridcast.features.calendar import add_calendar_features

logger = logging.getLogger(__name__)


def build_feature_table(
    actual: pd.DataFrame, day_ahead: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """actual/day_ahead: (timestamp, load_mw); weather: (timestamp, vars...)."""
    df = actual.copy()
    da = day_ahead.rename(columns={"load_mw": "da_forecast_mw"})
    df = df.merge(da, on="timestamp", how="left")
    df = df.merge(weather, on="timestamp", how="left")
    df = add_calendar_features(df)

    n = len(df)
    for col in ["da_forecast_mw", "temperature_2m"]:
        cov = 100.0 * df[col].notna().sum() / n
        logger.info("coverage %-16s %.1f%%", col, cov)
    logger.info("feature table: %d rows, %d columns", n, df.shape[1])
    return df
