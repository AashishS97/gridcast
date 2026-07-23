"""Shared types for load data sources.

Any source (ENTSO-E, energy-charts.info, ...) implements LoadSource, so the
rest of the pipeline never knows which concrete source it's talking to.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Protocol

import pandas as pd


class LoadKind(str, Enum):
    """Which load series we're fetching."""

    ACTUAL = "actual"  # measured system load (processType A16)
    DAY_AHEAD = "day_ahead"  # TSO's own day-ahead forecast (processType A01)


class DataSourceError(RuntimeError):
    """The source returned something we can't work with."""


class NoDataError(DataSourceError):
    """The request was valid but the source has no data for this window."""


class LoadSource(Protocol):
    """Anything that can produce a load time series.

    Contract: fetch_load returns a DataFrame with columns
      - timestamp: tz-aware UTC, sorted, unique
      - load_mw:   float, MW (average power over the interval)
    at the source's NATIVE resolution. Sources never resample; that is
    the cleaning module's job.
    """

    def fetch_load(self, kind: LoadKind, start: datetime, end: datetime) -> pd.DataFrame: ...
