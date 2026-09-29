"""Known-bad data periods, applied at model-load time.

Corrupted upstream data is MASKED (target set to NaN), never dropped:
- the hourly grid stays intact, so the row-based rolling windows in
  build_design_matrix (y.rolling(24), y.rolling(168)) keep meaning
  "24 hours" / "168 hours" -- dropping rows would silently stretch them;
- rolling stats (min_periods = window) and timestamp-reindexed lags turn
  masked hours into NaN automatically, so contamination is visible.

features.parquet is never modified: it records what ENTSO-E published.
Which data to trust is a modelling decision, kept in configs/ with a
reason and evidence for every period.

Scoring rule: an origin is contaminated if its read window
[origin - LOOKBACK_HOURS, origin + N_HORIZONS h) overlaps any exclusion.
Contaminated origins are not scored -- their error would mix model error
with the data defect. Deliberately conservative: lags are sparse, so the
builder does not read every hour of that window and a few flagged origins
may be technically clean. A simple interval rule beats squeezing out a few
extra origins.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.models.ml_features import (
    DEFAULT_LAGS,
    N_HORIZONS,
    TARGET_COL,
    daily_origins,
)

logger = logging.getLogger(__name__)

HOUR = pd.Timedelta("1h")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = PROJECT_ROOT / "configs" / "data_exclusions.toml"

# Longest row-based rolling window in build_design_matrix (roll168_mean).
ROLLING_MAX_HOURS = 168
# Furthest back any feature of an origin can read. If the builder gains a
# longer window, the soundness test in tests/test_exclusions.py fails.
LOOKBACK_HOURS = max(max(DEFAULT_LAGS), ROLLING_MAX_HOURS)


@dataclass(frozen=True)
class Exclusion:
    """Half-open [start, end); end=None means open-ended (still corrupt)."""

    start: pd.Timestamp
    end: pd.Timestamp | None
    reason: str
    evidence: str

    def mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        m = np.asarray(index >= self.start)
        if self.end is not None:
            m &= np.asarray(index < self.end)
        return m


def _utc(value: str, field: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        raise ValueError(
            f"exclusion {field}={value!r} has no timezone; write UTC with a trailing Z"
        )
    return ts.tz_convert("UTC")


def load_exclusions(path: str | Path = DEFAULT_PATH) -> list[Exclusion]:
    """Parse the exclusions TOML. Every period needs start, reason, evidence."""
    # utf-8-sig: Windows PowerShell may write a BOM, which tomllib rejects.
    text = Path(path).read_bytes().decode("utf-8-sig")
    cfg = tomllib.loads(text)
    out: list[Exclusion] = []
    for i, raw in enumerate(cfg.get("exclusion", [])):
        missing = [k for k in ("start", "reason", "evidence") if k not in raw]
        if missing:
            raise ValueError(f"exclusion #{i} is missing {missing}")
        start = _utc(raw["start"], "start")
        end = _utc(raw["end"], "end") if "end" in raw else None
        if end is not None and end <= start:
            raise ValueError(f"exclusion #{i}: end {end} is not after start {start}")
        out.append(Exclusion(start, end, raw["reason"], raw["evidence"]))
    return out


def apply_exclusions(
    frame: pd.DataFrame, exclusions: list[Exclusion], column: str = TARGET_COL
) -> pd.DataFrame:
    """Copy of `frame` with `column` set to NaN inside every exclusion."""
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        index = out.index
    else:
        index = pd.DatetimeIndex(out["timestamp"])
    total = np.zeros(len(out), dtype=bool)
    for ex in exclusions:
        m = ex.mask(index)
        end = "open" if ex.end is None else ex.end
        logger.info(
            "exclusion %s -> %s: masking %d hour(s) of %s", ex.start, end, int(m.sum()), column
        )
        total |= m
    out.loc[total, column] = np.nan
    return out


def contaminated_origins(
    origins: pd.DatetimeIndex,
    exclusions: list[Exclusion],
    lookback_hours: int = LOOKBACK_HOURS,
    horizon_hours: int = N_HORIZONS,
) -> np.ndarray:
    """True where [origin - lookback, origin + horizon) overlaps an exclusion.

    Interval overlap of [lo, hi) and [start, end): lo < end and start < hi.
    """
    lo = origins - lookback_hours * HOUR
    hi = origins + horizon_hours * HOUR
    bad = np.zeros(len(origins), dtype=bool)
    for ex in exclusions:
        hit = np.asarray(hi > ex.start)
        if ex.end is not None:
            hit &= np.asarray(lo < ex.end)
        bad |= hit
    return bad


def _demo() -> None:
    """Show the exclusions' effect on the real feature table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.set_index("timestamp")
    frame = frame.sort_index()

    exclusions = load_exclusions()
    for ex in exclusions:
        end = "open-ended" if ex.end is None else ex.end
        print(f"exclusion: {ex.start} -> {end}")
        print(f"  reason  : {ex.reason}")

    masked = apply_exclusions(frame, exclusions)
    newly = int(masked[TARGET_COL].isna().sum() - frame[TARGET_COL].isna().sum())
    print(f"\nhours newly masked : {newly}")

    origins = daily_origins(frame.index)
    bad = contaminated_origins(origins, exclusions)
    print(f"daily origins      : {len(origins)}")
    print(f"contaminated       : {int(bad.sum())}")
    print(f"lookback hours     : {LOOKBACK_HOURS}")

    in_aug = np.asarray((origins >= "2026-08-01") & (origins < "2026-09-01"))
    aug = origins[in_aug & ~bad]
    if len(aug):
        print(f"clean Aug origins  : {len(aug)} ({aug.min().date()} -> {aug.max().date()})")
    else:
        print("clean Aug origins  : 0")


if __name__ == "__main__":
    _demo()
