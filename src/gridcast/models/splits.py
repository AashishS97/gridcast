"""Time-series splitting: final holdout + rolling-origin folds.

Random K-fold CV is invalid for forecasting: hourly load is strongly
autocorrelated, so randomly held-out hours have near-duplicate neighbours in
the training set and CV measures interpolation, while production requires
extrapolation into an unseen (and drifting) future. Everything here is
strictly chronological.

Conventions:
- All intervals are half-open: train = [train_start, origin),
  test = [origin, test_end).
- Timestamps must be UTC (or at least a fixed-offset zone): DST makes
  local-time arithmetic lie about what "24 hours" means.
- Data is assumed hourly.

Every model in Phase 2 is evaluated on the same frozen folds so the
comparison table is fair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOUR = pd.Timedelta("1h")


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold.

    train = [train_start, origin), test = [origin, test_end).
    The origin is the first timestamp being forecast.
    """

    fold_id: int
    train_start: pd.Timestamp
    origin: pd.Timestamp
    test_end: pd.Timestamp

    def train_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index >= self.train_start) & (index < self.origin))

    def test_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index >= self.origin) & (index < self.test_end))


def _validate_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"expected DatetimeIndex, got {type(index).__name__}")
    if len(index) == 0:
        raise ValueError("index is empty")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")
    if index.has_duplicates:
        raise ValueError(
            "index has duplicate timestamps — classic symptom of local-time "
            "data around DST fall-back; fix upstream, use UTC"
        )


def holdout_cutoff(
    index: pd.DatetimeIndex, *, test_span: str | pd.Timedelta = "56D"
) -> pd.Timestamp:
    """Midnight timestamp separating development data from the final test set.

    dev = index < cutoff, test = index >= cutoff. The test set is the last
    `test_span` of data and must not be touched until final evaluation.
    """
    _validate_index(index)
    test_span = pd.Timedelta(test_span)
    data_end_exclusive = (index[-1] + HOUR).ceil("D")
    cutoff = data_end_exclusive - test_span
    if cutoff <= index[0]:
        raise ValueError(
            f"test_span={test_span} leaves no development data "
            f"(data runs {index[0]} .. {index[-1]})"
        )
    return cutoff


def rolling_origin_folds(
    index: pd.DatetimeIndex,
    *,
    min_train: str | pd.Timedelta = "364D",
    horizon: str | pd.Timedelta = "24h",
    step: str | pd.Timedelta = "7D",
    expanding: bool = True,
    max_folds: int | None = None,
) -> list[Fold]:
    """Build midnight-aligned rolling-origin folds over `index`.

    At each origin: train on [train_start, origin), forecast
    [origin, origin + horizon). Origins advance by `step`. With
    expanding=True the training window grows from the start of the data
    (simulates periodic retraining on all history); with expanding=False it
    slides, always spanning exactly `min_train`.

    min_train defaults to 364 days (a multiple of 7) so every fold's
    training window has identical weekday balance.
    """
    _validate_index(index)
    min_train = pd.Timedelta(min_train)
    horizon = pd.Timedelta(horizon)
    step = pd.Timedelta(step)
    if min(min_train, horizon, step) <= pd.Timedelta(0):
        raise ValueError("min_train, horizon and step must all be positive")

    data_start, data_end = index[0], index[-1]
    first_origin = (data_start + min_train).ceil("D")
    last_origin = data_end - horizon + HOUR  # test window must fit inside data

    origins: list[pd.Timestamp] = []
    origin = first_origin
    while origin <= last_origin:
        origins.append(origin)
        origin = origin + step

    if not origins:
        raise ValueError(
            f"not enough data for a single fold: need more than "
            f"{min_train + horizon} of history, have {data_end - data_start}"
        )
    if max_folds is not None and len(origins) > max_folds:
        origins = origins[-max_folds:]  # keep the most recent folds

    return [
        Fold(
            fold_id=i,
            train_start=data_start if expanding else o - min_train,
            origin=o,
            test_end=o + horizon,
        )
        for i, o in enumerate(origins)
    ]


def folds_summary(folds: list[Fold], index: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per fold, with actual row counts (reveals gaps in the data)."""
    return pd.DataFrame(
        {
            "fold": [f.fold_id for f in folds],
            "train_start": [f.train_start for f in folds],
            "origin": [f.origin for f in folds],
            "test_end": [f.test_end for f in folds],
            "train_rows": [int(f.train_mask(index).sum()) for f in folds],
            "test_rows": [int(f.test_mask(index).sum()) for f in folds],
        }
    )


def _demo() -> None:
    """Inspect the split plan on the real feature table."""
    from pathlib import Path

    path = Path("data/processed/features.parquet")
    df = pd.read_parquet(path)

    if isinstance(df.index, pd.DatetimeIndex):
        index = df.index
    else:
        for col in ("timestamp", "datetime", "time", "ds"):
            if col in df.columns:
                index = pd.DatetimeIndex(df[col])
                break
        else:
            raise SystemExit(
                f"No datetime index or recognised time column in {path}.\n"
                f"Columns: {list(df.columns)}\n"
                "Paste this output back so we can adapt."
            )
    index = index.sort_values()

    full_range = pd.date_range(index[0], index[-1], freq="1h", tz=index.tz)
    missing = full_range.difference(index)

    print(f"file           : {path}")
    print(f"rows           : {len(index)}")
    print(f"range          : {index[0]}  ->  {index[-1]}")
    print(f"timezone       : {index.tz}")
    print(f"missing hours  : {len(missing)}")

    cutoff = holdout_cutoff(index, test_span="56D")
    dev_index = index[index < cutoff]
    test_index = index[index >= cutoff]
    print(f"\nfinal holdout cutoff : {cutoff}")
    print(f"dev rows             : {len(dev_index)}")
    print(f"test rows (frozen)   : {len(test_index)}")

    folds = rolling_origin_folds(dev_index)
    summary = folds_summary(folds, dev_index)
    print(f"\nrolling-origin folds on dev period: {len(folds)}")
    with pd.option_context("display.max_rows", 12, "display.width", 120):
        print(summary)


if __name__ == "__main__":
    _demo()
