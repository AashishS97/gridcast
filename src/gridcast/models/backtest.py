"""Generic rolling-origin backtest runner.

Runs any set of forecasters over the frozen folds and emits one long tidy
frame: (fold, model, timestamp, horizon, y_true, y_pred). Every Phase 2
model — baselines, SARIMAX, LightGBM — goes through this exact harness, so
the final comparison table is a groupby over identical folds, not an
argument.

horizon: hours ahead of the origin, 1..24 (timestamp = origin + (h-1) * 1h).
hour_local: Europe/Amsterdam hour-of-day, because "struggles at 18:00"
should mean Dutch dinner peak regardless of season.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.models.baselines import (
    Forecaster,
    make_bias_corrected_benchmark,
    make_external_benchmark,
    persistence,
    seasonal_naive_24,
    seasonal_naive_168,
)
from gridcast.models.metrics import summarize
from gridcast.models.sarimax import sarimax_forecaster
from gridcast.models.splits import Fold, holdout_cutoff, rolling_origin_folds

FEATURES_PATH = Path("data/processed/features.parquet")
DAY_AHEAD_PATH = Path("data/processed/day_ahead_hourly.parquet")
OUTPUT_DIR = Path("data/backtests")

TARGET_CANDIDATES = [
    "load_mw",
    "actual_load_mw",
    "actual_load",
    "load",
    "total_load",
    "actual",
    "y",
    "target",
]


def _to_hourly_series(df: pd.DataFrame, path: Path, candidates: list[str]) -> pd.Series:
    """Extract a UTC-indexed hourly series from a parquet, detecting the
    time index and the value column. Prints what it chose so a wrong guess
    is caught by eyeball, not discovered three steps later."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        time_cols = [c for c in ("timestamp", "datetime", "time", "ds") if c in df.columns]
        if not time_cols:
            raise SystemExit(
                f"{path}: no DatetimeIndex and no recognised time column.\n"
                f"Columns: {list(df.columns)}\nPaste this back and we adapt."
            )
        idx = pd.DatetimeIndex(df[time_cols[0]])
        df = df.drop(columns=time_cols[:1])
    hits = [c for c in candidates if c in df.columns]
    if hits:
        col = hits[0]
    else:
        numeric = df.select_dtypes(include="number").columns
        if len(numeric) != 1:
            raise SystemExit(
                f"{path}: could not identify the value column.\n"
                f"Columns: {list(df.columns)}\nPaste this back and we adapt."
            )
        col = numeric[0]
    series = pd.Series(df[col].to_numpy(dtype=float), index=idx, name=col).sort_index()
    print(
        f"{path.name}: using column '{col}', {len(series)} rows, "
        f"{series.index[0]} -> {series.index[-1]}"
    )
    return series


def run_backtest(
    y: pd.Series,
    folds: list[Fold],
    forecasters: dict[str, Forecaster],
) -> pd.DataFrame:
    records = []
    index = y.index
    for fold in folds:
        history = y[fold.train_mask(index)]
        test = y[fold.test_mask(index)]
        test_index = test.index
        horizons = ((test_index - fold.origin) / pd.Timedelta("1h")).astype(int) + 1
        for name, forecaster in forecasters.items():
            preds = forecaster(history, test_index)
            if len(preds) != len(test_index):
                raise ValueError(
                    f"{name} returned {len(preds)} predictions for "
                    f"{len(test_index)} timestamps in fold {fold.fold_id}"
                )
            records.append(
                pd.DataFrame(
                    {
                        "fold": fold.fold_id,
                        "model": name,
                        "timestamp": test_index,
                        "horizon": horizons,
                        "y_true": test.to_numpy(dtype=float),
                        "y_pred": np.asarray(preds, dtype=float),
                    }
                )
            )
    results = pd.concat(records, ignore_index=True)
    local = results["timestamp"].dt.tz_convert("Europe/Amsterdam")
    results["hour_local"] = local.dt.hour
    return results


def main() -> None:
    y = _to_hourly_series(pd.read_parquet(FEATURES_PATH), FEATURES_PATH, TARGET_CANDIDATES)

    day_ahead = _to_hourly_series(
        pd.read_parquet(DAY_AHEAD_PATH),
        DAY_AHEAD_PATH,
        ["day_ahead_mw", "forecast_load_mw", "day_ahead_load", "forecast", "load_forecast"],
    )

    cutoff = holdout_cutoff(y.index, test_span="56D")
    y_dev = y[y.index < cutoff]
    folds = rolling_origin_folds(y_dev.index, step="5D")
    print(f"\nDev period: {y_dev.index[0]} -> {y_dev.index[-1]}, {len(folds)} folds")

    forecasters: dict[str, Forecaster] = {
        "persistence": persistence,
        "seasonal_naive_24": seasonal_naive_24,
        "seasonal_naive_168": seasonal_naive_168,
        "entsoe_day_ahead": make_external_benchmark(day_ahead),
        "entsoe_da_debiased": make_bias_corrected_benchmark(day_ahead, y),
        "sarimax_fourier": sarimax_forecaster,
    }

    results = run_backtest(y_dev, folds, forecasters)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "baselines.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\nSaved {len(results)} rows -> {out_path}")

    print("\n=== Overall (all folds, all horizons) ===")
    overall = summarize(results).sort_values("mae")
    print(overall.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    print("\n=== MAE by horizon (teaser — full breakdown in step 5) ===")
    by_h = summarize(results, by=["model", "horizon"])
    pivot = by_h.pivot(index="horizon", columns="model", values="mae")
    print(pivot.loc[[1, 6, 12, 18, 24]].to_string(float_format=lambda v: f"{v:,.0f}"))


if __name__ == "__main__":
    main()
