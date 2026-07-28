"""Diagnose the day-ahead forecast misalignment.

If the ENTSO-E day-ahead series is time-shifted relative to actuals (e.g. a
CET/CEST vs UTC mistake in parsing), then re-aligning it by the right number
of hours should collapse its error from ~16% MAPE to the 1.5-3% a TSO
forecast actually achieves. We test shifts of -3..+3 hours and report MAE
for each; the minimum tells us the true offset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

actual = pd.read_parquet(Path("data/processed/actual_hourly.parquet"))
da = pd.read_parquet(Path("data/processed/day_ahead_hourly.parquet"))


def to_series(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        vals = df.select_dtypes("number").iloc[:, 0]
    else:
        time_col = next(c for c in ("timestamp", "datetime", "time", "ds") if c in df.columns)
        idx = pd.DatetimeIndex(df[time_col])
        vals = df.drop(columns=[time_col]).select_dtypes("number").iloc[:, 0]
    return pd.Series(vals.to_numpy(dtype=float), index=idx).sort_index()


y = to_series(actual)
f = to_series(da)

print(f"actuals   : {len(y)} rows, {y.index[0]} -> {y.index[-1]}")
print(f"day-ahead : {len(f)} rows, {f.index[0]} -> {f.index[-1]}")

common = y.index.intersection(f.index)
identical = np.allclose(y.reindex(common), f.reindex(common), equal_nan=True)
print(f"overlapping hours: {len(common)}, series identical: {identical}")

print("\nMAE and MAPE of day-ahead vs actuals, at candidate shifts:")
print(f"{'shift':>6} {'MAE':>10} {'MAPE %':>8}")
for shift in range(-3, 4):
    shifted = f.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=shift)
    aligned = pd.concat({"y": y, "f": shifted}, axis=1).dropna()
    err = aligned["y"] - aligned["f"]
    mae = err.abs().mean()
    mape = (err.abs() / aligned["y"]).mean() * 100
    print(f"{shift:>+6} {mae:>10,.1f} {mape:>8,.2f}")

print("\nOne sample day, side by side (2025-01-15 UTC):")
day = pd.date_range("2025-01-15", periods=24, freq="1h", tz="UTC")
sample = pd.DataFrame({"actual": y.reindex(day), "day_ahead": f.reindex(day)})
sample.index = sample.index.strftime("%H:%M")
print(sample.to_string(float_format=lambda v: f"{v:,.0f}"))
