"""Is the day-ahead level problem present in the raw monthly files,
or introduced during processing? Compare January 2025 raw actual vs
raw day-ahead directly."""

from pathlib import Path

import pandas as pd

for kind in ("actual", "day_ahead"):
    path = Path(f"data/raw/entsoe/{kind}/2025-01.parquet")
    df = pd.read_parquet(path)
    print(f"\n=== {path} ===")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"index: {type(df.index).__name__}", end="")
    if isinstance(df.index, pd.DatetimeIndex):
        print(f", tz={df.index.tz}, {df.index[0]} -> {df.index[-1]}")
    else:
        print()
    print(df.head(6).to_string())
    num = df.select_dtypes("number")
    print("value stats:")
    print(num.describe().loc[["min", "mean", "max"]].to_string(float_format=lambda v: f"{v:,.0f}"))
