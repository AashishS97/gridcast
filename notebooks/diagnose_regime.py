"""Is the midday collapse real, or an artifact in the actuals series?

The actual load and the TSO day-ahead forecast come from independent data
streams. If BOTH collapse, the phenomenon is real and was foreseeable.
If only the actuals do, it is either a genuinely unforecast event or an
artifact in that one stream.
"""

from pathlib import Path

import pandas as pd

AMS = "Europe/Amsterdam"

frame = pd.read_parquet(Path("data/processed/features.parquet")).set_index("timestamp").sort_index()
y = frame["load_mw"].astype(float)
da = frame["da_forecast_mw"].astype(float)
loc = y.index.tz_convert(AMS)
mid = (loc.hour >= 11) & (loc.hour <= 16)

print("=== 1. Worst day (2026-06-29): two independent streams ===")
day = pd.date_range(pd.Timestamp("2026-06-29", tz=AMS), periods=24, freq="1h").tz_convert("UTC")
v = pd.DataFrame(
    {
        "actual": y.reindex(day),
        "tso_day_ahead": da.reindex(day),
        "radiation": frame["shortwave_radiation"].reindex(day),
    }
)
v.index = v.index.tz_convert(AMS).strftime("%H:%M")
print(v.to_string(float_format=lambda x: f"{x:,.0f}"))

print("\n=== 2. Midday (local 11-16) minimum per day, June-July 2026 ===")
tab = pd.DataFrame(
    {
        "actual_min": y[mid].groupby(loc[mid].date).min(),
        "tso_fc_min": da[mid].groupby(loc[mid].date).min(),
    }
)
tab.index = pd.to_datetime(tab.index)
print(tab.loc["2026-06-01":].to_string(float_format=lambda x: f"{x:,.0f}"))

print("\n=== 3. Same window (29 May - 23 Jul), year by year ===")
for yr in (2023, 2024, 2025, 2026):
    lo = pd.Timestamp(f"{yr}-05-29", tz="UTC")
    hi = pd.Timestamp(f"{yr}-07-24", tz="UTC")
    m = mid & (y.index >= lo) & (y.index < hi)
    if m.sum():
        print(f"{yr}: actual min {y[m].min():8,.0f} | tso fc min {da[m].min():8,.0f}")
