"""Post-refetch check: is late June now sane, and how much of July is still bad?"""

from pathlib import Path

import pandas as pd

AMS = "Europe/Amsterdam"
frames = [pd.read_parquet(p) for p in sorted(Path("data/raw/entsoe/actual").glob("*.parquet"))]
y = (
    pd.concat(frames, ignore_index=True)
    .drop_duplicates("timestamp", keep="last")
    .set_index("timestamp")["load_mw"]
    .sort_index()
)
loc = y.index.tz_convert(AMS)
mid = (loc.hour >= 11) & (loc.hour <= 16)

daily_min = y[mid].groupby(loc[mid].date).min()
daily_min.index = pd.to_datetime(daily_min.index)
window = daily_min.loc["2026-06-20":]
flagged = window[window < 6000]
print("Midday (11-16 local) daily minimum, from 2026-06-20:")
print(window.to_string(float_format=lambda v: f"{v:,.0f}"))
print(f"\nDays still below 6 GW (implausible): {len(flagged)}")
print("First bad day:", flagged.index.min().date() if len(flagged) else "-")
