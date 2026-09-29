"""Phase 3 step 0b: data-state check that doesn't trust the TSO stream.

Step 0 assumed actual/da_forecast ~ 1.0 in normal operation; the data
falsified that (normal months sit at ~1.3). This version compares 2026
actual load against its OWN history: same ISO week + weekday, 2023-2025.
"""

import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")
if "timestamp" in df.columns:
    df = df.set_index("timestamp")
df = df.sort_index()

# 1. weather extent
w = df["temperature_2m"].dropna()
print(f"weather last non-NaN hour : {w.index.max()}")
print(f"load last non-NaN hour    : {df['load_mw'].dropna().index.max()}")
print()

# 2. monthly mean actual load, month x year
m = df["load_mw"].resample("1MS").mean()
t = pd.DataFrame({"year": m.index.year, "month": m.index.month, "v": m.values})
print("monthly mean ACTUAL load (MW), month x year:")
print(
    t.pivot(index="month", columns="year", values="v").to_string(float_format=lambda x: f"{x:,.0f}")
)
print()

# 3. is the ~1.3 actual/TSO offset new, or always there?
daily = df[["load_mw", "da_forecast_mw"]].resample("1D").mean()
daily["ratio_da"] = daily["load_mw"] / daily["da_forecast_mw"]
r = daily["ratio_da"].resample("1MS").median()
t = pd.DataFrame({"year": r.index.year, "month": r.index.month, "v": r.values})
print("monthly median of daily actual/TSO ratio, month x year:")
print(
    t.pivot(index="month", columns="year", values="v").to_string(float_format=lambda x: f"{x:.2f}")
)
print()

# 4. year-over-year check against same ISO week + weekday, 2023-2025
d = daily[["load_mw"]].copy()
iso = d.index.isocalendar()
d["week"] = iso["week"].astype(int).values
d["dow"] = iso["day"].astype(int).values
d["year"] = d.index.year
ref = d[d["year"] <= 2025].groupby(["week", "dow"])["load_mw"].mean().rename("ref_mw")
cur = d[d.index >= "2026-05-01"].join(ref, on=["week", "dow"])
cur["yoy"] = cur["load_mw"] / cur["ref_mw"]

print("weekly median YoY ratio (2026 vs 2023-25 same week+weekday):")
wk = cur["yoy"].resample("W-SUN").median()
print(wk.to_string(float_format=lambda x: f"{x:.2f}"))
print()

for label, flag in [
    ("LOW  (yoy < 0.85)", cur["yoy"] < 0.85),
    ("HIGH (yoy > 1.15)", cur["yoy"] > 1.15),
]:
    print(f"runs flagged {label}:")
    runs = (flag != flag.shift()).cumsum()
    hits = cur[flag]
    if hits.empty:
        print("  none")
    for _, g in hits.groupby(runs[flag]):
        print(
            f"  {g.index.min().date()} -> {g.index.max().date()}  "
            f"({len(g)} days, median yoy {g['yoy'].median():.2f})"
        )
    print()

n_nan = cur["load_mw"].isna().sum()
print(f"days with NaN load since 2026-05-01: {n_nan}")
