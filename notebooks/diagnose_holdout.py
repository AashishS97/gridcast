# notebooks/diagnose_holdout.py
"""Why did every model collapse on the holdout?

H1 (data)   : load or weather broken/missing in the holdout window.
H2 (season) : summer midday was always the hardest period; the dev average
              hid it, since the holdout is 56 consecutive summer days.
H3 (drift)  : summer 2026 net load is outside anything in training (solar
              growth) -- and trees cannot predict below their training range.
"""

from pathlib import Path

import pandas as pd

CUTOFF = pd.Timestamp("2026-05-29", tz="UTC")
AMS = "Europe/Amsterdam"

frame = pd.read_parquet(Path("data/processed/features.parquet")).set_index("timestamp").sort_index()
dev = pd.read_parquet(Path("data/backtests/baselines.parquet"))
hold = pd.read_parquet(Path("data/backtests/holdout.parquet"))

y = frame["load_mw"].astype(float)
train, test = y[y.index < CUTOFF], y[y.index >= CUTOFF]

print("=== H1: data integrity in the holdout window ===")
hw = frame.loc[CUTOFF:]
print(f"rows: {len(hw)}")
print("NaNs per column:")
print(hw.isna().sum().to_string())
print(
    f"\nload_mw  min {hw['load_mw'].min():,.0f} | "
    f"mean {hw['load_mw'].mean():,.0f} | max {hw['load_mw'].max():,.0f}"
)
print(
    f"radiation max {hw['shortwave_radiation'].max():,.0f} | "
    f"temp range {hw['temperature_2m'].min():.1f} .. {hw['temperature_2m'].max():.1f}"
)

print("\n=== H3: same calendar window (29 May - 23 Jul), local 12:00-15:00 ===")
loc_all = y.index.tz_convert(AMS)
midday = y[(loc_all.hour >= 12) & (loc_all.hour <= 15)]
for yr in (2023, 2024, 2025, 2026):
    lo = pd.Timestamp(f"{yr}-05-29", tz="UTC")
    hi = pd.Timestamp(f"{yr}-07-24", tz="UTC")
    w = midday[(midday.index >= lo) & (midday.index < hi)]
    if len(w):
        print(
            f"{yr}: n={len(w):5d}  min {w.min():8,.0f}  p10 {w.quantile(.10):8,.0f}"
            f"  median {w.median():8,.0f}  mean {w.mean():8,.0f}"
        )
    else:
        print(f"{yr}: no data in window")

print("\n=== H3: can the trees even reach these values? (per local hour) ===")
th = train.index.tz_convert(AMS).hour
hh = test.index.tz_convert(AMS).hour
train_min = train.groupby(th).min()
cmp = pd.DataFrame({"hour_local": hh, "y": test.to_numpy()})
cmp["train_min"] = cmp["hour_local"].map(train_min)
cmp["below"] = (cmp["y"] < cmp["train_min"]).astype(float) * 100.0
tab = cmp.groupby("hour_local").agg(
    holdout_min=("y", "min"),
    holdout_median=("y", "median"),
    train_min=("train_min", "first"),
    pct_below_train_min=("below", "mean"),
)
print(tab.to_string(float_format=lambda v: f"{v:,.0f}"))
print(
    f"\noverall: {(test < train.min()).sum()} of {len(test)} holdout hours "
    f"fall below the global training minimum ({train.min():,.0f} MW)"
)

print("\n=== H2: dev-fold MAE by month-of-year ===")
d = dev[dev["model"].isin(["lightgbm", "seasonal_naive_168"])].copy()
d["ae"] = (d["y_true"] - d["y_pred"]).abs()
d["moy"] = d["timestamp"].dt.tz_convert(AMS).dt.month
print(
    d.groupby(["moy", "model"])["ae"].mean().unstack().to_string(float_format=lambda v: f"{v:,.0f}")
)

print("\n=== H2/H3: dev-fold MAE by year-month (last 14) ===")
d["ym"] = d["timestamp"].dt.tz_convert(AMS).dt.to_period("M").astype(str)
print(
    d.groupby(["ym", "model"])["ae"]
    .mean()
    .unstack()
    .tail(14)
    .to_string(float_format=lambda v: f"{v:,.0f}")
)

print("\n=== Worst holdout day, hour by hour (2026-06-29) ===")
w = hold[(hold["model"] == "lightgbm_no_da") & (hold["fold"] == 31)].set_index("timestamp")
view = pd.DataFrame(
    {
        "y_true": w["y_true"],
        "y_pred": w["y_pred"],
        "err": w["y_pred"] - w["y_true"],
        "radiation": frame["shortwave_radiation"].reindex(w.index),
        "temp": frame["temperature_2m"].reindex(w.index),
        "lag168": y.reindex(w.index - pd.Timedelta("168h")).to_numpy(),
    }
)
view.index = view.index.tz_convert(AMS).strftime("%H:%M")
print(view.to_string(float_format=lambda v: f"{v:,.0f}"))
