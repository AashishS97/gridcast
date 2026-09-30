"""Failure drivers: is miscalibration concentrated where training data is
thin? Temperature rarity is computed PER FOLD, against only the data that
fold could see (timestamps before its origin) -- "rare" must mean rare at
forecast time. Plus holidays, and the worst individual days.

Caveat: temperature is OBSERVED weather (perfect prognosis). The model saw
the true temperature; failures here are failures to USE it, not surprises."""

import numpy as np
import pandas as pd
from gridcast.models.metrics import interval_coverage, quantile_coverage

r = pd.read_parquet("data/backtests/quantile_lgbm.parquet")
f = pd.read_parquet("data/processed/features.parquet")
if "timestamp" in f.columns:
    f = f.set_index("timestamp")
f = f.sort_index()
r = r.join(f[["temperature_2m", "shortwave_radiation", "is_holiday"]], on="timestamp")
r["err"] = r["y_true"] - r["q50"]
r["inside"] = (r["y_true"] >= r["q10"]) & (r["y_true"] <= r["q90"])

# temperature percentile vs that fold's own history
temps = f["temperature_2m"].dropna()
r["temp_pct"] = np.nan
for origin, idx in r.groupby("origin").groups.items():
    hist = np.sort(temps[temps.index < origin].to_numpy())
    t = r.loc[idx, "temperature_2m"].to_numpy()
    r.loc[idx, "temp_pct"] = np.searchsorted(hist, t, side="right") / len(hist)

bins = [-1.0, 0.0, 0.02, 0.10, 0.90, 0.98, 0.999999, 1.0]
labels = ["below-min", "0-2%", "2-10%", "10-90%", "90-98%", "98-100%", "at/above-max"]
r["temp_band"] = pd.cut(r["temp_pct"], bins=bins, labels=labels)

cols = ["y_true", "q10", "q50", "q90", "err", "temperature_2m"]


def calib(g: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "n": len(g),
            "mean_temp": g["temperature_2m"].mean(),
            "cov10": quantile_coverage(g["y_true"], g["q10"]),
            "cov50": quantile_coverage(g["y_true"], g["q50"]),
            "cov90": quantile_coverage(g["y_true"], g["q90"]),
            "int_cov": interval_coverage(g["y_true"], g["q10"], g["q90"]),
            "bias_mw": g["err"].mean(),
            "width": (g["q90"] - g["q10"]).mean(),
        }
    )


fmt = lambda v: f"{v:,.2f}"  # noqa: E731
print("targets: cov10 0.10 | cov50 0.50 | cov90 0.90 | int_cov 0.80 | bias 0")
print("\n=== by temperature band (percentile vs the fold's own training history) ===")
print(r.groupby("temp_band", observed=True)[cols].apply(calib).to_string(float_format=fmt))
print("\n=== by holiday flag ===")
print(r.groupby("is_holiday")[cols].apply(calib).to_string(float_format=fmt))

daily = r.groupby("origin").agg(
    bias_mw=("err", "mean"),
    inside=("inside", "mean"),
    temp_mean=("temperature_2m", "mean"),
    temp_pct_min=("temp_pct", "min"),
    rad_mean=("shortwave_radiation", "mean"),
    holiday=("is_holiday", "max"),
)
daily.index = daily.index.tz_convert("Europe/Amsterdam").strftime("%Y-%m-%d %a")
print("\n=== 12 worst under-forecast days (largest positive bias) ===")
print(daily.sort_values("bias_mw", ascending=False).head(12).to_string(float_format=fmt))
print("\n=== 5 worst over-forecast days ===")
print(daily.sort_values("bias_mw").head(5).to_string(float_format=fmt))
