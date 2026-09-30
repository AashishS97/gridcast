"""TSO as independent witness for the worst days. The TSO day-ahead stream
is biased low (seasonal ~1.06-1.36x), so "surprise" is measured against its
OWN trailing 28-day median actual/TSO ratio, strictly earlier days only.

TSO surprised too  -> nobody anticipated it (demand shock or data defect).
TSO not surprised  -> information existed that our features lack."""

import numpy as np
import pandas as pd

r = pd.read_parquet("data/backtests/quantile_lgbm.parquet")
f = pd.read_parquet("data/processed/features.parquet")
if "timestamp" in f.columns:
    f = f.set_index("timestamp")
f = f.sort_index()

d = f[["load_mw", "da_forecast_mw"]].resample("1D").mean()
d["ratio"] = d["load_mw"] / d["da_forecast_mw"]
d["base"] = d["ratio"].shift(1).rolling(28, min_periods=20).median()
d["tso_surprise_pct"] = 100 * (d["ratio"] / d["base"] - 1)

m = r.groupby("origin").agg(err=("y_true", "mean"), q50=("q50", "mean"))
m["model_bias_pct"] = 100 * (m["err"] - m["q50"]) / m["err"]
m = m[["model_bias_pct"]].join(d[["tso_surprise_pct"]])

rho = m["model_bias_pct"].corr(m["tso_surprise_pct"], method="spearman")
print(f"Spearman(model bias %, TSO surprise %) over {len(m)} folds: {rho:.2f}")
same_sign = np.mean(np.sign(m["model_bias_pct"]) == np.sign(m["tso_surprise_pct"]))
print(f"folds where both err in the same direction: {same_sign:.0%}")

m.index = m.index.tz_convert("Europe/Amsterdam").strftime("%Y-%m-%d %a")
fmt = lambda v: f"{v:,.2f}"  # noqa: E731
print("\n=== 12 worst under-forecast days: model vs TSO surprise (both in %) ===")
print(m.sort_values("model_bias_pct", ascending=False).head(12).to_string(float_format=fmt))
print("\n=== 5 worst over-forecast days ===")
print(m.sort_values("model_bias_pct").head(5).to_string(float_format=fmt))
