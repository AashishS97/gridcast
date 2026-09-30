"""Where is the quantile model miscalibrated? Coverage, bias and width by
month, local hour and day type. Start of the Phase 3 failure analysis.

Note: origins are midnight UTC, so horizon and hour-of-day are perfectly
confounded -- only hour_local is reported."""

import pandas as pd
from gridcast.models.metrics import interval_coverage, quantile_coverage

r = pd.read_parquet("data/backtests/quantile_lgbm.parquet")
r["err"] = r["y_true"] - r["q50"]
local = r["timestamp"].dt.tz_convert("Europe/Amsterdam")
r["month"] = local.dt.strftime("%Y-%m")
r["day_type"] = local.dt.dayofweek.map(lambda d: "weekend" if d >= 5 else "weekday")
cols = ["y_true", "q10", "q50", "q90", "err"]


def calib(g: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "n": len(g),
            "cov10": quantile_coverage(g["y_true"], g["q10"]),
            "cov50": quantile_coverage(g["y_true"], g["q50"]),
            "cov90": quantile_coverage(g["y_true"], g["q90"]),
            "int_cov": interval_coverage(g["y_true"], g["q10"], g["q90"]),
            "bias_mw": g["err"].mean(),
            "bias_pct": 100 * g["err"].mean() / g["y_true"].mean(),
            "width": (g["q90"] - g["q10"]).mean(),
        }
    )


fmt = lambda v: f"{v:,.2f}"  # noqa: E731
print("targets: cov10 0.10 | cov50 0.50 | cov90 0.90 | int_cov 0.80 | bias 0\n")
print("=== overall ===")
print(calib(r[cols]).to_string(float_format=fmt))
for key in ["month", "hour_local", "day_type"]:
    print(f"\n=== by {key} ===")
    print(r.groupby(key)[cols].apply(calib).to_string(float_format=fmt))
