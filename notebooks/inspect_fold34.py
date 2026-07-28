# notebooks/inspect_fold34.py
"""Why did LightGBM lose to the naive on 2024-12-16, an easy Monday?
Suspect an input artifact (weather or da_forecast) rather than the model."""

from pathlib import Path

import pandas as pd

frame = pd.read_parquet(Path("data/processed/features.parquet")).set_index("timestamp")
res = pd.read_parquet(Path("data/backtests/baselines.parquet"))
f34 = res[(res["fold"] == 34) & (res["model"] == "lightgbm")].set_index("timestamp")

day = f34.index
view = pd.DataFrame(
    {
        "y_true": f34["y_true"],
        "y_pred": f34["y_pred"],
        "err": f34["y_pred"] - f34["y_true"],
        "temp": frame["temperature_2m"].reindex(day),
        "temp_prev_wk": frame["temperature_2m"].reindex(day - pd.Timedelta("168h")).to_numpy(),
        "da_fc": frame["da_forecast_mw"].reindex(day),
        "lag168": frame["load_mw"].reindex(day - pd.Timedelta("168h")).to_numpy(),
    }
)
view.index = view.index.tz_convert("Europe/Amsterdam").strftime("%H:%M")
print(view.to_string(float_format=lambda v: f"{v:,.1f}"))
