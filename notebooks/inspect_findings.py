"""One-off exploration of quality findings. Not part of the pipeline."""

import pandas as pd
from gridcast.data import quality
from gridcast.data.base import LoadKind
from gridcast.data.fetch import load_raw

pd.set_option("display.width", 140)

for kind in LoadKind:
    df = load_raw(kind)
    print(f"\n================ {kind.value} ================")

    oob = quality.find_out_of_bounds(df)
    print(f"\n--- out_of_bounds: {len(oob)} points ---")
    print(oob["load_mw"].describe().to_string())
    print(
        "below low:",
        int((oob["load_mw"] < quality.LOAD_MW_LOW).sum()),
        "| above high:",
        int((oob["load_mw"] > quality.LOAD_MW_HIGH).sum()),
        "| NaN:",
        int(oob["load_mw"].isna().sum()),
    )
    local = oob["timestamp"].dt.tz_convert("Europe/Amsterdam")
    print("\nOOB count by local hour:")
    print(local.dt.hour.value_counts().sort_index().to_string())
    print("\nOOB count by month:")
    print(local.dt.strftime("%Y-%m").value_counts().sort_index().to_string())
    print("\nOOB count by local weekday (0=Mon):")
    print(local.dt.dayofweek.value_counts().sort_index().to_string())

    run_id = (oob["timestamp"].diff() != pd.Timedelta(minutes=15)).cumsum()
    runs = oob.groupby(run_id).agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        n=("timestamp", "size"),
        min_mw=("load_mw", "min"),
    )
    print(f"\nOOB forms {len(runs)} consecutive runs; 10 longest:")
    print(runs.sort_values("n", ascending=False).head(10).to_string())

    spikes = quality.find_spikes(df)
    print(f"\n--- spikes: {len(spikes)} ---")
    if len(spikes):
        local_sp = spikes["timestamp"].dt.tz_convert("Europe/Amsterdam")
        print("spike count by local hour:")
        print(local_sp.dt.hour.value_counts().sort_index().to_string())

print("\n================ context around the 3 actual spike events ================")
df = load_raw(LoadKind.ACTUAL).set_index("timestamp")
for t in [
    "2024-10-07 14:30:00+00:00",
    "2025-01-08 16:00:00+00:00",
    "2026-06-25 17:45:00+00:00",
]:
    ts = pd.Timestamp(t)
    print(f"\ncontext around {t}:")
    print(df.loc[ts - pd.Timedelta(hours=1) : ts + pd.Timedelta(hours=1)].to_string())
