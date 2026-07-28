# notebooks/
"""Did ENTSO-E revise the June/July 2026 actuals since our last fetch?"""

import pandas as pd

for month in ("2026-06", "2026-07"):
    old = pd.read_parquet(f"data/raw/entsoe/actual/{month}.parquet.bak").set_index("timestamp")
    new = pd.read_parquet(f"data/raw/entsoe/actual/{month}.parquet").set_index("timestamp")
    j = old.join(new, lsuffix="_old", rsuffix="_new", how="inner")
    diff = (j["load_mw_new"] - j["load_mw_old"]).abs()
    print(
        f"{month}: {len(j)} common rows, "
        f"{(diff > 1.0).sum()} changed by >1 MW, max change {diff.max():,.1f} MW"
    )
    changed = j[diff > 1.0]
    if len(changed):
        print(changed.head(8).to_string(float_format=lambda v: f"{v:,.1f}"))
