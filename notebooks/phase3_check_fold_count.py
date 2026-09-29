"""Confirm Phase 2's scored folds = first 140 generated origins, and that
the dev backtest and holdout did not overlap."""

import pandas as pd
from gridcast.models.splits import holdout_cutoff, rolling_origin_folds

snap = pd.read_parquet("data/processed/features_phase2_snapshot.parquet")
if not isinstance(snap.index, pd.DatetimeIndex):
    snap = snap.set_index("timestamp")
idx = snap.sort_index().index
cut = holdout_cutoff(idx, test_span="56D")
gen = [f.origin for f in rolling_origin_folds(idx[idx < cut], step="5D")]

b = pd.read_parquet("data/backtests/baselines.parquet")
rec = b.groupby("fold")["timestamp"].min().sort_index()
same = all(pd.Timestamp(rec.loc[i]) == gen[i] for i in rec.index)
print(f"scored folds            : {len(rec)}")
print(f"origins match generated : {same}")
print(f"first / last origin     : {rec.iloc[0]}  /  {rec.iloc[-1]}")
dev_last = b["timestamp"].max()
print(f"last dev test hour      : {dev_last}")

h = pd.read_parquet("data/backtests/holdout.parquet")
print(f"\nholdout columns         : {list(h.columns)}")
ts = h["timestamp"] if "timestamp" in h.columns else h.index.to_series()
ts = pd.to_datetime(ts, utc=True)
print(f"holdout range           : {ts.min()}  ->  {ts.max()}")
gap = ts.min() - dev_last
print(f"gap dev -> holdout      : {gap}  ({'OVERLAP' if gap <= pd.Timedelta(0) else 'no overlap'})")
