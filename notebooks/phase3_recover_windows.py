"""Recover Phase 2's evaluation windows from the snapshot, and check the
current data reproduces the identical fold set under a pinned cutoff."""

import pandas as pd
from gridcast.models.splits import holdout_cutoff, rolling_origin_folds


def load_index(path: str) -> pd.DatetimeIndex:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    return df.sort_index().index


snap = load_index("data/processed/features_phase2_snapshot.parquet")
cut = holdout_cutoff(snap, test_span="56D")
folds = rolling_origin_folds(snap[snap < cut], step="5D")

print(f"snapshot range   : {snap[0]} -> {snap[-1]}")
print(f"phase2 cutoff    : {cut}")
print(f"phase2 folds     : {len(folds)}")
print(f"first origin     : {folds[0].origin}")
print(f"last origin      : {folds[-1].origin}")

now = load_index("data/processed/features.parquet")
folds_now = rolling_origin_folds(now[now < cut], step="5D")
same = [f.origin for f in folds] == [f.origin for f in folds_now]
print(f"\ncurrent data, same pinned cutoff -> {len(folds_now)} folds, identical origins: {same}")
