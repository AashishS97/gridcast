"""Phase 4: what did the 2026-11-15 refetch change in features.parquet?

Compares the pre-refetch snapshot with the rebuilt table, column by column.
The Phase 3 backtest only reads rows before BACKTEST_END (last fold's test
end), so if nothing changed there, Phase 3's backtest results reproduce from
the new table. Weather is the main suspect: the archive serves preliminary
reanalysis for recent months and replaces it with final values later.

Run:  uv run python notebooks/phase4_vintage_diff.py
"""

from __future__ import annotations

import pandas as pd

OLD = "data/snapshots/2026-11-15_pre_refetch/processed/features.parquet"
NEW = "data/processed/features.parquet"
BACKTEST_END = pd.Timestamp("2026-05-26T00:00:00Z")
TOL = 1e-6


def main() -> None:
    old = pd.read_parquet(OLD).set_index("timestamp").sort_index()
    new = pd.read_parquet(NEW).set_index("timestamp").sort_index()
    print(f"old: {len(old)} rows, {old.index.min()} -> {old.index.max()}")
    print(f"new: {len(new)} rows, {new.index.min()} -> {new.index.max()}")
    common = old.index.intersection(new.index)
    print(
        f"common hours: {len(common)}, only in old: {len(old.index.difference(new.index))}, "
        f"only in new: {len(new.index.difference(old.index))}\n"
    )

    o, n = old.loc[common], new.loc[common]
    masks: dict[str, pd.Series] = {}
    rows = []
    for col in old.columns:
        a, b = o[col].astype(float), n[col].astype(float)
        changed = ((a - b).abs() > TOL) | (a.isna() != b.isna())
        masks[col] = changed
        idx = common[changed.to_numpy()]
        rows.append(
            {
                "column": col,
                "changed": int(changed.sum()),
                "in_backtest": int((idx < BACKTEST_END).sum()),
                "first": idx.min() if len(idx) else "",
                "last": idx.max() if len(idx) else "",
                "max_abs_diff": float((a - b).abs()[changed].max()) if changed.any() else 0.0,
            }
        )
    table = pd.DataFrame(rows).set_index("column")
    print("=== changed hours per column (old vs new, common hours) ===")
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

    m = pd.DataFrame(masks, index=common)
    m = m.loc[:, m.any()]
    if not m.empty:
        by_month = m.groupby(common.strftime("%Y-%m")).sum()
        by_month = by_month[by_month.any(axis=1)]
        print("\n=== changed hours by month ===")
        print(by_month.to_string())

    n_bt = int(table["in_backtest"].sum())
    print(
        "\nVERDICT: backtest window (< 2026-05-26) unchanged -> Phase 3 backtest reproduces"
        if n_bt == 0
        else f"\nVERDICT: {n_bt} changed values INSIDE the backtest window"
    )


if __name__ == "__main__":
    main()
