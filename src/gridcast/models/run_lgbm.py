"""Backtest LightGBM on the same frozen folds as every other model, and
merge its results into data/backtests/baselines.parquet so step 5's
comparison reads one file.

Run:  uv run python -m gridcast.models.run_lgbm
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from gridcast.models.backtest import FEATURES_PATH, OUTPUT_DIR
from gridcast.models.lgbm import fit_predict_fold
from gridcast.models.metrics import summarize
from gridcast.models.ml_features import build_design_matrix, daily_origins
from gridcast.models.splits import holdout_cutoff, rolling_origin_folds

MODEL_NAME = "lightgbm"
RESULTS_PATH = OUTPUT_DIR / "baselines.parquet"


def main() -> None:
    frame = pd.read_parquet(FEATURES_PATH)
    index = pd.DatetimeIndex(frame["timestamp"]).sort_values()

    cutoff = holdout_cutoff(index, test_span="56D")
    dev_index = index[index < cutoff]
    folds = rolling_origin_folds(dev_index, step="5D")
    print(f"Dev period: {dev_index[0]} -> {dev_index[-1]}, {len(folds)} folds")

    # Build the design matrix ONCE over all dev origins (features depend only
    # on (origin, horizon), never on fold membership), then slice per fold.
    dev_frame = frame[frame["timestamp"] < cutoff]
    origins = daily_origins(dev_index)
    t0 = time.perf_counter()
    matrix = build_design_matrix(dev_frame, origins)
    print(
        f"Design matrix: {matrix.shape[0]} rows x {matrix.shape[1]} cols "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    records: list[pd.DataFrame] = []
    iters: list[int] = []
    t0 = time.perf_counter()
    for fold in folds:
        preds, best_iter = fit_predict_fold(matrix, fold.origin)
        iters.append(best_iter)
        test = matrix[matrix["origin"] == fold.origin].sort_values("horizon")
        records.append(
            pd.DataFrame(
                {
                    "fold": fold.fold_id,
                    "model": MODEL_NAME,
                    "timestamp": test["timestamp"].to_numpy(),
                    "horizon": test["horizon"].astype(int).to_numpy(),
                    "y_true": test["y_true"].to_numpy(),
                    "y_pred": preds,
                }
            )
        )
        if fold.fold_id % 20 == 0:
            print(
                f"  fold {fold.fold_id:3d}/{len(folds)}  "
                f"({time.perf_counter() - t0:.0f}s elapsed)"
            )

    results = pd.concat(records, ignore_index=True)
    local = results["timestamp"].dt.tz_convert("Europe/Amsterdam")
    results["hour_local"] = local.dt.hour
    elapsed = time.perf_counter() - t0
    print(
        f"\n{len(folds)} folds in {elapsed:.0f}s "
        f"({elapsed / len(folds):.1f}s/fold); "
        f"median best_iteration {int(np.median(iters))}"
    )

    existing = pd.read_parquet(RESULTS_PATH)
    existing = existing[existing["model"] != MODEL_NAME]  # idempotent reruns
    merged = pd.concat([existing, results], ignore_index=True)
    merged.to_parquet(RESULTS_PATH, index=False)
    print(f"Merged into {RESULTS_PATH} ({len(merged)} rows total)")

    print("\n=== Overall (all folds, all horizons) ===")
    print(
        summarize(merged)
        .sort_values("mae")
        .to_string(index=False, float_format=lambda v: f"{v:,.1f}")
    )

    print("\n=== MAE by horizon ===")
    by_h = summarize(merged, by=["model", "horizon"])
    pivot = by_h.pivot(index="horizon", columns="model", values="mae")
    print(pivot.loc[[1, 6, 12, 18, 24]].to_string(float_format=lambda v: f"{v:,.0f}"))


if __name__ == "__main__":
    main()
