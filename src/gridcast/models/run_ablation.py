"""Ablation: rerun LightGBM on the same frozen folds with feature groups
removed, to quantify what each information source contributes.

- lightgbm_no_weather : drops all weather columns. The gap to full lightgbm
  is the value of weather information, and (since production weather
  forecasts are imperfect) production skill is bracketed between the two.
- lightgbm_no_da      : drops the TSO day-ahead forecast feature.
- lightgbm_lags_only  : drops both — pure lags + calendar.

Run:  uv run python -m gridcast.models.run_ablation   (expect ~3x the
lightgbm backtest runtime; ~45-55 min at 7.5 s/fold)
"""

from __future__ import annotations

import time

import pandas as pd

from gridcast.models.backtest import FEATURES_PATH, OUTPUT_DIR
from gridcast.models.lgbm import fit_predict_fold
from gridcast.models.metrics import summarize
from gridcast.models.ml_features import (
    EXTERNAL_FORECAST_COL,
    WEATHER_COLS,
    build_design_matrix,
    daily_origins,
)
from gridcast.models.splits import holdout_cutoff, rolling_origin_folds

RESULTS_PATH = OUTPUT_DIR / "baselines.parquet"

VARIANTS: dict[str, tuple[str, ...]] = {
    "lightgbm_no_weather": tuple(WEATHER_COLS),
    "lightgbm_no_da": (EXTERNAL_FORECAST_COL,),
    "lightgbm_lags_only": tuple(WEATHER_COLS) + (EXTERNAL_FORECAST_COL,),
}


def main() -> None:
    frame = pd.read_parquet(FEATURES_PATH)
    index = pd.DatetimeIndex(frame["timestamp"]).sort_values()
    cutoff = holdout_cutoff(index, test_span="56D")
    dev_index = index[index < cutoff]
    folds = rolling_origin_folds(dev_index, step="5D")

    dev_frame = frame[frame["timestamp"] < cutoff]
    matrix = build_design_matrix(dev_frame, daily_origins(dev_index))
    print(f"{len(folds)} folds, matrix {matrix.shape}")

    all_records: list[pd.DataFrame] = []
    for name, drop in VARIANTS.items():
        t0 = time.perf_counter()
        records = []
        for fold in folds:
            preds, _ = fit_predict_fold(matrix, fold.origin, drop_features=drop)
            test = matrix[matrix["origin"] == fold.origin].sort_values("horizon")
            records.append(
                pd.DataFrame(
                    {
                        "fold": fold.fold_id,
                        "model": name,
                        "timestamp": test["timestamp"].to_numpy(),
                        "horizon": test["horizon"].astype(int).to_numpy(),
                        "y_true": test["y_true"].to_numpy(),
                        "y_pred": preds,
                    }
                )
            )
        out = pd.concat(records, ignore_index=True)
        out["hour_local"] = out["timestamp"].dt.tz_convert("Europe/Amsterdam").dt.hour
        all_records.append(out)
        print(f"{name}: {time.perf_counter() - t0:.0f}s")

    new = pd.concat(all_records, ignore_index=True)
    existing = pd.read_parquet(RESULTS_PATH)
    existing = existing[~existing["model"].isin(VARIANTS)]
    merged = pd.concat([existing, new], ignore_index=True)
    merged.to_parquet(RESULTS_PATH, index=False)

    keep = ["lightgbm", *VARIANTS, "seasonal_naive_168"]
    print("\n=== Ablation (all folds, all horizons) ===")
    print(
        summarize(merged[merged["model"].isin(keep)])
        .sort_values("mae")
        .to_string(index=False, float_format=lambda v: f"{v:,.1f}")
    )


if __name__ == "__main__":
    main()
