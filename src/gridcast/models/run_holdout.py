"""One-shot final evaluation on the frozen 56-day holdout.

This is the ONLY time the holdout is scored. Every Phase 2 decision — split
protocol, model set, hyperparameters, feature set, dropping the TSO forecast
feature — was fixed using dev folds alone.

Protocol matches development: midnight origins, 24h horizon, expanding
training window, retrain at each origin. Origins step daily here (not 5D) to
cover the whole frozen window; all weekdays appear, so no aliasing.

Training at a holdout origin uses everything before it, including earlier
holdout days. That is chronologically valid and simulates daily retraining
in production.

Data incident note: an upstream ENTSO-E publication defect corrupted NL
actual load from late June 2026 onward (implausible midday collapse; the
TSO's own day-ahead forecast stream was unaffected, which is how the defect
was detected). The late-June portion has since been revised upstream and is
now correct; July 2026 has not yet been revised. Origins from
QUARANTINE_START onward are therefore scored separately for documentation
and excluded from all claims. Re-run when ENTSO-E revises July.

Run:  uv run python -m gridcast.models.run_holdout
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from gridcast.models.backtest import FEATURES_PATH, OUTPUT_DIR, run_backtest
from gridcast.models.baselines import (
    Forecaster,
    make_bias_corrected_benchmark,
    make_external_benchmark,
    persistence,
    seasonal_naive_24,
    seasonal_naive_168,
)
from gridcast.models.lgbm import fit_predict_fold
from gridcast.models.metrics import summarize
from gridcast.models.ml_features import (
    EXTERNAL_FORECAST_COL,
    build_design_matrix,
    daily_origins,
)
from gridcast.models.sarimax import sarimax_forecaster
from gridcast.models.splits import Fold, holdout_cutoff

HOLDOUT_PATH = OUTPUT_DIR / "holdout.parquet"
DEV_PATH = OUTPUT_DIR / "baselines.parquet"

# Primary configuration first. lightgbm (with the TSO feature) is reported
# for completeness, not selected on.
LGBM_VARIANTS: dict[str, tuple[str, ...]] = {
    "lightgbm_no_da": (EXTERNAL_FORECAST_COL,),
    "lightgbm": (),
}
MAIN = ["lightgbm_no_da", "seasonal_naive_168", "sarimax_fourier"]

# Origins from this date onward have corrupted y_true (see module docstring).
QUARANTINE_START = pd.Timestamp("2026-06-30", tz="UTC")


def fmt(df: pd.DataFrame, spec: str = "{:,.1f}") -> str:
    return df.to_string(float_format=lambda v: spec.format(v))


def main() -> None:
    frame = pd.read_parquet(FEATURES_PATH).sort_values("timestamp").reset_index(drop=True)
    indexed = frame.set_index("timestamp")
    y = indexed["load_mw"].astype(float)
    da = indexed[EXTERNAL_FORECAST_COL].astype(float)

    cutoff = holdout_cutoff(y.index, test_span="56D")
    last_origin = (y.index[-1] - pd.Timedelta("23h")).floor("D")
    origins = pd.date_range(cutoff, last_origin, freq="1D", tz="UTC")
    folds = [
        Fold(fold_id=i, train_start=y.index[0], origin=o, test_end=o + pd.Timedelta("24h"))
        for i, o in enumerate(origins)
    ]
    n_clean = sum(f.origin < QUARANTINE_START for f in folds)
    print(f"Holdout window : {cutoff} -> {y.index[-1]}")
    print(
        f"Origins        : {len(folds)} daily "
        f"({folds[0].origin.date()} .. {folds[-1].origin.date()})"
    )
    print(
        f"Clean origins  : {n_clean} (before {QUARANTINE_START.date()}), "
        f"quarantined: {len(folds) - n_clean}"
    )

    parts = []

    forecasters: dict[str, Forecaster] = {
        "persistence": persistence,
        "seasonal_naive_24": seasonal_naive_24,
        "seasonal_naive_168": seasonal_naive_168,
        "entsoe_day_ahead": make_external_benchmark(da),
        "entsoe_da_debiased": make_bias_corrected_benchmark(da, y),
        "sarimax_fourier": sarimax_forecaster,
    }
    t0 = time.perf_counter()
    parts.append(run_backtest(y, folds, forecasters))
    print(f"baselines + sarimax : {time.perf_counter() - t0:.0f}s")

    matrix = build_design_matrix(frame, daily_origins(y.index))
    print(f"design matrix       : {matrix.shape}")

    for name, drop in LGBM_VARIANTS.items():
        t0 = time.perf_counter()
        recs = []
        for fold in folds:
            preds, _ = fit_predict_fold(matrix, fold.origin, drop_features=drop)
            test = matrix[matrix["origin"] == fold.origin].sort_values("horizon")
            recs.append(
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
        out = pd.concat(recs, ignore_index=True)
        out["hour_local"] = out["timestamp"].dt.tz_convert("Europe/Amsterdam").dt.hour
        parts.append(out)
        print(f"{name:20s}: {time.perf_counter() - t0:.0f}s")

    holdout = pd.concat(parts, ignore_index=True)
    holdout["window"] = np.where(holdout["timestamp"] < QUARANTINE_START, "clean", "quarantined")
    holdout.to_parquet(HOLDOUT_PATH, index=False)
    print(f"\nSaved {len(holdout)} rows -> {HOLDOUT_PATH}")

    clean = holdout[holdout["window"] == "clean"]
    quarantined = holdout[holdout["window"] == "quarantined"]

    print(f"\n=== HOLDOUT (CLEAN window, origins before " f"{QUARANTINE_START.date()}) ===")
    overall = summarize(clean).sort_values("mae")
    print(fmt(overall.set_index("model")))

    print("\n=== HOLDOUT (QUARANTINED window — upstream data incident, " "documentation only) ===")
    print(fmt(summarize(quarantined).sort_values("mae").set_index("model")))

    print("\n=== Dev estimate vs CLEAN holdout (MAE) ===")
    dev = summarize(pd.read_parquet(DEV_PATH)).set_index("model")["mae"].rename("dev_mae")
    hold = overall.set_index("model")["mae"].rename("holdout_mae")
    compare = pd.concat([dev, hold], axis=1).dropna()
    compare["delta_%"] = (compare["holdout_mae"] / compare["dev_mae"] - 1) * 100
    print(fmt(compare.sort_values("holdout_mae")))

    print("\n=== CLEAN holdout: MAE by horizon ===")
    by_h = summarize(clean[clean["model"].isin(MAIN)], by=["model", "horizon"])
    print(fmt(by_h.pivot(index="horizon", columns="model", values="mae"), "{:,.0f}"))

    per_fold = (
        clean[clean["model"].isin(MAIN)]
        .groupby(["model", "fold"])
        .apply(lambda g: (g["y_true"] - g["y_pred"]).abs().mean(), include_groups=False)
        .rename("fold_mae")
        .reset_index()
    )
    print("\n=== CLEAN holdout: per-day MAE distribution ===")
    print(
        fmt(
            per_fold.groupby("model")["fold_mae"].describe(percentiles=[0.5, 0.9])[
                ["mean", "50%", "90%", "max"]
            ]
        )
    )

    wide = per_fold.pivot(index="fold", columns="model", values="fold_mae")
    wins = (wide["lightgbm_no_da"] < wide["seasonal_naive_168"]).mean()
    print(f"\nlightgbm_no_da beats seasonal_naive_168 on {wins:.1%} of " f"{len(wide)} clean days")

    dates = (
        clean[clean["model"] == "lightgbm_no_da"]
        .groupby("fold")["timestamp"]
        .min()
        .dt.tz_convert("Europe/Amsterdam")
        .dt.date.rename("day")
    )
    worst = (
        per_fold[per_fold["model"] == "lightgbm_no_da"]
        .nlargest(5, "fold_mae")
        .set_index("fold")
        .join(dates)
    )
    print("\n=== CLEAN holdout: worst 5 days ===")
    print(fmt(worst[["day", "fold_mae"]]))


if __name__ == "__main__":
    main()
