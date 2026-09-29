"""Quantile LightGBM (p10/p50/p90) on the pinned Phase 2 folds.

One model per quantile level: objective="quantile", alpha=tau. Everything
else is Phase 2's FIXED hyperparameters (copied, never mutated) -- no
tuning, same evaluation-selection-leakage argument as Phase 2.

Early stopping uses each quantile's OWN pinball loss (eval_metric
"quantile"): stopping a p90 model on L1 would choose its tree count by a
median criterion.

Features follow the Phase 2 production decision: da_forecast_mw dropped.
Exclusions are applied (a no-op for dev folds, correct for any folds), and
the frame is cut at the last fold's test end so post-backtest data cannot
enter the design matrix.

Independently fitted quantiles can cross: raw predictions are stored, the
crossing rate reported, and rows sorted (rearrangement -- never increases
summed pinball loss, see metrics.py).

Run:  uv run python -m gridcast.models.quantile_lgbm            (140 folds)
      uv run python -m gridcast.models.quantile_lgbm --limit 3  (smoke test)
"""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast.models.backtest import FEATURES_PATH, OUTPUT_DIR
from gridcast.models.evaluation import backtest_folds
from gridcast.models.exclusions import apply_exclusions, load_exclusions
from gridcast.models.lgbm import LGBM_PARAMS, VAL_SPAN
from gridcast.models.metrics import (
    DEFAULT_QUANTILES,
    crossing_rate,
    rearrange,
    summarize,
    summarize_quantiles,
)
from gridcast.models.ml_features import (
    EXTERNAL_FORECAST_COL,
    build_design_matrix,
    daily_origins,
    feature_columns,
)

MODEL_NAME = "lgbm_quantile"
DROP_FEATURES: tuple[str, ...] = (EXTERNAL_FORECAST_COL,)
NAMES = sorted(DEFAULT_QUANTILES, key=DEFAULT_QUANTILES.__getitem__)
RESULTS_PATH = OUTPUT_DIR / "quantile_lgbm.parquet"
SMOKE_PATH = OUTPUT_DIR / "quantile_lgbm_smoke.parquet"


def quantile_params(tau: float) -> dict:
    """Phase 2 params with the quantile objective. Returns a COPY."""
    params = dict(LGBM_PARAMS)
    params["objective"] = "quantile"
    params["alpha"] = tau
    return params


def fit_predict_fold_quantiles(
    matrix: pd.DataFrame,
    fold_origin: pd.Timestamp,
    quantiles: dict[str, float] = DEFAULT_QUANTILES,
    drop_features: tuple[str, ...] = DROP_FEATURES,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Train one model per quantile on origins strictly before fold_origin,
    predict its 24 rows. Returns (raw predictions by name, best_iteration by
    name). Same train/validation carving as Phase 2's fit_predict_fold."""
    feats = [c for c in feature_columns(matrix) if c not in drop_features]

    train = matrix[(matrix["origin"] < fold_origin) & matrix["y_true"].notna()]
    if train.empty:
        raise ValueError(f"no training rows before origin {fold_origin}")
    val_cut = fold_origin - VAL_SPAN
    core = train[train["origin"] < val_cut]
    val = train[train["origin"] >= val_cut]
    if core.empty or val.empty:
        raise ValueError(f"fold at {fold_origin}: cannot carve a {VAL_SPAN} validation span")

    test = matrix[matrix["origin"] == fold_origin].sort_values("horizon")
    if len(test) != 24:
        raise ValueError(f"expected 24 rows for origin {fold_origin}, got {len(test)}")

    preds: dict[str, np.ndarray] = {}
    iters: dict[str, int] = {}
    for name, tau in quantiles.items():
        model = lgb.LGBMRegressor(**quantile_params(tau))
        model.fit(
            core[feats],
            core["y_true"],
            eval_X=val[feats],  # eval_set is deprecated in this LightGBM version
            eval_y=val["y_true"],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        preds[name] = np.asarray(
            model.predict(test[feats], num_iteration=model.best_iteration_), dtype=float
        )
        iters[name] = int(model.best_iteration_ or 0)
    return preds, iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N folds only")
    args = ap.parse_args()

    folds = backtest_folds()
    if args.limit is not None:
        folds = folds[: args.limit]
    out_path = SMOKE_PATH if args.limit is not None else RESULTS_PATH
    fold_ids = [f.fold_id for f in folds]

    frame = pd.read_parquet(FEATURES_PATH)
    frame = apply_exclusions(frame, load_exclusions())
    ts = pd.DatetimeIndex(frame["timestamp"])
    frame = frame[np.asarray(ts < folds[-1].test_end)]
    index = pd.DatetimeIndex(frame["timestamp"]).sort_values()
    print(f"folds: {len(folds)} ({folds[0].origin.date()} -> {folds[-1].origin.date()})")
    print(f"frame: {index[0]} -> {index[-1]}, dropped features: {list(DROP_FEATURES)}")

    t0 = time.perf_counter()
    matrix = build_design_matrix(frame, daily_origins(index))
    print(
        f"design matrix: {matrix.shape[0]} rows x {matrix.shape[1]} cols "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    records: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for k, fold in enumerate(folds, start=1):
        preds, iters = fit_predict_fold_quantiles(matrix, fold.origin)
        test = matrix[matrix["origin"] == fold.origin].sort_values("horizon")
        raw = np.column_stack([preds[n] for n in NAMES])
        fixed = rearrange(raw)
        rec = pd.DataFrame(
            {
                "fold": fold.fold_id,
                "model": MODEL_NAME,
                "origin": fold.origin,
                "timestamp": test["timestamp"].to_numpy(),
                "horizon": test["horizon"].astype(int).to_numpy(),
                "y_true": test["y_true"].to_numpy(),
            }
        )
        for j, n in enumerate(NAMES):
            rec[n] = fixed[:, j]
            rec[f"{n}_raw"] = raw[:, j]
            rec[f"iter_{n}"] = iters[n]
        records.append(rec)
        if k == 1 or k % 10 == 0 or k == len(folds):
            el = time.perf_counter() - t0
            eta = el / k * (len(folds) - k)
            print(f"  fold {k:3d}/{len(folds)}  {el:5.0f}s elapsed  ~{eta / 60:4.1f} min left")

    results = pd.concat(records, ignore_index=True)
    results["hour_local"] = results["timestamp"].dt.tz_convert("Europe/Amsterdam").dt.hour
    results.to_parquet(out_path, index=False)
    el = time.perf_counter() - t0
    print(f"\n{len(folds)} folds in {el:.0f}s ({el / len(folds):.1f}s/fold) -> {out_path}")

    # --- raw vs rearranged, same rows
    raw_view = results[["y_true", *[f"{n}_raw" for n in NAMES]]].copy()
    raw_view.columns = ["y_true", *NAMES]
    raw_view["model"] = f"{MODEL_NAME}_raw"
    fixed_view = results[["model", "y_true", *NAMES]]
    both = pd.concat([raw_view, fixed_view], ignore_index=True)
    table = summarize_quantiles(both).set_index("model").T
    print("\n=== quantile metrics (raw vs rearranged) ===")
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

    raw_q = results[[f"{n}_raw" for n in NAMES]].to_numpy()
    print(f"\nraw crossing rate: {crossing_rate(raw_q):.2%} of rows")
    for n in NAMES:
        print(
            f"median best_iteration {n}: {int(results.groupby('fold')[f'iter_{n}'].first().median())}"  # noqa: E501
        )

    # --- Phase 2 point models on the SAME folds, for the q50 comparison
    ref = pd.read_parquet(OUTPUT_DIR / "baselines.parquet")
    ref = ref[ref["fold"].isin(fold_ids) & ref["model"].str.contains("lightgbm")]
    print("\n=== Phase 2 LightGBM variants, same folds (compare mae with mae_q50) ===")
    if ref.empty:
        print("  none found in baselines.parquet")
    else:
        print(summarize(ref).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
