"""Phase 3 SEALED holdout: 15 August origins, scored exactly once.

Guard: refuses to run if the output already exists; origins requested with
one_shot=True. After a successful run, flip configs/evaluation.toml
[holdout.phase3] status to "spent".

Production-faithful pipeline per origin:
  1. retrain p10/p50/p90 on all clean data before the origin (July masked;
     the 60-day early-stopping window is mostly masked July, and Aug 2-15
     training rows have NaN lags -- consequences of the outage, documented);
  2. conformal calibration from a pool of EARLIER daily forecasts whose
     actuals closed before the origin, trailing CAL_WINDOW, clean origins
     only. 60 days = backtest W=12 folds x 5D (declared equivalent). Aug 16
     is therefore calibrated purely on June residuals (48-60 days old):
     stale-history calibration after a data outage, reported per origin;
  3. trust flags (expected to barely fire: no holidays, no cold, clean
     lookback -- this holdout tests model + calibration, not the flag).

Run:  uv run python -m gridcast.models.run_holdout_phase3
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from gridcast.models.backtest import FEATURES_PATH, OUTPUT_DIR
from gridcast.models.conformal import conformal_shift
from gridcast.models.evaluation import holdout_origins
from gridcast.models.exclusions import apply_exclusions, contaminated_origins, load_exclusions
from gridcast.models.metrics import DEFAULT_QUANTILES, mae, rearrange, summarize_quantiles
from gridcast.models.ml_features import N_HORIZONS, build_design_matrix, daily_origins
from gridcast.models.quantile_lgbm import NAMES, fit_predict_fold_quantiles
from gridcast.models.trust import FLAG_COLS, trust_flags

HOUR = pd.Timedelta("1h")
CAL_WINDOW = pd.Timedelta("60D")
MIN_CAL_ORIGINS = 6
OUT_PATH = OUTPUT_DIR / "holdout_phase3.parquet"
TZ = "Europe/Amsterdam"


def calibration_pool(
    results: pd.DataFrame, origin: pd.Timestamp, window: pd.Timedelta = CAL_WINDOW
) -> pd.DataFrame:
    """Earlier forecasts usable for calibrating `origin`: issued within the
    trailing window AND with their whole target day observed before origin."""
    o = results["origin"]
    return results[(o >= origin - window) & (o + N_HORIZONS * HOUR <= origin)]


def _view(df: pd.DataFrame, suffix: str, label: str) -> pd.DataFrame:
    v = df[["y_true", *[f"{n}{suffix}" for n in NAMES]]].copy()
    v.columns = ["y_true", *NAMES]
    v["model"] = label
    return v


def main() -> None:
    if OUT_PATH.exists():
        raise SystemExit(f"{OUT_PATH} exists: the Phase 3 holdout was already scored (one-shot).")

    exclusions = load_exclusions()
    hold = holdout_origins("phase3", one_shot=True)
    raw = pd.read_parquet(FEATURES_PATH)
    frame = apply_exclusions(raw, exclusions)
    ts = pd.DatetimeIndex(frame["timestamp"])
    frame = frame[np.asarray(ts < hold[-1] + N_HORIZONS * HOUR)]
    index = pd.DatetimeIndex(frame["timestamp"]).sort_values()
    all_origins = daily_origins(index)

    cand = all_origins[(all_origins >= hold[0] - CAL_WINDOW) & (all_origins < hold[0])]
    cand = cand[~contaminated_origins(cand, exclusions)]
    to_forecast = cand.union(hold)
    print(f"holdout origins : {len(hold)} ({hold[0].date()} -> {hold[-1].date()})")
    print(
        f"pre-holdout calibration origins (clean): {len(cand)} "
        f"({cand[0].date()} -> {cand[-1].date()})"
    )

    matrix = build_design_matrix(frame, all_origins)
    records: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for k, o in enumerate(to_forecast, start=1):
        preds, _ = fit_predict_fold_quantiles(matrix, o)
        test = matrix[matrix["origin"] == o].sort_values("horizon")
        q = rearrange(np.column_stack([preds[n] for n in NAMES]))
        rec = pd.DataFrame(
            {
                "origin": o,
                "timestamp": test["timestamp"].to_numpy(),
                "horizon": test["horizon"].astype(int).to_numpy(),
                "y_true": test["y_true"].to_numpy(),
                "is_holdout": o in hold,
            }
        )
        for j, n in enumerate(NAMES):
            rec[n] = q[:, j]
        records.append(rec)
        if k % 7 == 0 or k == len(to_forecast):
            print(f"  forecast {k:2d}/{len(to_forecast)}  {time.perf_counter() - t0:5.0f}s")
    res = pd.concat(records, ignore_index=True)

    held: list[pd.DataFrame] = []
    for o in hold:
        g = res[res["origin"] == o].copy()
        pool = calibration_pool(res, o)
        n_pool = pool["origin"].nunique()
        g["cal_origins"] = n_pool
        g["cal_age_days"] = (o - pool["origin"]).dt.days.median() if n_pool else np.nan
        if n_pool < MIN_CAL_ORIGINS:
            for n in NAMES:
                g[f"{n}_cal"] = np.nan
                g[f"shift_{n}"] = np.nan
        else:
            shifts = {
                n: conformal_shift((pool["y_true"] - pool[n]).to_numpy(), DEFAULT_QUANTILES[n])
                for n in NAMES
            }
            qc = rearrange(np.column_stack([g[n].to_numpy() + shifts[n] for n in NAMES]))
            for j, n in enumerate(NAMES):
                g[f"{n}_cal"] = qc[:, j]
                g[f"shift_{n}"] = shifts[n]
        held.append(g)
    hr = trust_flags(pd.concat(held, ignore_index=True), raw, exclusions)
    hr.to_parquet(OUT_PATH, index=False)

    fmt = lambda v: f"{v:,.3f}"  # noqa: E731
    bt = pd.read_parquet(OUTPUT_DIR / "quantile_lgbm_conformal.parquet")
    bt = bt[bt["q50_cal"].notna()]
    table = (
        summarize_quantiles(
            pd.concat(
                [
                    _view(hr, "", "holdout_uncalibrated"),
                    _view(hr, "_cal", "holdout_calibrated"),
                    _view(bt, "_cal", "backtest_calibrated"),
                ],
                ignore_index=True,
            )
        )
        .set_index("model")
        .T
    )
    print("\n=== SEALED HOLDOUT (Aug 2026) vs backtest reference ===")
    print(table.to_string(float_format=fmt))

    load = frame.set_index("timestamp")["load_mw"]
    naive = load.reindex(pd.DatetimeIndex(hr["timestamp"]) - 168 * HOUR).to_numpy()
    ok = ~np.isnan(naive)
    print(
        f"\nweekly seasonal naive MAE (scale reference): {mae(hr['y_true'][ok], naive[ok]):.1f} MW "  # noqa: E501
        f"on {int(ok.sum())} hours"
    )

    hr["err"] = hr["y_true"] - hr["q50_cal"]
    hr["abs_err"] = hr["err"].abs()
    hr["inside"] = (hr["y_true"] >= hr["q10_cal"]) & (hr["y_true"] <= hr["q90_cal"])
    hr["width"] = hr["q90_cal"] - hr["q10_cal"]
    per = hr.groupby("origin").agg(
        mae_q50=("abs_err", "mean"),
        bias=("err", "mean"),
        int_cov=("inside", "mean"),
        width=("width", "mean"),
        cal_origins=("cal_origins", "first"),
        cal_age_days=("cal_age_days", "first"),
        flagged=("trust_low", "mean"),
    )
    per.index = per.index.tz_convert(TZ).strftime("%Y-%m-%d %a")
    print("\n=== per origin (calibrated) ===")
    print(per.to_string(float_format=lambda v: f"{v:,.2f}"))

    print("\n=== trust flag rates on holdout ===")
    for c in [*FLAG_COLS, "trust_low"]:
        print(f"  {c:17s} {hr[c].mean():6.1%}")
    print(f'\nwrote {OUT_PATH} -- now mark [holdout.phase3] status = "spent"')


if __name__ == "__main__":
    main()
