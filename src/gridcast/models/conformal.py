"""Conformal calibration of quantile forecasts (asymmetric CQR-style).

For each quantile tau, shift q_tau by the tau-quantile of RECENT
out-of-sample residuals (y - q_tau):
    q_tau_cal = q_tau + Q_tau(y - q_tau)
p90 residuals too often positive -> p90 moves up; p10 too often negative ->
p10 moves down (interval widens); p50 shifts by the median residual (bias
removal). Romano et al. 2019 (CQR), asymmetric variant. Finite-sample
order statistics are chosen conservatively (outward).

Calibration data: earlier folds whose test windows ended BEFORE this fold's
origin (5D step -> previous fold's actuals are 4 days old: available in
production). Trailing window of W folds.

Declared before seeing results: PRIMARY W=12 (~60 days); W=24 is reported
as a sensitivity check only -- picking the better one would be evaluation-
selection leakage. First MIN_FOLDS folds stay uncalibrated.

Honest limits: the conformal coverage guarantee needs exchangeability, which
time series violate -- this is an approximation, verified empirically. It
fixes AVERAGE calibration, not conditional (e.g. cold-tail) calibration.

Run:  uv run python -m gridcast.models.conformal
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast.models.backtest import OUTPUT_DIR
from gridcast.models.metrics import DEFAULT_QUANTILES, rearrange, summarize_quantiles

WINDOW_FOLDS = 12  # primary, declared in advance
SENSITIVITY_WINDOW = 24
MIN_FOLDS = 6
IN_PATH = OUTPUT_DIR / "quantile_lgbm.parquet"
OUT_PATH = OUTPUT_DIR / "quantile_lgbm_conformal.parquet"


def conformal_shift(residuals: np.ndarray, tau: float) -> float:
    """tau-level order statistic of residuals (y - q), chosen conservatively:
    upper quantiles round the rank up, lower quantiles round it down."""
    r = np.sort(np.asarray(residuals, dtype=float))
    n = len(r)
    if n == 0:
        raise ValueError("no residuals to calibrate on")
    if np.isclose(tau, 0.5):
        return float(np.median(r))
    if tau > 0.5:
        k = min(n, int(np.ceil((n + 1) * tau)))
    else:
        k = max(1, int(np.floor((n + 1) * tau)))
    return float(r[k - 1])


def calibrate_backtest(
    results: pd.DataFrame,
    quantiles: dict[str, float] = DEFAULT_QUANTILES,
    window_folds: int = WINDOW_FOLDS,
    min_folds: int = MIN_FOLDS,
) -> pd.DataFrame:
    """Add <name>_cal and shift_<name> columns, fold by fold, using only
    earlier folds whose last target hour is strictly before this origin."""
    names = sorted(quantiles, key=quantiles.__getitem__)
    res = results.sort_values(["origin", "horizon"])
    groups = {o: g for o, g in res.groupby("origin")}
    origins = sorted(groups)

    out: list[pd.DataFrame] = []
    for i, origin in enumerate(origins):
        g = groups[origin].copy()
        past = [groups[p] for p in origins[max(0, i - window_folds) : i]]
        past = [p for p in past if p["timestamp"].max() < origin]
        if len(past) < min_folds:
            for n in names:
                g[f"{n}_cal"] = np.nan
                g[f"shift_{n}"] = np.nan
        else:
            cal = pd.concat(past).dropna(subset=["y_true"])
            shifts = {
                n: conformal_shift((cal["y_true"] - cal[n]).to_numpy(), quantiles[n]) for n in names
            }
            q = rearrange(np.column_stack([g[n].to_numpy() + shifts[n] for n in names]))
            for j, n in enumerate(names):
                g[f"{n}_cal"] = q[:, j]
                g[f"shift_{n}"] = shifts[n]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _view(df: pd.DataFrame, suffix: str, label: str) -> pd.DataFrame:
    names = sorted(DEFAULT_QUANTILES, key=DEFAULT_QUANTILES.__getitem__)
    v = df[["y_true", "timestamp", "hour_local", *[f"{n}{suffix}" for n in names]]].copy()
    v.columns = ["y_true", "timestamp", "hour_local", *names]
    v["model"] = label
    return v


def main() -> None:
    raw = pd.read_parquet(IN_PATH)
    primary = calibrate_backtest(raw, window_folds=WINDOW_FOLDS)
    sens = calibrate_backtest(raw, window_folds=SENSITIVITY_WINDOW)

    common = primary["q50_cal"].notna() & sens["q50_cal"].notna()
    n_folds = primary.loc[common, "origin"].nunique()
    print(f"calibrated folds compared: {n_folds} (first {MIN_FOLDS} have no history)")

    both = pd.concat(
        [
            _view(primary[common], "", "uncalibrated"),
            _view(primary[common], "_cal", f"conformal_w{WINDOW_FOLDS}"),
            _view(sens[common], "_cal", f"conformal_w{SENSITIVITY_WINDOW} (sensitivity)"),
        ],
        ignore_index=True,
    )
    table = summarize_quantiles(both).set_index("model").T
    print("\n=== same folds: uncalibrated vs conformal (primary W=12) ===")
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

    both["inside"] = (both["y_true"] >= both["q10"]) & (both["y_true"] <= both["q90"])
    both["err"] = both["y_true"] - both["q50"]
    local = both["timestamp"].dt.tz_convert("Europe/Amsterdam")
    both["month"] = local.dt.strftime("%Y-%m")
    keep = ["uncalibrated", f"conformal_w{WINDOW_FOLDS}"]
    sub = both[both["model"].isin(keep)]

    for key in ["month", "hour_local"]:
        t = sub.groupby([key, "model"]).agg(int_cov=("inside", "mean"), bias=("err", "mean"))
        t = t.unstack("model")
        print(f"\n=== interval coverage and median bias (MW) by {key} ===")
        print(t.to_string(float_format=lambda v: f"{v:,.2f}"))

    primary.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} (primary W={WINDOW_FOLDS})")


if __name__ == "__main__":
    main()
