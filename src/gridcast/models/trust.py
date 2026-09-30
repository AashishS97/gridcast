"""Trust flag: conditions under which the system warns its own forecast is
unreliable. Every signal uses only information available AT FORECAST TIME,
and each comes from a Phase 3 failure-analysis finding:

cold         target-hour temperature in the bottom COLD_PCT of the model's
             training history (timestamps before origin). Bias grew
             monotonically into the cold tail (+164 MW 2-10%, +383 MW 0-2%)
             although no hour was outside the training range.
calendar     holiday, bridge day (weekday between a holiday and a weekend),
             Dec 24 or Dec 31 -- rare days with noisy point forecasts; the
             bridge day is not in the holiday feature at all.
instability  |conformal median shift| > SHIFT_FRAC of the forecast: the model
             was recently off by an episode-sized amount (monthly episodes
             were 2.2-2.7%); also catches post-episode overcorrection.
integrity    any load hour in the LOOKBACK_HOURS window before origin is
             missing or excluded: corrupted inputs need an OOD flag, which
             intervals cannot provide.

Thresholds are round numbers declared before evaluation, but motivated by
the same backtest they are evaluated on -- backtest results are
descriptive; the sealed holdout is the out-of-sample check. `cold` uses
observed temperature (perfect prognosis); in production it runs on a
temperature forecast. No flag can anticipate shocks nobody could see.

Run:  uv run python -m gridcast.models.trust
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from gridcast.models.backtest import FEATURES_PATH, OUTPUT_DIR
from gridcast.models.exclusions import (
    LOOKBACK_HOURS,
    Exclusion,
    apply_exclusions,
    load_exclusions,
)
from gridcast.models.metrics import interval_coverage, interval_score, mae

HOUR = pd.Timedelta("1h")
TZ = "Europe/Amsterdam"
COLD_PCT = 0.10
SHIFT_FRAC = 0.02
SPECIAL_DAYS = {(12, 24), (12, 31)}
FLAG_COLS = ["flag_cold", "flag_calendar", "flag_instability", "flag_integrity"]
IN_PATH = OUTPUT_DIR / "quantile_lgbm_conformal.parquet"
OUT_PATH = OUTPUT_DIR / "quantile_lgbm_trust.parquet"


def _indexed(features: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in features.columns:
        features = features.set_index("timestamp")
    return features.sort_index()


def temperature_percentile(rows: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    """Percentile of each row's target-hour temperature within the
    temperatures observed strictly before that row's origin."""
    temps = features["temperature_2m"].dropna()
    t_all = features["temperature_2m"].reindex(pd.DatetimeIndex(rows["timestamp"])).to_numpy()
    out = np.full(len(rows), np.nan)
    for origin, g in rows.groupby("origin"):
        pos = rows.index.get_indexer(g.index)
        hist = np.sort(temps[temps.index < origin].to_numpy())
        if len(hist) == 0:
            continue
        t = t_all[pos]
        pct = np.searchsorted(hist, t, side="right") / len(hist)
        out[pos] = np.where(np.isnan(t), np.nan, pct)
    return out


def irregular_calendar(timestamps: pd.DatetimeIndex, features: pd.DataFrame) -> np.ndarray:
    """Holiday, bridge day, Dec 24 or Dec 31, judged on the LOCAL date."""
    f_local = features.index.tz_convert(TZ)
    holidays = set(f_local[features["is_holiday"].to_numpy() == 1].date)

    def irregular(d: date) -> bool:
        if d in holidays or (d.month, d.day) in SPECIAL_DAYS:
            return True
        if d.weekday() >= 5:
            return False
        prev, nxt = d - timedelta(days=1), d + timedelta(days=1)
        return (prev in holidays and nxt.weekday() >= 5) or (
            nxt in holidays and prev.weekday() >= 5
        )

    days = pd.DatetimeIndex(timestamps).tz_convert(TZ).date
    cache = {d: irregular(d) for d in set(days)}
    return np.array([cache[d] for d in days], dtype=bool)


def recent_instability(shift_q50: pd.Series, q50: pd.Series) -> np.ndarray:
    """|conformal median shift| above SHIFT_FRAC of the forecast. Rows
    without a calibration shift (NaN) are not flagged by this signal."""
    s = np.abs(np.asarray(shift_q50, dtype=float))
    q = np.asarray(q50, dtype=float)
    return np.nan_to_num(s > SHIFT_FRAC * q, nan=False).astype(bool)


def integrity_breach(
    rows: pd.DataFrame, features: pd.DataFrame, exclusions: list[Exclusion]
) -> np.ndarray:
    """True if any load hour in [origin - LOOKBACK_HOURS, origin) is missing
    or masked by an exclusion."""
    load = apply_exclusions(features, exclusions)["load_mw"]
    out = np.zeros(len(rows), dtype=bool)
    for origin, g in rows.groupby("origin"):
        pos = rows.index.get_indexer(g.index)
        window = load[(load.index >= origin - LOOKBACK_HOURS * HOUR) & (load.index < origin)]
        out[pos] = len(window) < LOOKBACK_HOURS or bool(window.isna().any())
    return out


def trust_flags(
    rows: pd.DataFrame, features: pd.DataFrame, exclusions: list[Exclusion]
) -> pd.DataFrame:
    """Copy of `rows` (needs origin, timestamp, q50, shift_q50) with one
    column per signal, trust_low (any fired) and trust_reasons."""
    out = rows.reset_index(drop=True).copy()
    features = _indexed(features)
    out["temp_pct"] = temperature_percentile(out, features)
    out["flag_cold"] = np.nan_to_num(out["temp_pct"] < COLD_PCT, nan=False).astype(bool)
    out["flag_calendar"] = irregular_calendar(pd.DatetimeIndex(out["timestamp"]), features)
    out["flag_instability"] = recent_instability(out["shift_q50"], out["q50"])
    out["flag_integrity"] = integrity_breach(out, features, exclusions)
    m = out[FLAG_COLS].to_numpy()
    out["trust_low"] = m.any(axis=1)
    names = np.array([c.removeprefix("flag_") for c in FLAG_COLS])
    out["trust_reasons"] = [",".join(names[r]) for r in m]
    return out


def _summary(g: pd.DataFrame) -> pd.Series:
    y, lo, med, hi = (g[c].to_numpy() for c in ["y_true", "q10_cal", "q50_cal", "q90_cal"])
    return pd.Series(
        {
            "n": len(g),
            "int_cov": interval_coverage(y, lo, hi),
            "bias_mw": float(np.mean(y - med)),
            "mae_q50": mae(y, med),
            "width": float(np.mean(hi - lo)),
            "interval_score": interval_score(y, lo, hi, alpha=0.2),
        }
    )


def main() -> None:
    rows = pd.read_parquet(IN_PATH)
    rows = rows[rows["q50_cal"].notna()].reset_index(drop=True)
    features = _indexed(pd.read_parquet(FEATURES_PATH))
    flagged = trust_flags(rows, features, load_exclusions())
    fmt = lambda v: f"{v:,.2f}"  # noqa: E731

    print(f"calibrated rows evaluated: {len(flagged)} ({flagged['origin'].nunique()} folds)")
    print("\n=== flag rates ===")
    for c in [*FLAG_COLS, "trust_low"]:
        print(f"  {c:17s} {flagged[c].mean():6.1%}")

    print("\n=== flagged vs unflagged (calibrated forecasts) ===")
    print(
        flagged.groupby("trust_low")
        .apply(_summary, include_groups=False)
        .to_string(float_format=fmt)
    )

    print("\n=== per signal (rows where it fires) ===")
    per = {c: _summary(flagged[flagged[c]]) for c in FLAG_COLS if flagged[c].any()}
    per["all rows"] = _summary(flagged)
    print(pd.DataFrame(per).T.to_string(float_format=fmt))

    err = (flagged["y_true"] - flagged["q50_cal"]).abs()
    big = err >= err.quantile(0.90)
    outside = ~(
        (flagged["y_true"] >= flagged["q10_cal"]) & (flagged["y_true"] <= flagged["q90_cal"])
    )
    rate = flagged["trust_low"].mean()
    for label, mask in [("worst 10% abs errors", big), ("hours outside the interval", outside)]:
        recall = flagged.loc[mask, "trust_low"].mean()
        print(f"\n{label}: {recall:.1%} flagged (flag rate {rate:.1%}, lift {recall / rate:.2f})")

    s_un = _summary(flagged[~flagged["trust_low"]])["interval_score"]
    s_fl = _summary(flagged[flagged["trust_low"]])["interval_score"]
    lift_big = flagged.loc[big, "trust_low"].mean() / rate
    print("\n=== declared success criteria ===")
    print(f"  flag rate <= 30%              : {rate:.1%}  {'PASS' if rate <= 0.30 else 'FAIL'}")
    print(
        f"  interval score ratio >= 1.3x  : {s_fl / s_un:.2f}x  {'PASS' if s_fl / s_un >= 1.3 else 'FAIL'}"  # noqa: E501
    )
    print(
        f"  big-miss lift >= 1.5          : {lift_big:.2f}  {'PASS' if lift_big >= 1.5 else 'FAIL'}"
    )

    flagged.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
