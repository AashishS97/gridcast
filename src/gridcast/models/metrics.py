"""Forecast error metrics: point (Phase 2) and probabilistic (Phase 3).

Point
-----
MAE: average absolute miss, in MW. Robust, interpretable.
RMSE: squares errors before averaging, so large misses dominate. The
    MAE/RMSE gap is diagnostic: a large gap means rare-but-severe failures.
MAPE: scale-free percentage. Safe for national load (never near zero);
    guarded here anyway.

Probabilistic
-------------
Pinball loss L_tau(y, q) = max(tau*(y-q), (tau-1)*(y-q)). Its expected value
is minimized exactly at the tau-quantile (d/dq E[L] = P(Y<=q) - tau), so it
is a PROPER scoring rule: rewards calibration and sharpness jointly. At
tau=0.5 it equals MAE/2. Headline metric for quantile forecasts.
Coverage: fraction of actuals <= q (per quantile) or inside [lo, hi]
    (interval). Calibration diagnostic only -- gameable on its own
    (climatological intervals are well-calibrated and useless).
Width: sharpness diagnostic.
Interval score (Gneiting & Raftery 2007): width + (2/alpha) * misses outside.
    Proper for central (1-alpha) intervals; equals
    (2/alpha) * (L_{alpha/2}(lo) + L_{1-alpha/2}(hi)).
Crossing / rearrangement: independently fitted quantiles can cross.
    Sorting each row never increases the summed pinball loss, pointwise:
    L_tau(y,q) = tau*(y-q) + max(q-y, 0); the max terms are
    permutation-invariant and -sum(tau_k * q_k) is minimized by the sorted
    assignment (rearrangement inequality).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_QUANTILES: dict[str, float] = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


# --------------------------------------------------------------------- point
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(np.abs(y_true) < 1e-9):
        raise ValueError("MAPE undefined: y_true contains (near-)zero values")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def summarize(results: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Aggregate a long results frame into MAE/RMSE/MAPE per group.

    `results` must have columns y_true, y_pred, plus whatever is in `by`
    (default: ["model"]). Rows with NaN in y_true or y_pred are dropped and
    counted in n_missing so coverage problems stay visible instead of
    silently shrinking the average.
    """
    by = by if by is not None else ["model"]
    out = []
    for keys, group in results.groupby(by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ok = group.dropna(subset=["y_true", "y_pred"])
        row = dict(zip(by, keys, strict=False))
        row["n"] = len(ok)
        row["n_missing"] = len(group) - len(ok)
        if len(ok) > 0:
            row["mae"] = mae(ok["y_true"].values, ok["y_pred"].values)
            row["rmse"] = rmse(ok["y_true"].values, ok["y_pred"].values)
            row["mape"] = mape(ok["y_true"].values, ok["y_pred"].values)
        else:
            row["mae"] = row["rmse"] = row["mape"] = np.nan
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------- probabilistic
def pinball_loss(y_true: np.ndarray, y_q: np.ndarray, tau: float) -> float:
    """Mean pinball (quantile) loss of predictions y_q for quantile level tau."""
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    d = np.asarray(y_true, dtype=float) - np.asarray(y_q, dtype=float)
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def quantile_coverage(y_true: np.ndarray, y_q: np.ndarray) -> float:
    """Fraction of actuals at or below the predicted quantile (target: tau)."""
    return float(np.mean(np.asarray(y_true, dtype=float) <= np.asarray(y_q, dtype=float)))


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of actuals inside the closed interval [lo, hi]."""
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lo, dtype=float)) & (y <= np.asarray(hi, dtype=float))))


def mean_interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)))


def interval_score(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Mean interval score for a central (1 - alpha) interval. Lower is better."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    below = (2.0 / alpha) * np.maximum(lo - y, 0.0)
    above = (2.0 / alpha) * np.maximum(y - hi, 0.0)
    return float(np.mean((hi - lo) + below + above))


def crossing_rate(q: np.ndarray) -> float:
    """Fraction of rows where quantile columns (ordered by tau) are not
    non-decreasing."""
    q = np.asarray(q, dtype=float)
    return float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1)))


def rearrange(q: np.ndarray) -> np.ndarray:
    """Sort each row: removes crossing, never increases summed pinball loss."""
    return np.sort(np.asarray(q, dtype=float), axis=1)


def summarize_quantiles(
    results: pd.DataFrame,
    by: list[str] | None = None,
    quantiles: dict[str, float] | None = None,
    interval: tuple[str, str] = ("q10", "q90"),
) -> pd.DataFrame:
    """Aggregate a long results frame of quantile forecasts per group.

    `results` needs y_true plus one column per quantile name (default
    q10/q50/q90) and the `by` columns. NaN rows are dropped and counted in
    n_missing, as in summarize(). `interval` must be central (lo = alpha/2).
    """
    by = by if by is not None else ["model"]
    quantiles = quantiles if quantiles is not None else DEFAULT_QUANTILES
    names = sorted(quantiles, key=quantiles.__getitem__)
    lo_name, hi_name = interval
    nominal = quantiles[hi_name] - quantiles[lo_name]
    alpha = 1.0 - nominal
    if not np.isclose(quantiles[lo_name], alpha / 2.0):
        raise ValueError(f"interval {interval} is not central: tau_lo must equal alpha/2")
    median = [n for n in names if np.isclose(quantiles[n], 0.5)]

    out = []
    for keys, group in results.groupby(by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ok = group.dropna(subset=["y_true", *names])
        row: dict = dict(zip(by, keys, strict=False))
        row["n"] = len(ok)
        row["n_missing"] = len(group) - len(ok)
        if len(ok) == 0:
            out.append(row)
            continue
        y = ok["y_true"].to_numpy(dtype=float)
        losses = []
        for name in names:
            q = ok[name].to_numpy(dtype=float)
            loss = pinball_loss(y, q, quantiles[name])
            losses.append(loss)
            row[f"pinball_{name}"] = loss
            row[f"cov_{name}"] = quantile_coverage(y, q)
        row["pinball_mean"] = float(np.mean(losses))
        lo = ok[lo_name].to_numpy(dtype=float)
        hi = ok[hi_name].to_numpy(dtype=float)
        row["interval_nominal"] = nominal
        row["interval_cov"] = interval_coverage(y, lo, hi)
        row["width_mean"] = mean_interval_width(lo, hi)
        row["interval_score"] = interval_score(y, lo, hi, alpha)
        row["crossing_rate"] = crossing_rate(ok[names].to_numpy(dtype=float))
        if median:
            row["mae_q50"] = mae(y, ok[median[0]].to_numpy(dtype=float))
        out.append(row)
    return pd.DataFrame(out)
