"""Point-forecast error metrics.

MAE: average absolute miss, in MW. Robust, interpretable.
RMSE: squares errors before averaging, so large misses dominate. The
    MAE/RMSE gap is diagnostic: a large gap means rare-but-severe failures.
MAPE: scale-free percentage. Safe for national load (never near zero);
    guarded here anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
