"""Direct multi-horizon LightGBM on the leak-proof design matrix.

One model for all 24 horizons, with `horizon` as a feature: horizons share
almost all structure (h=13 vs h=14 is nearly the same problem), one model
pools all training data, and trees split on horizon where behavior differs.

Hyperparameters are conservative FIXED choices, not tuned. Tuning by
re-running the backtest and keeping the best table would make the folds the
training signal for the config — evaluation-selection leakage — and the
reported score would overstate fresh-data performance. Any future tuning
happens on dev folds only and is verified once on the untouched holdout.

Early stopping uses the LAST 60 days of each fold's training origins,
chronologically — random validation rows would be near-duplicates of
adjacent training rows and make the stopping decision itself leaky.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast.models.ml_features import feature_columns

VAL_SPAN = pd.Timedelta("60D")

LGBM_PARAMS: dict = {
    "objective": "l1",  # optimize MAE: robust, and it's our headline metric
    "num_leaves": 63,  # moderate interaction depth (hour x dow x temp)
    "learning_rate": 0.05,  # slow, with many rounds + early stopping
    "n_estimators": 2000,  # ceiling; early stopping picks the real count
    "min_child_samples": 40,  # don't memorize rare hours
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}


def fit_predict_fold(
    matrix: pd.DataFrame,
    fold_origin: pd.Timestamp,
    drop_features: tuple[str, ...] = (),
) -> tuple[np.ndarray, int]:
    """Train on all origins strictly before `fold_origin`, predict its 24 rows.

    drop_features: feature columns to exclude (ablation runs).
    Returns (predictions aligned to horizon 1..24, best_iteration).
    """
    feats = [c for c in feature_columns(matrix) if c not in drop_features]

    train = matrix[(matrix["origin"] < fold_origin) & matrix["y_true"].notna()]
    if train.empty:
        raise ValueError(f"no training rows before origin {fold_origin}")

    val_cut = fold_origin - VAL_SPAN
    core = train[train["origin"] < val_cut]
    val = train[train["origin"] >= val_cut]
    if core.empty or val.empty:
        raise ValueError(
            f"fold at {fold_origin}: cannot carve a {VAL_SPAN} validation span "
            f"from {train['origin'].min()} .. {train['origin'].max()}"
        )

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        core[feats],
        core["y_true"],
        eval_set=[(val[feats], val["y_true"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    test = matrix[matrix["origin"] == fold_origin].sort_values("horizon")
    if len(test) != 24:
        raise ValueError(f"expected 24 rows for origin {fold_origin}, got {len(test)}")
    preds = model.predict(test[feats], num_iteration=model.best_iteration_)
    return np.asarray(preds, dtype=float), int(model.best_iteration_ or 0)
