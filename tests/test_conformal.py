"""Tests for conformal calibration: order statistics, coverage restoration
on exchangeable data, and a perturb-the-future leakage test."""

import numpy as np
import pandas as pd
import pytest
from gridcast.models.conformal import calibrate_backtest, conformal_shift
from gridcast.models.metrics import crossing_rate, quantile_coverage


def test_conformal_shift_order_statistics():
    r = np.arange(1, 10, dtype=float)  # n = 9, so (n+1)*tau is an exact rank
    assert conformal_shift(r, 0.9) == 9.0  # ceil(10*0.9) = 9th smallest
    assert conformal_shift(r, 0.1) == 1.0  # floor(10*0.1) = 1st smallest
    assert conformal_shift(r, 0.5) == 5.0


def test_calibration_restores_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    y_cal, y_new = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    wrong = {0.1: -0.5, 0.5: 0.3, 0.9: 0.5}  # biased and far too narrow
    for tau, q in wrong.items():
        shift = conformal_shift(y_cal - q, tau)
        assert quantile_coverage(y_new, np.full(5000, q + shift)) == pytest.approx(tau, abs=0.02)


def _results(n_folds: int = 20, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k, origin in enumerate(pd.date_range("2025-01-01", periods=n_folds, freq="5D", tz="UTC")):
        y = rng.normal(12000, 500, 24)
        q50 = y + rng.normal(150, 300, 24)
        rows.append(
            pd.DataFrame(
                {
                    "fold": k,
                    "model": "m",
                    "origin": origin,
                    "timestamp": origin + pd.to_timedelta(np.arange(24), unit="h"),
                    "horizon": np.arange(1, 25),
                    "y_true": y,
                    "q10": q50 - 200,
                    "q50": q50,
                    "q90": q50 + 200,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_calibration_never_uses_the_future():
    base = _results()
    cal_a = calibrate_backtest(base)
    changed = base.copy()
    cut = changed["origin"].unique()[15]
    changed.loc[changed["origin"] == cut, "y_true"] += 50_000  # corrupt fold 15
    cal_b = calibrate_backtest(changed)
    upto = (cal_a["origin"] <= cut).to_numpy()
    for col in ["q10_cal", "q50_cal", "q90_cal"]:
        np.testing.assert_allclose(cal_a.loc[upto, col], cal_b.loc[upto, col], equal_nan=True)
    assert not np.allclose(cal_a.loc[~upto, "q90_cal"], cal_b.loc[~upto, "q90_cal"])


def test_first_folds_uncalibrated_and_no_crossing():
    cal = calibrate_backtest(_results(), min_folds=6)
    first = cal["origin"] < cal["origin"].unique()[6]
    assert cal.loc[first, "q50_cal"].isna().all()
    done = cal.loc[~first, ["q10_cal", "q50_cal", "q90_cal"]].to_numpy()
    assert not np.isnan(done).any()
    assert crossing_rate(done) == 0.0
