"""Tests for probabilistic metrics: hand-computed values, the defining
property of the pinball loss, and the identities that tie metrics together."""

import numpy as np
import pandas as pd
import pytest
from gridcast.models.metrics import (
    crossing_rate,
    interval_coverage,
    interval_score,
    mae,
    mean_interval_width,
    pinball_loss,
    quantile_coverage,
    rearrange,
    summarize_quantiles,
)


@pytest.mark.parametrize(
    "y,q,tau,expected",
    [
        (100.0, 90.0, 0.9, 9.0),  # under-predict by 10 at p90: 0.9 * 10
        (100.0, 90.0, 0.1, 1.0),  # under-predict by 10 at p10: 0.1 * 10
        (100.0, 110.0, 0.9, 1.0),  # over-predict by 10 at p90: 0.1 * 10
    ],
    ids=["p90-under", "p10-under", "p90-over"],
)
def test_pinball_hand_values(y, q, tau, expected):
    assert pinball_loss([y], [q], tau) == pytest.approx(expected)


def test_pinball_at_median_is_half_mae():
    rng = np.random.default_rng(1)
    y, q = rng.normal(12000, 1500, 500), rng.normal(12000, 1500, 500)
    assert pinball_loss(y, q, 0.5) == pytest.approx(0.5 * mae(y, q))


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_pinball_minimizer_is_the_quantile(tau):
    """The defining property: argmin_q E[L_tau(Y, q)] is the tau-quantile.
    On y = 1..100 the loss is flat on [100*tau, 100*tau + 1]."""
    y = np.arange(1, 101, dtype=float)
    grid = np.arange(0, 102, dtype=float)
    losses = [pinball_loss(y, np.full_like(y, q), tau) for q in grid]
    best = grid[int(np.argmin(losses))]
    k = round(100 * tau)
    assert best in (k, k + 1)


@pytest.mark.parametrize("tau", [0.0, 1.0])
def test_pinball_rejects_degenerate_tau(tau):
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], tau)


def test_coverage_and_width_hand_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert quantile_coverage(y, np.full(4, 2.5)) == pytest.approx(0.5)
    assert interval_coverage(y, np.full(4, 2.0), np.full(4, 3.0)) == pytest.approx(0.5)
    assert mean_interval_width(np.array([0.0, 1.0]), np.array([10.0, 5.0])) == pytest.approx(7.0)


def test_interval_score_equals_scaled_pinball_pair():
    rng = np.random.default_rng(2)
    n, alpha = 1000, 0.2
    y = rng.normal(0, 1, n)
    lo = rng.normal(-1, 0.5, n)
    hi = lo + rng.uniform(0.1, 3.0, n)
    lhs = interval_score(y, lo, hi, alpha)
    rhs = (2 / alpha) * (pinball_loss(y, lo, alpha / 2) + pinball_loss(y, hi, 1 - alpha / 2))
    assert lhs == pytest.approx(rhs)


def test_rearrangement_never_hurts_pointwise():
    rng = np.random.default_rng(3)
    n = 2000
    taus = np.array([0.1, 0.5, 0.9])
    y = rng.normal(0, 1, n)
    q = np.column_stack([rng.normal(m, 1, n) for m in (-1.0, 0.0, 1.0)])

    def row_loss(qm):
        d = y[:, None] - qm
        return np.maximum(taus * d, (taus - 1) * d).sum(axis=1)

    fixed = rearrange(q)
    assert crossing_rate(q) > 0  # the test data really does cross
    assert crossing_rate(fixed) == 0.0
    assert np.all(row_loss(fixed) <= row_loss(q) + 1e-12)


def test_summarize_quantiles_hand_values():
    df = pd.DataFrame(
        {
            "model": ["m", "m"],
            "y_true": [100.0, 120.0],  # row 1 inside, row 2 above p90 by 10
            "q10": [90.0, 90.0],
            "q50": [100.0, 100.0],
            "q90": [110.0, 110.0],
        }
    )
    r = summarize_quantiles(df).iloc[0]
    assert r["n"] == 2 and r["n_missing"] == 0
    assert r["cov_q90"] == pytest.approx(0.5)
    assert r["interval_cov"] == pytest.approx(0.5)
    assert r["interval_nominal"] == pytest.approx(0.8)
    assert r["width_mean"] == pytest.approx(20.0)
    assert r["pinball_q90"] == pytest.approx((1.0 + 9.0) / 2)
    assert r["interval_score"] == pytest.approx((20.0 + 120.0) / 2)  # 20 + 10*10
    assert r["crossing_rate"] == pytest.approx(0.0)
    assert r["mae_q50"] == pytest.approx(10.0)


def test_summarize_rejects_noncentral_interval():
    df = pd.DataFrame({"model": ["m"], "y_true": [1.0], "q10": [0.0], "q50": [1.0], "q90": [2.0]})
    with pytest.raises(ValueError):
        summarize_quantiles(df, interval=("q10", "q50"))
