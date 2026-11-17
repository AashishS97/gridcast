"""Tests for the CI regression gate: fold selection, comparability rules,
the pass/fail threshold, and one real scoring pass on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from gridcast.models.evaluation import backtest_folds
from gridcast.models.regression_gate import (
    EXIT_NOT_COMPARABLE,
    EXIT_PASS,
    EXIT_REGRESSION,
    REPORTED,
    compare,
    gate_folds,
    gate_metrics,
    max_rel_increase,
    score_folds,
)
from gridcast.models.splits import Fold


def _run(pinball_mean: float, **overrides) -> dict:
    run = {
        "fold_ids": [0, 10],
        "n_rows": 48,
        "features_sha256": "abc",
        "metrics": {name: 1.0 for name in REPORTED} | {"pinball_mean": pinball_mean},
    }
    return run | overrides


def test_gate_folds_are_a_pinned_subset_covering_every_weekday():
    folds = gate_folds()
    pinned = {f.fold_id: f for f in backtest_folds()}
    assert all(pinned[f.fold_id] == f for f in folds)  # subset, never new folds
    assert len({f.origin.dayofweek for f in folds}) == 7


def test_gate_tolerance_is_configured_and_sane():
    assert 0.0 <= max_rel_increase() < 1.0


def test_compare_passes_within_tolerance_and_on_improvement():
    assert compare(_run(100.0), _run(101.9), 0.02)[0] == EXIT_PASS
    assert compare(_run(100.0), _run(90.0), 0.02)[0] == EXIT_PASS


def test_compare_fails_beyond_tolerance():
    code, lines = compare(_run(100.0), _run(102.1), 0.02)
    assert code == EXIT_REGRESSION
    assert any(line.startswith("FAIL") for line in lines)


@pytest.mark.parametrize(
    "override",
    [{"fold_ids": [0, 20]}, {"n_rows": 24}, {"features_sha256": "different"}],
)
def test_compare_refuses_runs_that_are_not_comparable(override):
    # a better number on different folds or data must not read as a pass
    code, _ = compare(_run(100.0), _run(50.0, **override), 0.02)
    assert code == EXIT_NOT_COMPARABLE


def test_score_folds_and_metrics_on_synthetic_data(synthetic_frame):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    origins = [pd.Timestamp("2024-08-01", tz="UTC"), pd.Timestamp("2024-09-01", tz="UTC")]
    folds = [Fold(i, start, o, o + pd.Timedelta("24h")) for i, o in enumerate(origins)]
    results = score_folds(synthetic_frame, folds)
    assert len(results) == 48
    assert results["fold"].tolist() == [0] * 24 + [1] * 24
    q = results[["q10", "q50", "q90"]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()  # rearranged: no crossing
    metrics = gate_metrics(results)
    assert set(metrics) == set(REPORTED)
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["crossing_rate"] == 0.0


def test_gate_metrics_refuses_unscorable_rows(synthetic_frame):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    origin = pd.Timestamp("2024-08-01", tz="UTC")
    results = score_folds(synthetic_frame, [Fold(0, start, origin, origin + pd.Timedelta("24h"))])
    results.loc[0, "y_true"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        gate_metrics(results)
