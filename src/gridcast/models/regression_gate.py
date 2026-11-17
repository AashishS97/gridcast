"""CI regression gate: quantile backtest on a pinned SUBSET of the dev folds.

Unit tests say the code does what it was written to do. They cannot say a
change made the forecasts worse. This module re-scores the quantile model
on a fixed set of folds and a fixed data snapshot, writes the metrics to
JSON, and (with --baseline) fails when mean pinball loss degrades by more
than the tolerance pinned in configs/evaluation.toml.

What it is NOT:
- not a performance claim. A subset of folds is a regression detector;
  headline numbers stay with the full 140-fold backtest.
- not a holdout. It runs on dev folds only; the spent holdouts are never
  touched, so repeatedly passing this gate leaks nothing from them.
- not the served model. It scores the raw (rearranged) quantiles, before
  conformal calibration.

Folds are every `stride`-th pinned fold. With a 5-day fold step, a stride
of 10 puts gate folds 50 days apart: 50 mod 7 = 1, so consecutive gate
folds land on consecutive weekdays (a stride of 7 would put every fold on
the same weekday -- the fold-step aliasing bug from Phase 2).

Training is exactly quantile_lgbm.fit_predict_fold_quantiles; nothing is
re-implemented here, so the gate cannot drift from the real backtest.

Run:  uv run python -m gridcast.models.regression_gate --out gate.json
      uv run python -m gridcast.models.regression_gate --out gate.json \
          --baseline configs/gate_baseline.json
Exit: 0 pass (or no baseline given), 1 regression, 2 baseline not comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast.models.backtest import FEATURES_PATH
from gridcast.models.evaluation import DEFAULT_PATH, backtest_folds, load_config
from gridcast.models.exclusions import apply_exclusions, load_exclusions
from gridcast.models.metrics import DEFAULT_QUANTILES, rearrange, summarize_quantiles
from gridcast.models.ml_features import build_design_matrix, daily_origins
from gridcast.models.quantile_lgbm import fit_predict_fold_quantiles
from gridcast.models.splits import Fold

NAMES = sorted(DEFAULT_QUANTILES, key=DEFAULT_QUANTILES.__getitem__)
GATED_METRIC = "pinball_mean"
REPORTED = (
    "pinball_mean",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "mae_q50",
    "interval_cov",
    "width_mean",
    "crossing_rate",
)
EXIT_PASS, EXIT_REGRESSION, EXIT_NOT_COMPARABLE = 0, 1, 2


def gate_folds(path: str | Path = DEFAULT_PATH) -> list[Fold]:
    """Every `stride`-th pinned dev fold. Fails loudly if the count drifts."""
    cfg = load_config(path)["ci_gate"]
    stride = int(cfg["stride"])
    if stride < 1:
        raise ValueError(f"ci_gate.stride must be >= 1, got {stride}")
    folds = backtest_folds(path)[::stride]
    if len(folds) != cfg["expected_folds"]:
        raise ValueError(f"ci_gate yields {len(folds)} folds, expected {cfg['expected_folds']}")
    return folds


def max_rel_increase(path: str | Path = DEFAULT_PATH) -> float:
    """Allowed relative increase of the gated metric (0.02 = +2%)."""
    value = float(load_config(path)["ci_gate"]["max_rel_increase"])
    if not 0.0 <= value < 1.0:
        raise ValueError(f"ci_gate.max_rel_increase must be in [0, 1), got {value}")
    return value


def score_folds(frame: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Long results frame (fold, origin, timestamp, horizon, y_true, q10,
    q50, q90) for `folds`. Same frame cut and same training call as
    quantile_lgbm.main, so a fold scored here equals that fold in the full
    backtest."""
    ts = pd.DatetimeIndex(frame["timestamp"])
    frame = frame[np.asarray(ts < folds[-1].test_end)]
    index = pd.DatetimeIndex(frame["timestamp"]).sort_values()
    matrix = build_design_matrix(frame, daily_origins(index))

    records: list[pd.DataFrame] = []
    for fold in folds:
        preds, _ = fit_predict_fold_quantiles(matrix, fold.origin)
        test = matrix[matrix["origin"] == fold.origin].sort_values("horizon")
        fixed = rearrange(np.column_stack([preds[n] for n in NAMES]))
        rec = pd.DataFrame(
            {
                "fold": fold.fold_id,
                "origin": fold.origin,
                "timestamp": test["timestamp"].to_numpy(),
                "horizon": test["horizon"].astype(int).to_numpy(),
                "y_true": test["y_true"].to_numpy(),
            }
        )
        for j, n in enumerate(NAMES):
            rec[n] = fixed[:, j]
        records.append(rec)
    return pd.concat(records, ignore_index=True)


def gate_metrics(results: pd.DataFrame) -> dict[str, float]:
    """Pooled metrics over all scored rows. Raises if any row is unscorable:
    a silently shrinking evaluation set would make the comparison void."""
    row = summarize_quantiles(results.assign(model="gate")).iloc[0]
    if int(row["n_missing"]) != 0:
        raise ValueError(f"{int(row['n_missing'])} gate row(s) have NaN y_true or predictions")
    return {name: float(row[name]) for name in REPORTED}


def file_sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def compare(baseline: dict, current: dict, tolerance: float) -> tuple[int, list[str]]:
    """(exit code, report lines). Comparable only if both runs scored the
    same folds on byte-identical data; otherwise the difference in the
    metric means nothing and the baseline must be regenerated on purpose."""
    lines: list[str] = []
    for key in ("fold_ids", "n_rows", "features_sha256"):
        if baseline[key] != current[key]:
            lines.append(f"NOT COMPARABLE: {key} differs (baseline {baseline[key]!r})")
            lines.append(f"                {' ' * len(key)}          (current  {current[key]!r})")
    if lines:
        return EXIT_NOT_COMPARABLE, lines

    lines.append(f"{'metric':<14} {'baseline':>12} {'current':>12} {'change':>9}")
    for name in REPORTED:
        base, cur = baseline["metrics"][name], current["metrics"][name]
        change = f"{(cur - base) / base:+.2%}" if base else "n/a"
        mark = "  <- gated" if name == GATED_METRIC else ""
        lines.append(f"{name:<14} {base:>12.4f} {cur:>12.4f} {change:>9}{mark}")

    base = baseline["metrics"][GATED_METRIC]
    cur = current["metrics"][GATED_METRIC]
    rel = (cur - base) / base
    if rel > tolerance:
        lines.append(f"FAIL: {GATED_METRIC} rose {rel:+.2%}, tolerance is +{tolerance:.2%}")
        return EXIT_REGRESSION, lines
    lines.append(f"PASS: {GATED_METRIC} changed {rel:+.2%}, tolerance is +{tolerance:.2%}")
    return EXIT_PASS, lines


def reference_metrics(path: Path, fold_ids: list[int]) -> dict[str, float]:
    """The same metrics from a stored full-backtest parquet, restricted to
    the gate folds. A one-off check that the gate reproduces the backtest."""
    ref = pd.read_parquet(path)
    ref = ref[ref["fold"].isin(fold_ids)]
    if sorted(ref["fold"].unique().tolist()) != sorted(fold_ids):
        raise ValueError(f"{path} does not contain every gate fold")
    return gate_metrics(ref[["y_true", *NAMES]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", type=Path, default=FEATURES_PATH, help="features parquet")
    ap.add_argument("--out", type=Path, required=True, help="where to write this run's JSON")
    ap.add_argument("--baseline", type=Path, default=None, help="compare against this JSON")
    ap.add_argument("--reference", type=Path, default=None, help="stored full-backtest parquet")
    args = ap.parse_args()

    folds = gate_folds()
    fold_ids = [f.fold_id for f in folds]
    weekdays = sorted({f.origin.day_name()[:3] for f in folds})
    print(f"gate folds: {len(folds)} ({folds[0].origin.date()} -> {folds[-1].origin.date()})")
    print(f"fold ids: {fold_ids}")
    print(f"weekdays covered: {len(weekdays)}/7 {weekdays}")

    sha = file_sha256(args.features)
    print(f"features: {args.features} sha256 {sha[:16]}...")
    frame = apply_exclusions(pd.read_parquet(args.features), load_exclusions())

    t0 = time.perf_counter()
    results = score_folds(frame, folds)
    seconds = time.perf_counter() - t0
    print(f"scored {len(results)} rows in {seconds:.0f}s ({seconds / len(folds):.1f}s/fold)")

    current = {
        "fold_ids": fold_ids,
        "n_rows": int(len(results)),
        "features_sha256": sha,
        "metrics": gate_metrics(results),
        "env": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "cpu_count": os.cpu_count(),
            "seconds": round(seconds, 1),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for name in REPORTED:
        print(f"  {name:<14} {current['metrics'][name]:>12.6f}")

    if args.reference is not None:
        ref = reference_metrics(args.reference, fold_ids)
        print(f"\nsame folds in {args.reference}:")
        for name in REPORTED:
            diff = current["metrics"][name] - ref[name]
            print(f"  {name:<14} {ref[name]:>12.6f}  (gate - reference = {diff:+.6f})")

    if args.baseline is None:
        return EXIT_PASS
    baseline = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
    code, lines = compare(baseline, current, max_rel_increase())
    print(f"\n=== gate vs {args.baseline} ===")
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
