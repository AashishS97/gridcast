"""Daily shadow log: forecast every newly servable origin, score every logged
forecast whose actuals have become final, append both to CSV files.

Why retrospective: ENTSO-E publishes NL actual load ~20-35% low for ~28 days
and revises it later, and every model input sits in the 336h before the
origin. A forecast for tomorrow would be built on provisional numbers, so the
serving gates refuse it. What CAN be done daily, honestly, is:
  1. forecast the newest origins that pass every gate (~30 days behind), with
     the same Forecaster the API uses;
  2. score each logged forecast once its own 24 target hours are final --
     the frozen model evaluated on data it never saw, one day at a time.

The scores are of the SERVED forecast: conformally calibrated p10/p50/p90.

Files in --log-dir (created on the first run, appended afterwards):
  forecasts.csv  24 rows per origin: p10/p50/p90 (MW), trust flag, provenance
  inputs.csv     24 rows per origin: the exact feature values the boosters saw.
                 The feature table is rebuilt from a fresh upstream pull every
                 run, so without this a past forecast cannot be reproduced once
                 the upstream revises; it is also the reference for feature drift
  scores.csv     one row per scored origin: MAE, pinball, 80% coverage, width
  hourly.csv     24 rows per scored origin: the final actual, the served and the
                 raw (pre-calibration) quantiles, and the seasonal-naive value
                 (load one week earlier). The per-hour detail behind scores.csv,
                 read by gridcast.monitoring.accuracy
  vintages.csv   per run, the last VINTAGE_DAYS calendar days as published at
                 that run (hours present, mean load, mean temperature): the only
                 record of when a revision arrived, which the API data cannot
                 reconstruct after the fact
  runs.csv       one row per run: data vintage, gate state, what changed, code
  README.md      the current state, rewritten each run

Run:  uv run python -m gridcast.serving.daily_log --log-dir forecast-log
Exit: 0 ok; 1 the data is stale or nothing is servable (the run is still
logged, so the failure is visible in runs.csv as well as in CI).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.models.metrics import interval_coverage, mae, mean_interval_width, pinball_loss
from gridcast.serving.forecaster import Forecaster, from_config

FORECAST_COLS = [
    "origin", "timestamp", "horizon", "p10", "p50", "p90",
    "trust_low", "trust_reasons", "artifact", "data_as_of", "logged_at",
    "p10_raw", "p50_raw", "p90_raw",
]  # fmt: skip
HOURLY_COLS = [
    "origin", "timestamp", "horizon", "y", "p10", "p50", "p90",
    "p10_raw", "p50_raw", "p90_raw", "naive_168", "artifact", "scored_at",
]  # fmt: skip
SCORE_COLS = [
    "origin", "n_hours", "mae_p50", "pinball_p10", "pinball_p50", "pinball_p90",
    "pinball_mean", "coverage_80", "width_mean", "trust_low_hours", "data_as_of", "scored_at",
]  # fmt: skip
RUN_COLS = [
    "run_at", "artifact", "data_as_of", "provisional_from", "n_servable",
    "last_servable", "new_forecasts", "new_scores", "status", "code_sha",
]  # fmt: skip
INPUT_KEYS = ["origin", "timestamp", "horizon", "artifact", "logged_at"]
VINTAGE_COLS = ["run_at", "data_as_of", "day", "load_hours", "load_mean_mw", "temp_mean_c"]
# columns added after a log file first existed: older files get them as empty
ADDED_COLS = {"code_sha", "p10_raw", "p50_raw", "p90_raw"}
TIME_COLS = {"origin", "timestamp", "data_as_of", "logged_at", "scored_at", "run_at",
             "provisional_from", "last_servable", "day", "window_first",
             "window_last"}  # fmt: skip
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
ROUNDING = {
    "p10": 1, "p50": 1, "p90": 1, "mae_p50": 2, "pinball_p10": 2, "pinball_p50": 2,
    "pinball_p90": 2, "pinball_mean": 2, "coverage_80": 4, "width_mean": 1,
    "load_mean_mw": 1, "temp_mean_c": 2, "p10_raw": 1, "p50_raw": 1, "p90_raw": 1,
    "y": 1, "naive_168": 1,
}  # fmt: skip
MAX_DATA_AGE = pd.Timedelta(days=3)
# 28 provisional days + the revision landing + margin: every day is seen both
# before and after its expected revision
VINTAGE_DAYS = 60


def read_log(path: Path, columns: list[str], extra_ok: bool = False) -> pd.DataFrame:
    """Read one log file. `columns` must be present (except ADDED_COLS, which
    older files get as empty); with extra_ok, other columns are kept too (the
    inputs log, whose feature columns depend on the artifact)."""
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in ADDED_COLS & (set(columns) - set(df.columns)):
        df[col] = np.nan
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    for col in TIME_COLS & set(columns):
        df[col] = pd.to_datetime(df[col], utc=True)
    if extra_ok:
        return df[columns + [c for c in df.columns if c not in columns]]
    return df[columns]


def write_log(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in TIME_COLS & set(out.columns):
        out[col] = pd.to_datetime(out[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = out.round({k: v for k, v in ROUNDING.items() if k in out.columns})
    out.to_csv(path, index=False)


def append(log: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Concatenate, keeping numeric dtypes when the log is still empty."""
    if new.empty:
        return log
    if log.empty:
        return new.reset_index(drop=True)
    return pd.concat([log, new], ignore_index=True)


def new_forecasts(
    fc: Forecaster, logged: pd.DataFrame, now: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast every servable origin not already in the log; return the
    forecasts and the feature rows they were computed from."""
    done = set(pd.to_datetime(logged["origin"], utc=True)) if len(logged) else set()
    rows, inputs = [], []
    for origin in fc.servable_origins():
        if origin in done:
            continue
        f, x = fc.forecast_with_inputs(origin)
        x.insert(3, "artifact", fc.artifact.name)
        x.insert(4, "logged_at", now)
        inputs.append(x)
        rows.append(
            pd.DataFrame(
                {
                    "origin": f["origin"],
                    "timestamp": f["timestamp"],
                    "horizon": f["horizon"],
                    "p10": f["q10_cal"],
                    "p50": f["q50_cal"],
                    "p90": f["q90_cal"],
                    "p10_raw": f["q10"],
                    "p50_raw": f["q50"],
                    "p90_raw": f["q90"],
                    "trust_low": f["trust_low"].astype(bool),
                    "trust_reasons": f["trust_reasons"],
                    "artifact": fc.artifact.name,
                    "data_as_of": fc.data_as_of,
                    "logged_at": now,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=FORECAST_COLS), pd.DataFrame(columns=INPUT_KEYS)
    return pd.concat(rows, ignore_index=True), pd.concat(inputs, ignore_index=True)


def score_origin(group: pd.DataFrame, actuals: np.ndarray) -> dict:
    g = group.sort_values("horizon")
    y = np.asarray(actuals, dtype=float)
    q = {name: g[name].to_numpy(dtype=float) for name in QUANTILES}
    pin = {name: pinball_loss(y, q[name], tau) for name, tau in QUANTILES.items()}
    return {
        "n_hours": len(y),
        "mae_p50": mae(y, q["p50"]),
        **{f"pinball_{name}": value for name, value in pin.items()},
        "pinball_mean": float(np.mean(list(pin.values()))),
        "coverage_80": interval_coverage(y, q["p10"], q["p90"]),
        "width_mean": mean_interval_width(q["p10"], q["p90"]),
        "trust_low_hours": int(g["trust_low"].astype(bool).sum()),
    }


def new_scores(
    fc: Forecaster, forecasts: pd.DataFrame, scored: pd.DataFrame, now: pd.Timestamp
) -> pd.DataFrame:
    """Score each logged origin, once, as soon as its 24 actuals are final."""
    done = set(pd.to_datetime(scored["origin"], utc=True)) if len(scored) else set()
    rows = []
    for origin, group in forecasts.groupby("origin", sort=True):
        origin = pd.Timestamp(origin)
        if origin in done or len(group) != 24:
            continue
        actuals = fc.final_actuals(origin)
        if actuals is None:
            continue
        rows.append(
            {
                "origin": origin,
                **score_origin(group, actuals),
                "data_as_of": fc.data_as_of,
                "scored_at": now,
            }  # fmt: skip
        )
    return pd.DataFrame(rows, columns=SCORE_COLS)


def raw_quantiles(group: pd.DataFrame, fc: Forecaster) -> dict[str, np.ndarray]:
    """Pre-calibration quantiles for one logged origin. Logged since Phase 6;
    for older rows of the CURRENT artifact they are recovered exactly as
    served minus shift (the shifts increase with tau, so the rearrangement
    after shifting is a no-op). Rows of another artifact: unknown, NaN."""
    out = {}
    for name in QUANTILES:
        logged = group[f"{name}_raw"].to_numpy(dtype=float)
        if not np.isnan(logged).any():
            out[name] = logged
        elif (group["artifact"] == fc.artifact.name).all():
            shift = fc.artifact.shifts[name.replace("p", "q")]
            out[name] = group[name].to_numpy(dtype=float) - shift
        else:
            out[name] = np.full(len(group), np.nan)
    return out


def new_hourly(
    fc: Forecaster,
    forecasts: pd.DataFrame,
    scores: pd.DataFrame,
    hourly: pd.DataFrame,
    now: pd.Timestamp,
) -> pd.DataFrame:
    """Per-hour rows for every scored origin not yet in the hourly log. Uses
    the same final actuals as scoring; origins scored before this file existed
    are filled in on the first run (their target days are final already)."""
    done = set(pd.to_datetime(hourly["origin"], utc=True)) if len(hourly) else set()
    rows = []
    for origin in sorted(set(pd.to_datetime(scores["origin"], utc=True)) - done):
        g = forecasts[forecasts["origin"] == origin].sort_values("horizon")
        actuals = fc.final_actuals(origin)
        if len(g) != 24 or actuals is None:
            continue
        ts = pd.DatetimeIndex(g["timestamp"])
        raw = raw_quantiles(g, fc)
        rows.append(
            pd.DataFrame(
                {
                    "origin": origin,
                    "timestamp": ts,
                    "horizon": g["horizon"].to_numpy(int),
                    "y": actuals,
                    **{n: g[n].to_numpy(dtype=float) for n in QUANTILES},
                    **{f"{n}_raw": raw[n] for n in QUANTILES},
                    "naive_168": fc.actual_load(ts - pd.Timedelta(hours=168)),
                    "artifact": g["artifact"].to_numpy(),
                    "scored_at": now,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=HOURLY_COLS)


def summary_lines(fc: Forecaster, scores: pd.DataFrame, run: dict) -> list[str]:
    lines = [
        "# GridCast daily shadow log",
        "",
        "Written by `.github/workflows/daily.yml` on `main`. Forecasts are retrospective",
        "(~30 days behind): ENTSO-E NL actual load is provisional for ~28 days, so the",
        "serving gates refuse anything newer. Each forecast is scored once its own",
        "target day is final. Scores are of the served, conformally calibrated forecast.",
        "",
        "## Latest run",
        "",
        f"- run at: {run['run_at']:%Y-%m-%d %H:%M} UTC, status `{run['status']}`",
        f"- artifact: `{run['artifact']}`",
        f"- data vintage: {run['data_as_of']:%Y-%m-%d %H:%M} UTC "
        f"(provisional from {run['provisional_from']:%Y-%m-%d %H:%M})",
        f"- servable origins: {run['n_servable']}, newest "
        + (f"{run['last_servable']:%Y-%m-%d}" if pd.notna(run["last_servable"]) else "none"),
        f"- this run: {run['new_forecasts']} new forecast(s), {run['new_scores']} new score(s)",
        "",
        "## Shadow evaluation so far",
        "",
    ]
    if scores.empty:
        lines.append("No origin has final actuals yet.")
        return lines
    pooled = {
        "origins scored": f"{len(scores)} ({scores['origin'].min():%Y-%m-%d} to "
        f"{scores['origin'].max():%Y-%m-%d})",
        "MAE p50 (MW)": f"{scores['mae_p50'].mean():,.1f}",
        "mean pinball (MW)": f"{scores['pinball_mean'].mean():,.1f}",
        "80% interval coverage": f"{scores['coverage_80'].mean():.1%} (target 80%)",
        "mean interval width (MW)": f"{scores['width_mean'].mean():,.0f}",
    }
    lines += ["| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in pooled.items()]
    lines += [
        "",
        "These are the served (calibrated) quantiles. For comparisons with the backtest",
        "and calibration-pool baselines (which are raw, pre-calibration), rolling windows",
        "with uncertainty bands, and the per-hour breakdown, see ACCURACY.md.",
    ]
    lines += [
        "",
        "Every origin weighs equally (24 hours each). A few dozen origins is a small",
        "sample: read coverage with a margin of several points.",
    ]
    return lines


def run(fc: Forecaster, log_dir: Path, now: pd.Timestamp) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    names = ("forecasts", "inputs", "scores", "hourly", "vintages", "runs")
    paths = {name: log_dir / f"{name}.csv" for name in names}

    forecasts = read_log(paths["forecasts"], FORECAST_COLS)
    fresh, fresh_inputs = new_forecasts(fc, forecasts, now)
    forecasts = append(forecasts, fresh)
    inputs = append(read_log(paths["inputs"], INPUT_KEYS, extra_ok=True), fresh_inputs)

    scores = read_log(paths["scores"], SCORE_COLS)
    scored_now = new_scores(fc, forecasts, scores, now)
    scores = append(scores, scored_now)
    hourly = read_log(paths["hourly"], HOURLY_COLS)
    hourly = append(hourly, new_hourly(fc, forecasts, scores, hourly, now))

    servable = fc.servable_origins()
    problems = []
    if now - fc.data_as_of > MAX_DATA_AGE:
        problems.append(f"stale_data (newest actual load {fc.data_as_of})")
    if len(servable) == 0:
        problems.append("nothing_servable")
    status = "ok" if not problems else ";".join(problems)

    run_row = {
        "run_at": now,
        "artifact": fc.artifact.name,
        "data_as_of": fc.data_as_of,
        "provisional_from": fc.provisional_from,
        "n_servable": len(servable),
        "last_servable": servable[-1] if len(servable) else pd.NaT,
        "new_forecasts": int(fresh["origin"].nunique()) if len(fresh) else 0,
        "new_scores": len(scored_now),
        "status": status,
        "code_sha": (os.environ.get("GITHUB_SHA") or "local")[:12],
    }
    runs = read_log(paths["runs"], RUN_COLS)
    runs = append(runs, pd.DataFrame([run_row], columns=RUN_COLS))

    ledger = fc.daily_ledger(VINTAGE_DAYS)
    ledger.insert(0, "run_at", now)
    ledger.insert(1, "data_as_of", fc.data_as_of)
    vintages = append(read_log(paths["vintages"], VINTAGE_COLS), ledger[VINTAGE_COLS])

    write_log(forecasts.sort_values(["origin", "horizon"]), paths["forecasts"])
    write_log(inputs.sort_values(["origin", "horizon"]), paths["inputs"])
    write_log(vintages, paths["vintages"])
    write_log(scores.sort_values("origin"), paths["scores"])
    write_log(hourly.sort_values(["origin", "horizon"]), paths["hourly"])
    write_log(runs, paths["runs"])
    (log_dir / "README.md").write_text(
        "\n".join(summary_lines(fc, scores, run_row)) + "\n", encoding="utf-8"
    )

    print("\n".join(summary_lines(fc, scores, run_row)))
    if len(scored_now):
        print("\nnewly scored:")
        print(
            scored_now[
                ["origin", "mae_p50", "pinball_mean", "coverage_80", "width_mean"]
            ].to_string(index=False, float_format=lambda v: f"{v:,.2f}")
        )
    return 0 if status == "ok" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", type=Path, required=True, help="directory holding the CSV log")
    ap.add_argument("--artifact-dir", type=Path, help="another artifact (a challenger's trial)")
    args = ap.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    fc = from_config(args.artifact_dir)
    return run(fc, args.log_dir, pd.Timestamp.now(tz="UTC").floor("s"))


if __name__ == "__main__":
    sys.exit(main())
