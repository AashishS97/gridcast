"""Drift and data checks: is the model still operating in the world it was
trained on, and is the upstream data still behaving the way the gates assume?

Three kinds of drift, and which check answers which:
  feature drift   P(X) moves: the weather the model is fed (temperature,
                  irradiance, wind) is unusual for the season.
  target drift    P(y) moves: the load itself is unusual for the season (here
                  partly structural: rooftop solar keeps lowering daytime net
                  load year on year).
  concept drift   P(y | X) moves: same inputs, different outcome. Invisible in
                  the inputs; only the residuals show it. Here: are the served
                  quantiles' tail rates still 10% / 10%, and is the raw
                  model's median error still the one the conformal shift was
                  fitted to?

Why not a KS/PSI test against the training data: load and weather are
seasonal, so November against an all-year reference "drifts" every year, and
against the previous month it drifts every month. A fair reference compares
like with like: the same time of year, in the years the model was trained on.
Each hour is turned into an ANOMALY (value minus the reference-year average
for that hour of day, within +-15 days of that day of the year), and the
recent window's mean anomaly is set against the mean anomalies of all
reference windows from the same season. Reference anomalies are computed
leave-one-year-out: a year compared with an average that includes itself
looks less unusual than it is, and with three years that bias is large.

Feature and target drift are amber at most: unusual inputs are a reason to
look, not evidence that the forecast got worse (the accuracy zones are the
arbiter). The support check matters more for trees: LightGBM predicts a
constant beyond the range it was trained on, so hours outside the seasonal
training range are where it cannot extrapolate.

Data checks read only vintages.csv (the daily ledger of what was published):
  late_revision  a day the gates already treat as final (older than
                 provisional_days) changed by more than rel_tol between two
                 runs. Serving and scoring used a value that was not final.  RED
  final_level    final days look low: days just past the provisional line
                 against older final days. A revision that never came.       RED
  final_gaps     final days with fewer than 24 hours.                      AMBER
  weather_lag    the newest day with temperature is too old: weather gates
                 will start refusing origins.                              AMBER
  provisional    young days against final days: is the provisional
                 under-publication active right now (information, no zone).

Run:  uv run python -m gridcast.monitoring.drift --log-dir forecast-log
      (after gridcast.build has written the feature table)
      writes DRIFT.md, appends drift.csv. Exit 2 if any check is red.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.models.backtest import FEATURES_PATH
from gridcast.models.export_artifact import load_config as load_serving_config
from gridcast.monitoring.accuracy import (
    CONFIG_PATH,
    MonitoringConfig,
    band,
    coverage_zone,
    resample_index,
)
from gridcast.monitoring.accuracy import load_config as load_accuracy_config
from gridcast.serving.daily_log import HOURLY_COLS, VINTAGE_COLS, append, read_log, write_log
from gridcast.serving.forecaster import load_provisional_days

DRIFT_VARIABLES = ["load_mw", "temperature_2m", "shortwave_radiation", "wind_speed_10m"]
FAMILY = {"load_mw": "target"} | {v: "feature" for v in DRIFT_VARIABLES[1:]}
DRIFT_COLS = ["run_at", "family", "check", "subject", "value", "lo", "hi", "zone", "note"]


@dataclass(frozen=True)
class DriftConfig:
    window_days: int
    clim_halfwidth_days: int
    season_halfwidth_days: int
    min_valid_frac: float
    shift_central: float
    support_amber: float
    rel_tol: float
    level_ratio_red: float
    weather_lag_amber_days: int


def load_config(path: str | Path = CONFIG_PATH) -> DriftConfig:
    raw = tomllib.loads(Path(path).read_bytes().decode("utf-8-sig"))
    d, q = raw["drift"], raw["data"]
    return DriftConfig(
        window_days=int(d["window_days"]),
        clim_halfwidth_days=int(d["clim_halfwidth_days"]),
        season_halfwidth_days=int(d["season_halfwidth_days"]),
        min_valid_frac=float(d["min_valid_frac"]),
        shift_central=float(d["shift_central"]),
        support_amber=float(d["support_amber"]),
        rel_tol=float(q["rel_tol"]),
        level_ratio_red=float(q["level_ratio_red"]),
        weather_lag_amber_days=int(q["weather_lag_amber_days"]),
    )


def result(family, check, subject, value, lo=np.nan, hi=np.nan, zone="info", note="") -> dict:
    return {
        "family": family,
        "check": check,
        "subject": subject,
        "value": value,
        "lo": lo,
        "hi": hi,
        "zone": zone,
        "note": note,
    }


# --- seasonal anomalies -------------------------------------------------------

N_DOY = 365


def _doy_hour(ts: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Day of year 0-364 (29 February shares a bin with 28 February, so every
    bin holds one calendar day per year) and hour of day."""
    doy = (ts.dayofyear - 1).to_numpy()
    doy = np.where(ts.is_leap_year & (doy >= 59), doy - 1, doy)
    return doy, ts.hour.to_numpy()


def _circular_sum(grid: np.ndarray, k: int) -> np.ndarray:
    """Sum over a +-k day window along axis 0 (day of year), wrapping."""
    return sum(np.roll(grid, s, axis=0) for s in range(-k, k + 1))


def _circular_extreme(grid: np.ndarray, k: int, fn) -> np.ndarray:
    return fn(np.stack([np.roll(grid, s, axis=0) for s in range(-k, k + 1)]), axis=0)


def anomalies(
    x: pd.Series, reference_end: pd.Timestamp, k: int
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Hourly anomalies against the seasonal climatology of the reference
    period (timestamps < reference_end). Reference hours: leave-one-year-out;
    later hours: against all reference years. Also returns the seasonal
    min/max of the reference values per (day of year, hour)."""
    x = x.dropna()
    ts = pd.DatetimeIndex(x.index)
    doy, hour = _doy_hour(ts)
    in_ref = np.asarray(ts < reference_end)
    years = ts.year.to_numpy()
    ref_years = np.unique(years[in_ref])
    vals = x.to_numpy(float)

    sums = {y: np.zeros((N_DOY, 24)) for y in ref_years}
    counts = {y: np.zeros((N_DOY, 24)) for y in ref_years}
    cell_min = np.full((N_DOY, 24), np.inf)
    cell_max = np.full((N_DOY, 24), -np.inf)
    for y in ref_years:
        m = in_ref & (years == y)
        np.add.at(sums[y], (doy[m], hour[m]), vals[m])
        np.add.at(counts[y], (doy[m], hour[m]), 1)
    np.minimum.at(cell_min, (doy[in_ref], hour[in_ref]), vals[in_ref])
    np.maximum.at(cell_max, (doy[in_ref], hour[in_ref]), vals[in_ref])
    # one seasonal profile per year, then the years averaged with equal weight;
    # cells without reference data stay NaN ("mean of empty slice" is expected)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = [_circular_sum(sums[y], k) / _circular_sum(counts[y], k) for y in ref_years]
        stack = np.stack(profile) if profile else np.full((1, N_DOY, 24), np.nan)
        clim = np.full(len(x), np.nan)
        later = ~in_ref
        clim[later] = np.nanmean(stack, axis=0)[doy[later], hour[later]]
        for i, y in enumerate(ref_years):
            m = in_ref & (years == y)
            others = np.delete(stack, i, axis=0)
            if len(others):
                clim[m] = np.nanmean(others, axis=0)[doy[m], hour[m]]
        lo = _circular_extreme(np.where(np.isinf(cell_min), np.nan, cell_min), k, np.nanmin)
        hi = _circular_extreme(np.where(np.isinf(cell_max), np.nan, cell_max), k, np.nanmax)
    return pd.Series(vals - clim, index=ts), lo, hi


def _circular_doy_distance(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b) % 365
    return np.minimum(d, 365 - d)


def seasonal_drift(
    x: pd.Series,
    reference_end: pd.Timestamp,
    window_end: pd.Timestamp,
    cfg: DriftConfig,
    family: str,
    subject: str,
) -> list[dict]:
    """Shift check (window mean anomaly against same-season reference
    windows) and support check (share of window hours outside the seasonal
    training range) for one hourly series."""
    window_start = window_end - pd.Timedelta(days=cfg.window_days)
    anom, lo, hi = anomalies(x[x.index < window_end], reference_end, cfg.clim_halfwidth_days)
    hours_needed = cfg.window_days * 24 * cfg.min_valid_frac
    in_window = (anom.index >= window_start) & (anom.index < window_end)
    w = anom[in_window].dropna()
    label = f"{window_start:%Y-%m-%d} to {(window_end - pd.Timedelta('1h')):%Y-%m-%d}"
    if len(w) < hours_needed:
        return [result(family, "shift", subject, np.nan, zone="insufficient", note=label)]

    # null: rolling window means of reference daily anomalies, same season
    ref = anom[anom.index < reference_end]
    daily = ref.groupby(ref.index.floor("D")).agg(["mean", "count"])
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    ok = daily["count"] >= 24 * cfg.min_valid_frac
    min_days = int(cfg.window_days * cfg.min_valid_frac)
    roll = daily["mean"].where(ok).rolling(cfg.window_days, min_periods=min_days).mean()
    # index = last day of each window; drop windows that overlap the recent one
    roll = roll[roll.index < window_start - pd.Timedelta("1D")].dropna()
    centres = roll.index - pd.Timedelta(days=cfg.window_days / 2)
    target_doy = _doy_hour(pd.DatetimeIndex([window_end - pd.Timedelta(days=cfg.window_days / 2)]))
    same_season = _circular_doy_distance(_doy_hour(centres)[0], target_doy[0][0])
    null = roll[same_season <= cfg.season_halfwidth_days].to_numpy()
    value = float(w.mean())
    out = []
    if len(null) < cfg.window_days:  # less than ~one window's worth of daily steps
        out.append(
            result(
                family,
                "shift",
                subject,
                value,
                zone="insufficient",
                note=f"{label}; {len(null)} reference windows",
            )
        )
    else:
        c_lo, c_hi = band(null, cfg.shift_central)
        rank = float((null < value).mean())
        if c_lo <= value <= c_hi:
            zone = "green"
        else:
            beyond = value < null.min() or value > null.max()
            zone = "amber_unprecedented" if beyond else "amber"
        n_eff = len(null) / cfg.window_days
        out.append(
            result(
                family,
                "shift",
                subject,
                value,
                c_lo,
                c_hi,
                zone,
                f"{label}; percentile {rank:.0%} of {len(null)} same-season "
                f"reference windows (~{n_eff:.0f} independent)",
            )
        )

    doy, hour = _doy_hour(pd.DatetimeIndex(w.index))
    raw = x.reindex(w.index).to_numpy(float)
    outside = (raw < lo[doy, hour]) | (raw > hi[doy, hour])
    share = float(np.mean(outside))
    out.append(
        result(
            family,
            "support",
            subject,
            share,
            zone="green" if share <= cfg.support_amber else "amber",
            note=f"{label}; {int(outside.sum())} of {len(w)} hours outside "
            "the seasonal training range",
        )
    )
    return out


# --- concept drift: residuals --------------------------------------------------


def residual_drift(hourly: pd.DataFrame, shifts: dict, acc: MonitoringConfig) -> list[dict]:
    """Tail rates of the served p10 / p90 over the accuracy window, and the
    raw p50's median error against the shift fitted on the calibration pool."""
    if hourly.empty:
        return [result("concept", "tails", "p10/p90", np.nan, zone="no_data")]
    h = hourly.sort_values(["origin", "horizon"])
    origins = h["origin"].drop_duplicates().sort_values().tail(acc.window_origins)
    h = h[h["origin"].isin(origins)]
    n = len(origins)
    y = h["y"].to_numpy(float).reshape(n, 24)
    below = (y < h["p10"].to_numpy(float).reshape(n, 24)).mean(axis=1)
    above = (y > h["p90"].to_numpy(float).reshape(n, 24)).mean(axis=1)
    idx = resample_index(n, acc.n_boot, acc.seed)
    insufficient = n < acc.min_origins
    out = []
    for name, rate, nominal in (("below p10", below, 0.10), ("above p90", above, 0.10)):
        boot = rate[idx].mean(axis=1)
        a, r = band(boot, acc.amber_level), band(boot, acc.red_level)
        zone = "insufficient" if insufficient else coverage_zone(rate.mean(), a, r, nominal)
        zone = zone.replace("under", "low").replace("over", "high")
        out.append(
            result(
                "concept",
                "tail_rate",
                name,
                float(rate.mean()),
                *a,
                zone,
                f"nominal 10%; {n} origins",
            )
        )

    resid = y - h["p50_raw"].to_numpy(float).reshape(n, 24)
    if np.isnan(resid).any():
        out.append(result("concept", "median_error", "raw p50", np.nan, zone="no_data"))
        return out
    boot_med = np.median(resid[idx].reshape(acc.n_boot, -1), axis=1)
    a, r = band(boot_med, acc.amber_level), band(boot_med, acc.red_level)
    target = float(shifts["q50"])
    if insufficient:
        zone = "insufficient"
    elif a[0] <= target <= a[1]:
        zone = "green"
    elif r[0] <= target <= r[1]:
        zone = "amber"
    else:
        zone = "red"
    out.append(
        result(
            "concept",
            "median_error",
            "raw p50",
            float(np.median(resid)),
            *a,
            zone,
            f"y - raw p50; the calibration pool's median (the p50 shift) is {target:+.1f} MW",
        )
    )
    return out


# --- data checks on the vintage ledger ----------------------------------------


def _ages(v: pd.DataFrame) -> pd.Series:
    return (v["data_as_of"].dt.floor("D") - v["day"]).dt.days


def data_checks(vintages: pd.DataFrame, provisional_days: int, cfg: DriftConfig) -> list[dict]:
    if vintages.empty:
        return [result("data", "ledger", "vintages.csv", np.nan, zone="no_data")]
    v = vintages.copy()
    v["age"] = _ages(v)
    runs = sorted(v["run_at"].unique())
    latest = v[v["run_at"] == runs[-1]]
    out = []

    # revisions between consecutive runs
    if len(runs) < 2:
        out.append(
            result(
                "data",
                "late_revision",
                "load",
                np.nan,
                zone="insufficient",
                note="one vintage so far: revisions need two",
            )
        )
    else:
        events = []
        for prev, nxt in zip(runs[:-1], runs[1:], strict=True):
            a = v[(v["run_at"] == prev) & (v["load_hours"] == 24)].set_index("day")
            b = v[(v["run_at"] == nxt) & (v["load_hours"] == 24)].set_index("day")
            common = a.index.intersection(b.index)
            change = b.loc[common, "load_mean_mw"] / a.loc[common, "load_mean_mw"] - 1
            moved = change[change.abs() > cfg.rel_tol]
            for day, c in moved.items():
                events.append(
                    {"run_at": nxt, "day": day, "age": int(a.loc[day, "age"]), "change": float(c)}
                )
        ev = pd.DataFrame(events, columns=["run_at", "day", "age", "change"])
        late_now = ev[(ev["run_at"] == runs[-1]) & (ev["age"] > provisional_days)]
        if len(ev):
            out.append(
                result(
                    "data",
                    "revisions_seen",
                    "load",
                    float(len(ev)),
                    float(ev["age"].min()),
                    float(ev["age"].max()),
                    "info",
                    f"{len(ev)} day(s) revised by > {cfg.rel_tol:.0%} in the ledger; "
                    f"ages at revision {int(ev['age'].min())}-{int(ev['age'].max())} "
                    f"days, median change {ev['change'].median():+.1%}",
                )
            )
        days = ", ".join(
            f"{d:%Y-%m-%d} (age {a}, {c:+.1%})"
            for d, a, c in late_now[["day", "age", "change"]].itertuples(index=False)
        )
        out.append(
            result(
                "data",
                "late_revision",
                "load",
                float(len(late_now)),
                zone="red" if len(late_now) else "green",
                note=days or f"no day older than {provisional_days} days changed in the latest run",
            )
        )

    # level of the first final days against older final days (latest vintage)
    full = latest[latest["load_hours"] == 24]
    p = provisional_days
    young_final = full[(full["age"] > p) & (full["age"] <= p + 14)]["load_mean_mw"]
    old_final = full[(full["age"] > p + 14)]["load_mean_mw"]
    provisional = full[(full["age"] >= 1) & (full["age"] <= p)]["load_mean_mw"]
    if len(young_final) >= 7 and len(old_final) >= 7:
        ratio = float(young_final.mean() / old_final.mean())
        out.append(
            result(
                "data",
                "final_level",
                "load",
                ratio,
                zone="red" if ratio < cfg.level_ratio_red else "green",
                note=f"mean of final days aged {p + 1}-{p + 14} / aged > {p + 14}; "
                f"red below {cfg.level_ratio_red}",
            )
        )
        if len(provisional) >= 7:
            out.append(
                result(
                    "data",
                    "provisional",
                    "load",
                    float(provisional.mean() / young_final.mean()),
                    zone="info",
                    note=f"days aged 1-{p} / final days aged {p + 1}-{p + 14}: "
                    "well below 1 means under-publication is active",
                )
            )
    else:
        out.append(
            result(
                "data",
                "final_level",
                "load",
                np.nan,
                zone="insufficient",
                note="the ledger is too short for this check",
            )
        )

    gaps = latest[(latest["age"] > p) & (latest["load_hours"] < 24)]
    out.append(
        result(
            "data",
            "final_gaps",
            "load",
            float(len(gaps)),
            zone="amber" if len(gaps) else "green",
            note=", ".join(
                f"{d:%Y-%m-%d} ({h}h)"
                for d, h in gaps[["day", "load_hours"]].itertuples(index=False)
            )
            or "every final day has 24 hours",
        )
    )

    with_temp = latest[latest["temp_mean_c"].notna()]
    if with_temp.empty:
        out.append(
            result(
                "data",
                "weather_lag",
                "temperature",
                np.nan,
                zone="amber",
                note="no temperature in the ledger window",
            )
        )
    else:
        lag = int(with_temp["age"].min())
        out.append(
            result(
                "data",
                "weather_lag",
                "temperature",
                float(lag),
                zone="amber" if lag > cfg.weather_lag_amber_days else "green",
                note=f"newest day with temperature is {lag} days old",
            )
        )
    return out


# --- report -----------------------------------------------------------------


def _fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:,.3g}" if abs(x) < 10 else f"{x:,.1f}"


def report_lines(rows: list[dict], now: pd.Timestamp, reference_end: pd.Timestamp) -> list[str]:
    lines = [
        "# GridCast drift and data checks",
        "",
        f"Written by `gridcast.monitoring.drift` at {now:%Y-%m-%d %H:%M} UTC. Thresholds:",
        "`configs/monitoring.toml`. Feature and target drift compare the recent window with",
        f"the same season in the training years (before {reference_end:%Y-%m-%d}), as",
        "anomalies from the seasonal average; they are amber at most, because unusual",
        "inputs are a reason to look, not evidence of worse forecasts.",
        "",
    ]
    titles = {
        "feature": "Feature drift (weather inputs)",
        "target": "Target drift (load)",
        "concept": "Concept drift (residuals of the served model)",
        "data": "Data checks (vintage ledger)",
    }
    for family, title in titles.items():
        sub = [r for r in rows if r["family"] == family]
        if not sub:
            continue
        lines += [
            f"## {title}",
            "",
            "| check | subject | value | band | zone | note |",
            "|---|---|---|---|---|---|",
        ]
        for r in sub:
            b = f"[{_fmt(r['lo'])}, {_fmt(r['hi'])}]" if not pd.isna(r["lo"]) else ""
            lines.append(
                f"| {r['check']} | {r['subject']} | {_fmt(r['value'])} | {b} "
                f"| {r['zone']} | {r['note']} |"
            )
        lines.append("")
    return lines


def run(
    log_dir: Path,
    features: pd.DataFrame | None,
    manifest: dict,
    provisional_days: int,
    cfg: DriftConfig,
    acc: MonitoringConfig,
    now: pd.Timestamp,
) -> int:
    reference_end = pd.Timestamp(manifest["train_cutoff"]).tz_convert("UTC")
    rows: list[dict] = []
    if features is not None:
        f = features.set_index("timestamp").sort_index()
        load = f["load_mw"].dropna()
        data_as_of = load.index.max()
        load_final_end = (data_as_of - pd.Timedelta(days=provisional_days)).floor("D")
        for var in DRIFT_VARIABLES:
            series = f[var]
            end = load_final_end if var == "load_mw" else series.dropna().index.max().floor("D")
            rows += seasonal_drift(series, reference_end, end, cfg, FAMILY[var], var)
    hourly = read_log(log_dir / "hourly.csv", HOURLY_COLS)
    rows += residual_drift(hourly, manifest["calibration"]["shifts_mw"], acc)
    rows += data_checks(read_log(log_dir / "vintages.csv", VINTAGE_COLS), provisional_days, cfg)

    lines = report_lines(rows, now, reference_end)
    (log_dir / "DRIFT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    path = log_dir / "drift.csv"
    old = read_log(path, DRIFT_COLS) if path.exists() else pd.DataFrame(columns=DRIFT_COLS)
    new = pd.DataFrame([{"run_at": now, **r} for r in rows], columns=DRIFT_COLS)
    write_log(append(old, new), path)
    print("\n".join(lines))
    red = [f"{r['check']} {r['subject']}" for r in rows if str(r["zone"]).startswith("red")]
    if red:
        print(f"\nRED: {', '.join(red)}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", type=Path, required=True)
    args = ap.parse_args()
    serving = load_serving_config()
    manifest = json.loads(
        (serving.output_root / serving.name / "manifest.json").read_text(encoding="utf-8")
    )
    features = pd.read_parquet(FEATURES_PATH)
    return run(
        args.log_dir,
        features,
        manifest,
        load_provisional_days(),
        load_config(),
        load_accuracy_config(),
        pd.Timestamp.now(tz="UTC").floor("s"),
    )


if __name__ == "__main__":
    sys.exit(main())
