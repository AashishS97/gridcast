"""Accuracy tracking over the shadow log: rolling metrics with honest
uncertainty, traffic-light zones, a per-hour breakdown, and the baselines.

Input: hourly.csv from the daily shadow log (gridcast.serving.daily_log) --
one row per scored target hour with the final actual, the served (calibrated)
and raw quantiles, and the seasonal-naive value (load one week earlier).

Why the bands resample DAYS, not hours: the 24 hours of one forecast share a
weather situation and a model state, so they miss together (Oct 9, 13, 14
2026 each missed 11-14 of 24 hours). A binomial band over hours treats them
as 24 independent trials and comes out about half as wide as the data
support. Resampling whole origins keeps each day's hours together.

Zones (configs/monitoring.toml), after the Basel traffic-light test:
  coverage  green: nominal 80% inside the amber-level band; amber: outside
            it but inside the red-level band; red: outside both. Reported
            with a direction: under (intervals too narrow) or over (too wide).
  skill     1 - MAE(p50) / MAE(seasonal naive) on the same hours. green:
            band above 0 (the model beats "same hour last week"); red: the
            red-level band entirely below 0; amber: anything in between.
  windows smaller than min_origins are reported but not zoned.

Baselines are compared like for like: the calibration pool and the backtest
gate measure RAW quantiles, so they are set against the shadow's raw
quantiles (served minus the conformal shift), not against the served ones.

Run:  uv run python -m gridcast.monitoring.accuracy --log-dir forecast-log
      writes ACCURACY.md, rolling.csv (the trailing window ending at every
      origin, for the dashboard) and by_horizon.csv; appends zones.csv
Exit: 0 green/amber/insufficient; 2 if any zone is red (an escalation, not a
      crash: the files are written first).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.models.export_artifact import PROJECT_ROOT
from gridcast.models.export_artifact import load_config as load_serving_config
from gridcast.serving.daily_log import HOURLY_COLS, append, read_log, write_log

CONFIG_PATH = PROJECT_ROOT / "configs" / "monitoring.toml"
GATE_BASELINE_PATH = PROJECT_ROOT / "configs" / "gate_baseline.json"
ZONE_COLS = [
    "run_at", "window_first", "window_last", "n_origins",
    "coverage", "coverage_lo", "coverage_hi", "coverage_zone",
    "mae_p50", "mae_lo", "mae_hi",
    "skill", "skill_lo", "skill_hi", "skill_zone",
]  # fmt: skip


@dataclass(frozen=True)
class MonitoringConfig:
    window_origins: int
    min_origins: int
    n_boot: int
    seed: int
    nominal_coverage: float
    amber_level: float
    red_level: float


def load_config(path: str | Path = CONFIG_PATH) -> MonitoringConfig:
    raw = tomllib.loads(Path(path).read_bytes().decode("utf-8-sig"))["accuracy"]
    cfg = MonitoringConfig(
        window_origins=int(raw["window_origins"]),
        min_origins=int(raw["min_origins"]),
        n_boot=int(raw["n_boot"]),
        seed=int(raw["seed"]),
        nominal_coverage=float(raw["nominal_coverage"]),
        amber_level=float(raw["amber_level"]),
        red_level=float(raw["red_level"]),
    )
    if not 0 < cfg.amber_level < cfg.red_level < 1:
        raise ValueError("need 0 < amber_level < red_level < 1")
    if not 1 <= cfg.min_origins <= cfg.window_origins:
        raise ValueError("need 1 <= min_origins <= window_origins")
    return cfg


# --- per-day table ----------------------------------------------------------


def daily_table(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per origin. Every origin has 24 hours, so the mean of daily
    values equals the pooled hourly value: days are the resampling unit."""
    h = hourly.copy()
    h["abs_err"] = (h["y"] - h["p50"]).abs()
    h["hit"] = (h["y"] >= h["p10"]) & (h["y"] <= h["p90"])
    h["width"] = h["p90"] - h["p10"]
    h["naive_abs_err"] = (h["y"] - h["naive_168"]).abs()
    h["raw_abs_err"] = (h["y"] - h["p50_raw"]).abs()
    raw_known = h[["p10_raw", "p90_raw"]].notna().all(axis=1)
    h["raw_hit"] = ((h["y"] >= h["p10_raw"]) & (h["y"] <= h["p90_raw"])).where(raw_known)
    g = h.groupby("origin", sort=True)
    out = pd.DataFrame(
        {
            "n_hours": g.size(),
            "mae_p50": g["abs_err"].mean(),
            "coverage": g["hit"].mean(),
            "width": g["width"].mean(),
            "naive_mae": g["naive_abs_err"].mean(),
            "raw_mae_p50": g["raw_abs_err"].mean(),
            "raw_coverage": g["raw_hit"].apply(lambda s: s.astype(float).mean()),
        }
    ).reset_index()
    if (out["n_hours"] != 24).any():
        raise ValueError("every origin in hourly.csv must have 24 hours")
    return out


# --- bootstrap --------------------------------------------------------------


def resample_index(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) indices: each row one bootstrap sample of the n days."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def band(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Central percentile interval of the bootstrap distribution."""
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(samples, [tail, 100 - tail])
    return float(lo), float(hi)


def coverage_zone(
    point: float, amber: tuple[float, float], red: tuple[float, float], nominal: float
) -> str:
    direction = "under" if point < nominal else "over"
    if amber[0] <= nominal <= amber[1]:
        return "green"
    if red[0] <= nominal <= red[1]:
        return f"amber_{direction}"
    return f"red_{direction}"


def skill_zone(amber: tuple[float, float], red: tuple[float, float]) -> str:
    if amber[0] > 0:
        return "green"
    if red[1] < 0:
        return "red"
    return "amber"


def window_summary(daily: pd.DataFrame, cfg: MonitoringConfig) -> dict:
    """Metrics, bands and zones over the trailing window of origins."""
    w = daily.sort_values("origin").tail(cfg.window_origins).reset_index(drop=True)
    n = len(w)
    out: dict = {"n_origins": n}
    if n == 0:
        out["status"] = "no_data"
        return out
    out["window_first"], out["window_last"] = w["origin"].iloc[0], w["origin"].iloc[-1]

    idx = resample_index(n, cfg.n_boot, cfg.seed)
    cov, mae, naive = (w[c].to_numpy(float) for c in ("coverage", "mae_p50", "naive_mae"))
    boot_cov = cov[idx].mean(axis=1)
    boot_mae = mae[idx].mean(axis=1)
    boot_skill = 1 - mae[idx].sum(axis=1) / naive[idx].sum(axis=1)  # paired: same days

    out["coverage"] = float(cov.mean())
    out["coverage_band"] = band(boot_cov, cfg.amber_level)
    out["coverage_band_red"] = band(boot_cov, cfg.red_level)
    out["mae_p50"] = float(mae.mean())
    out["mae_band"] = band(boot_mae, cfg.amber_level)
    out["naive_mae"] = float(naive.mean())
    out["skill"] = float(1 - mae.sum() / naive.sum())
    out["skill_band"] = band(boot_skill, cfg.amber_level)
    out["skill_band_red"] = band(boot_skill, cfg.red_level)
    out["width"] = float(w["width"].mean())

    raw = w.dropna(subset=["raw_mae_p50", "raw_coverage"])
    out["n_raw"] = len(raw)
    if len(raw):
        ridx = resample_index(len(raw), cfg.n_boot, cfg.seed)
        rmae, rcov = (raw[c].to_numpy(float) for c in ("raw_mae_p50", "raw_coverage"))
        out["raw_mae_p50"] = float(rmae.mean())
        out["raw_mae_band"] = band(rmae[ridx].mean(axis=1), cfg.amber_level)
        out["raw_coverage"] = float(rcov.mean())
        out["raw_coverage_band"] = band(rcov[ridx].mean(axis=1), cfg.amber_level)

    if n < cfg.min_origins:
        out["status"] = "insufficient"
        out["coverage_zone"] = out["skill_zone"] = "insufficient"
        return out
    out["status"] = "zoned"
    out["coverage_zone"] = coverage_zone(
        out["coverage"], out["coverage_band"], out["coverage_band_red"], cfg.nominal_coverage
    )
    out["skill_zone"] = skill_zone(out["skill_band"], out["skill_band_red"])
    return out


def rolling_table(daily: pd.DataFrame, cfg: MonitoringConfig) -> pd.DataFrame:
    """window_summary ending at every origin with at least min_origins
    before it: the accuracy-over-time series for the dashboard."""
    d = daily.sort_values("origin").reset_index(drop=True)
    rows = []
    for end in range(cfg.min_origins, len(d) + 1):
        s = window_summary(d.iloc[:end], cfg)
        rows.append(
            {
                "origin": s["window_last"],
                "n_origins": s["n_origins"],
                "coverage": s["coverage"],
                "coverage_lo": s["coverage_band"][0],
                "coverage_hi": s["coverage_band"][1],
                "coverage_zone": s["coverage_zone"],
                "mae_p50": s["mae_p50"],
                "mae_lo": s["mae_band"][0],
                "mae_hi": s["mae_band"][1],
                "skill": s["skill"],
                "skill_lo": s["skill_band"][0],
                "skill_hi": s["skill_band"][1],
                "skill_zone": s["skill_zone"],
            }
        )
    return pd.DataFrame(rows)


def by_horizon(hourly: pd.DataFrame, cfg: MonitoringConfig) -> pd.DataFrame:
    """MAE and 80% coverage per horizon over ALL scored origins, with
    day-resampled bands. Origins are midnight UTC, so horizon h is UTC hour
    h-1: horizon and hour of day cannot be separated in this design."""
    h = hourly.sort_values(["origin", "horizon"])
    err = (h["y"] - h["p50"]).abs().to_numpy(float).reshape(-1, 24)
    hit = ((h["y"] >= h["p10"]) & (h["y"] <= h["p90"])).to_numpy(float).reshape(-1, 24)
    idx = resample_index(len(err), cfg.n_boot, cfg.seed)
    boot_err, boot_hit = err[idx].mean(axis=1), hit[idx].mean(axis=1)  # (n_boot, 24)
    tail = (1 - cfg.amber_level) / 2 * 100
    pct = [tail, 100 - tail]
    mae_lo, mae_hi = np.percentile(boot_err, pct, axis=0)
    cov_lo, cov_hi = np.percentile(boot_hit, pct, axis=0)
    return pd.DataFrame(
        {
            "horizon": np.arange(1, 25),
            "utc_hour": np.arange(24),
            "n_origins": len(err),
            "mae_p50": err.mean(axis=0),
            "mae_lo": mae_lo,
            "mae_hi": mae_hi,
            "coverage": hit.mean(axis=0),
            "coverage_lo": cov_lo,
            "coverage_hi": cov_hi,
        }
    )


# --- baselines --------------------------------------------------------------


def baselines(manifest: dict, gate_baseline: dict) -> dict:
    """Reference numbers for RAW quantiles, with what each one is."""
    cal = manifest.get("calibration", {})
    gm = gate_baseline.get("metrics", {})
    return {
        "pool": {
            "label": f"calibration pool ({cal.get('n_origins', '?')} origins "
            f"{str(cal.get('first_origin', ''))[:10]} to {str(cal.get('last_origin', ''))[:10]}; "
            "out-of-sample, one retrain per origin, not a sealed holdout)",
            "raw_mae_p50": cal.get("pool_q50_mae_mw"),
            "raw_coverage": cal.get("pool_raw_interval_coverage"),
        },
        "backtest": {
            "label": f"backtest gate ({len(gate_baseline.get('fold_ids', []))} pinned dev folds, "
            "daily retrain, an earlier period)",
            "raw_mae_p50": gm.get("mae_q50"),
            "raw_coverage": gm.get("interval_cov"),
        },
    }


# --- report -----------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _band(b: tuple[float, float], fmt) -> str:
    return f"[{fmt(b[0])}, {fmt(b[1])}]"


def report_lines(
    s: dict, horizon: pd.DataFrame | None, base: dict, cfg: MonitoringConfig, now: pd.Timestamp
) -> list[str]:
    mw = "{:,.1f}".format
    lvl = f"{cfg.amber_level:.0%}"
    lines = [
        "# GridCast accuracy",
        "",
        f"Written by `gridcast.monitoring.accuracy` at {now:%Y-%m-%d %H:%M} UTC from",
        "hourly.csv. Bands resample whole days (origins), not hours: a forecast's 24",
        f"hours fail together. Thresholds: `configs/monitoring.toml`. Bands are {lvl}.",
        "",
    ]
    if s.get("status") == "no_data":
        return lines + ["No scored origins yet."]
    lines += [
        f"## Trailing window: {s['n_origins']} origins "
        f"({s['window_first']:%Y-%m-%d} to {s['window_last']:%Y-%m-%d})",
        "",
    ]
    if s["status"] == "insufficient":
        lines += [
            f"Fewer than {cfg.min_origins} origins: metrics shown, zones not assigned.",
            "",
        ]
    lines += [
        "| served forecast | value | band | zone |",
        "|---|---|---|---|",
        f"| 80% interval coverage | {_pct(s['coverage'])} | {_band(s['coverage_band'], _pct)} "
        f"| {s['coverage_zone']} |",
        f"| p50 MAE (MW) | {mw(s['mae_p50'])} | {_band(s['mae_band'], mw)} | |",
        f"| seasonal-naive MAE (MW) | {mw(s['naive_mae'])} | | |",
        f"| skill vs naive (1 - MAE ratio) | {s['skill']:.3f} "
        f"| {_band(s['skill_band'], '{:.3f}'.format)} | {s['skill_zone']} |",
        f"| mean interval width (MW) | {s['width']:,.0f} | | |",
        "",
        "## Raw quantiles against the baselines",
        "",
        "The pool and the backtest gate measure raw (pre-calibration) quantiles, so the",
        "shadow is compared on its raw quantiles too. A baseline inside the shadow band",
        "is consistent with it; outside, the live period differs from the reference.",
        "",
    ]
    if s.get("n_raw", 0):
        lines += [
            "| raw quantiles | shadow window | band | pool | backtest gate |",
            "|---|---|---|---|---|",
            f"| p50 MAE (MW) | {mw(s['raw_mae_p50'])} | {_band(s['raw_mae_band'], mw)} "
            f"| {mw(base['pool']['raw_mae_p50'])} | {mw(base['backtest']['raw_mae_p50'])} |",
            f"| 80% coverage | {_pct(s['raw_coverage'])} | {_band(s['raw_coverage_band'], _pct)} "
            f"| {_pct(base['pool']['raw_coverage'])} | {_pct(base['backtest']['raw_coverage'])} |",
            "",
            f"- pool: {base['pool']['label']}",
            f"- backtest gate: {base['backtest']['label']}",
            f"- shadow raw values available for {s['n_raw']} of {s['n_origins']} origins",
            "",
        ]
    else:
        lines += ["No raw quantiles available in this window.", ""]
    if horizon is not None:
        lines += [
            f"## Per hour, all {int(horizon['n_origins'].iloc[0])} scored origins",
            "",
            "Horizon h is UTC hour h-1 (origins are midnight UTC): hour-of-day and",
            "horizon effects are confounded. Per-hour bands rest on few days; read them",
            "as where to look, not as findings.",
            "",
            "| h | UTC | p50 MAE (MW) | band | coverage | band |",
            "|---|---|---|---|---|---|",
        ]
        for r in horizon.itertuples(index=False):
            lines.append(
                f"| {r.horizon} | {r.utc_hour:02d} | {r.mae_p50:,.0f} "
                f"| [{r.mae_lo:,.0f}, {r.mae_hi:,.0f}] | {r.coverage:.0%} "
                f"| [{r.coverage_lo:.0%}, {r.coverage_hi:.0%}] |"
            )
    return lines


def zone_row(s: dict, now: pd.Timestamp) -> dict:
    row = {c: np.nan for c in ZONE_COLS}
    row.update({"run_at": now, "n_origins": s["n_origins"]})
    if s.get("status") == "no_data":
        row.update({"coverage_zone": "no_data", "skill_zone": "no_data"})
        return row
    row.update(
        {
            "window_first": s["window_first"],
            "window_last": s["window_last"],
            "coverage": s["coverage"],
            "coverage_lo": s["coverage_band"][0],
            "coverage_hi": s["coverage_band"][1],
            "coverage_zone": s["coverage_zone"],
            "mae_p50": s["mae_p50"],
            "mae_lo": s["mae_band"][0],
            "mae_hi": s["mae_band"][1],
            "skill": s["skill"],
            "skill_lo": s["skill_band"][0],
            "skill_hi": s["skill_band"][1],
            "skill_zone": s["skill_zone"],
        }
    )
    return row


def run(
    log_dir: Path,
    manifest: dict,
    gate_baseline: dict,
    cfg: MonitoringConfig,
    now: pd.Timestamp,
) -> int:
    hourly = read_log(log_dir / "hourly.csv", HOURLY_COLS)
    if hourly.empty:
        s: dict = {"n_origins": 0, "status": "no_data"}
        horizon = None
    else:
        daily = daily_table(hourly)
        s = window_summary(daily, cfg)
        horizon = by_horizon(hourly, cfg)
        # precomputed for the dashboard, which reads files and computes nothing
        write_log(rolling_table(daily, cfg), log_dir / "rolling.csv")
        horizon.to_csv(log_dir / "by_horizon.csv", index=False, float_format="%.4f")
    lines = report_lines(s, horizon, baselines(manifest, gate_baseline), cfg, now)
    (log_dir / "ACCURACY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zpath = log_dir / "zones.csv"
    zones = read_log(zpath, ZONE_COLS) if zpath.exists() else pd.DataFrame(columns=ZONE_COLS)
    zones = append(zones, pd.DataFrame([zone_row(s, now)], columns=ZONE_COLS))
    write_log(zones, zpath)

    print("\n".join(lines))
    red = [z for z in (s.get("coverage_zone", ""), s.get("skill_zone", "")) if z.startswith("red")]
    if red:
        print(f"\nRED: {', '.join(red)}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", type=Path, required=True, help="directory holding hourly.csv")
    args = ap.parse_args()
    serving = load_serving_config()
    manifest_path = serving.output_root / serving.name / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = json.loads(GATE_BASELINE_PATH.read_text(encoding="utf-8"))
    return run(args.log_dir, manifest, gate, load_config(), pd.Timestamp.now(tz="UTC").floor("s"))


if __name__ == "__main__":
    sys.exit(main())
