"""Freeze the Phase 3 production pipeline into a serving artifact.

One artifact = one training cutoff C (configs/serving.toml). Everything in it
is built from actual load strictly before C, the way run_holdout_phase3 built
each holdout forecast:
  1. three quantile models (p10/p50/p90) trained on origins < C with the
     backtest's own training code (quantile_lgbm.fit_quantile_models);
  2. conformal shifts from out-of-sample daily forecasts issued in the
     calibration window before C -- each from a model trained only on origins
     before ITS origin, clean origins only, whole target day observed by C;
  3. a reference forecast for origin C computed from the SAVED model files:
     the golden output the serving code must reproduce exactly.

Load at or after C is masked (NaN), never dropped: the hourly grid stays
intact (same reasoning as exclusions.py), so origin C's features are built
exactly as serving builds them. The artifact may serve origins >= C only:
earlier origins were inside its training data.

Models are saved in LightGBM's native text format at the early-stopping best
iteration -- no pickle: portable across library versions, human-readable,
and loading it cannot execute code. manifest.json hashes every file plus the
data, exclusions and config, and records the git commit, so a served
forecast traces back to exactly what produced it. The directory is written
as <name>.partial and renamed only when complete, so a crashed export never
leaves a half-written artifact that looks valid.

The training data must be final: the cutoff may not lie after the
provisional line of the feature table (newest load minus provisional_days,
configs/serving_gates.toml). Otherwise the last weeks of training and the
calibration pool would be ENTSO-E's provisional, 20-35%-low values.

Run:  uv run python -m gridcast.models.export_artifact
      uv run python -m gridcast.models.export_artifact --cutoff 2026-10-22
      (--cutoff builds a challenger at another cutoff without editing
      serving.toml; serving.toml names the champion)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast.models.backtest import FEATURES_PATH
from gridcast.models.conformal import conformal_shift
from gridcast.models.exclusions import DEFAULT_PATH as EXCLUSIONS_PATH
from gridcast.models.exclusions import (
    Exclusion,
    apply_exclusions,
    contaminated_origins,
    load_exclusions,
)
from gridcast.models.lgbm import VAL_SPAN
from gridcast.models.metrics import DEFAULT_QUANTILES, rearrange
from gridcast.models.ml_features import (
    N_HORIZONS,
    TARGET_COL,
    build_design_matrix,
    daily_origins,
)
from gridcast.models.quantile_lgbm import (
    DROP_FEATURES,
    NAMES,
    fit_predict_fold_quantiles,
    fit_quantile_models,
    quantile_params,
)
from gridcast.models.trust import COLD_PCT, FLAG_COLS, SHIFT_FRAC, trust_flags

HOUR = pd.Timedelta("1h")
TZ = "Europe/Amsterdam"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "serving.toml"
ROUNDTRIP_ATOL_MW = 1e-6


@dataclass(frozen=True)
class ArtifactConfig:
    train_cutoff: pd.Timestamp
    calibration_window: pd.Timedelta
    min_calibration_origins: int
    output_root: Path

    @property
    def name(self) -> str:
        return f"lgbm_quantile_{self.train_cutoff:%Y-%m-%d}"


def load_config(path: str | Path = CONFIG_PATH) -> ArtifactConfig:
    """Parse [artifact] from the serving TOML. The cutoff must be a
    timezone-aware midnight UTC: it is also the first servable origin."""
    raw = tomllib.loads(Path(path).read_bytes().decode("utf-8-sig"))["artifact"]
    cutoff = pd.Timestamp(raw["train_cutoff"])
    if cutoff.tz is None:
        raise ValueError("train_cutoff has no timezone; write UTC with a trailing Z")
    cutoff = cutoff.tz_convert("UTC")
    if cutoff != cutoff.floor("D"):
        raise ValueError(f"train_cutoff {cutoff} must be midnight UTC (a forecast origin)")
    root = Path(raw["output_root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return ArtifactConfig(
        train_cutoff=cutoff,
        calibration_window=pd.Timedelta(days=int(raw["calibration_window_days"])),
        min_calibration_origins=int(raw["min_calibration_origins"]),
        output_root=root,
    )


def mask_from(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Copy of `frame` with load at or after `cutoff` set to NaN. Rows are
    kept so row-based rolling windows keep their meaning."""
    out = frame.copy()
    ts = pd.DatetimeIndex(out["timestamp"])
    out.loc[np.asarray(ts >= cutoff), TARGET_COL] = np.nan
    return out


def calibration_origins(
    origins: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    window: pd.Timedelta,
    exclusions: list[Exclusion],
) -> pd.DatetimeIndex:
    """Origins whose forecasts may calibrate a model frozen at `cutoff`:
    issued within the trailing window, whole target day observed before the
    cutoff, and not contaminated by an exclusion."""
    keep = np.asarray(origins >= cutoff - window) & np.asarray(
        origins + N_HORIZONS * HOUR <= cutoff
    )
    cand = origins[keep]
    return cand[~contaminated_origins(cand, exclusions)]


def calibration_forecasts(matrix: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Out-of-sample forecasts, one retrain per origin (models see only
    origins before each forecast's own origin). Rearranged, uncalibrated."""
    records: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for k, o in enumerate(origins, start=1):
        preds, _ = fit_predict_fold_quantiles(matrix, o)
        test = matrix[matrix["origin"] == o].sort_values("horizon")
        q = rearrange(np.column_stack([preds[n] for n in NAMES]))
        rec = pd.DataFrame(
            {
                "origin": o,
                "timestamp": test["timestamp"].to_numpy(),
                "horizon": test["horizon"].astype(int).to_numpy(),
                "y_true": test["y_true"].to_numpy(),
            }
        )
        for j, n in enumerate(NAMES):
            rec[n] = q[:, j]
        records.append(rec)
        if k == 1 or k % 10 == 0 or k == len(origins):
            el = time.perf_counter() - t0
            eta = el / k * (len(origins) - k)
            print(f"  origin {k:2d}/{len(origins)}  {el:5.0f}s elapsed  ~{eta / 60:4.1f} min left")
    return pd.concat(records, ignore_index=True)


def conformal_shifts(pool: pd.DataFrame) -> dict[str, float]:
    """Per-quantile conformal shift from the pool's residuals (y - q)."""
    pool = pool.dropna(subset=["y_true"])
    return {
        n: conformal_shift((pool["y_true"] - pool[n]).to_numpy(), DEFAULT_QUANTILES[n])
        for n in NAMES
    }


def save_models(models: dict[str, lgb.LGBMRegressor], out_dir: Path) -> dict[str, Path]:
    """Native text format, truncated at each model's best iteration."""
    paths: dict[str, Path] = {}
    for name, model in models.items():
        path = out_dir / f"{name}.txt"
        model.booster_.save_model(str(path), num_iteration=model.best_iteration_)
        paths[name] = path
    return paths


def load_boosters(paths: dict[str, Path]) -> dict[str, lgb.Booster]:
    return {name: lgb.Booster(model_file=str(path)) for name, path in paths.items()}


def roundtrip_max_diff(
    models: dict[str, lgb.LGBMRegressor], boosters: dict[str, lgb.Booster], x: pd.DataFrame
) -> float:
    """Largest |in-memory - reloaded| prediction over all models and rows."""
    worst = 0.0
    for name, model in models.items():
        a = model.predict(x, num_iteration=model.best_iteration_)
        b = boosters[name].predict(x)
        worst = max(worst, float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    return worst


def reference_forecast(
    boosters: dict[str, lgb.Booster],
    feats: list[str],
    matrix: pd.DataFrame,
    origin: pd.Timestamp,
    shifts: dict[str, float],
    features_raw: pd.DataFrame,
    exclusions: list[Exclusion],
) -> pd.DataFrame:
    """Raw, rearranged and calibrated quantiles plus trust flags for one
    origin, computed from the reloaded boosters (the serving path)."""
    rows = matrix[matrix["origin"] == origin].sort_values("horizon").reset_index(drop=True)
    if len(rows) != N_HORIZONS:
        raise ValueError(f"expected {N_HORIZONS} rows for origin {origin}, got {len(rows)}")
    raw = np.column_stack([boosters[n].predict(rows[feats]) for n in NAMES])
    q = rearrange(raw)
    cal = rearrange(q + np.array([shifts[n] for n in NAMES]))
    out = pd.DataFrame(
        {
            "origin": origin,
            "timestamp": rows["timestamp"].to_numpy(),
            "horizon": rows["horizon"].astype(int).to_numpy(),
        }
    )
    for j, n in enumerate(NAMES):
        out[f"{n}_raw"] = raw[:, j]
        out[n] = q[:, j]
        out[f"{n}_cal"] = cal[:, j]
        out[f"shift_{n}"] = shifts[n]
    return trust_flags(out, features_raw, exclusions)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            res = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return res.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def parse_cutoff(text: str) -> pd.Timestamp:
    """A date or timestamp, read as UTC; must be midnight (an origin)."""
    ts = pd.Timestamp(text)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    if ts != ts.floor("D"):
        raise ValueError(f"cutoff {ts} must be midnight UTC")
    return ts


def check_final(cutoff: pd.Timestamp, load_ts: pd.DatetimeIndex, provisional_days: int) -> None:
    """Refuse a cutoff whose training data would still be provisional."""
    data_as_of = load_ts.max()
    line = data_as_of - pd.Timedelta(days=provisional_days)
    if cutoff > line:
        raise SystemExit(
            f"cutoff {cutoff} is after the provisional line {line} (data as of {data_as_of}, "
            f"{provisional_days} provisional days): training data would not be final"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="freeze a serving artifact")
    ap.add_argument("--cutoff", help="override train_cutoff (YYYY-MM-DD, midnight UTC)")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.cutoff:
        cfg = dataclasses.replace(cfg, train_cutoff=parse_cutoff(args.cutoff))
    cutoff = cfg.train_cutoff
    out_dir = cfg.output_root / cfg.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: artifacts are immutable; delete it deliberately")
    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)

    exclusions = load_exclusions()
    raw = pd.read_parquet(FEATURES_PATH)
    ts = pd.DatetimeIndex(raw["timestamp"])
    load_ts = ts[np.asarray(raw[TARGET_COL].notna())]
    from gridcast.serving.forecaster import load_provisional_days  # serving imports this module

    provisional_days = load_provisional_days()
    check_final(cutoff, load_ts, provisional_days)
    ref_end = cutoff + (N_HORIZONS - 1) * HOUR
    if ts.max() < ref_end:
        raise SystemExit(f"feature table ends {ts.max()}, before reference day end {ref_end}")
    frame = mask_from(apply_exclusions(raw, exclusions), cutoff)
    origins = daily_origins(ts.sort_values())
    origins = origins[np.asarray(origins <= cutoff)]

    git = _git_state()
    print(f"artifact     : {cfg.name}")
    print(f"git          : {git['commit']} (dirty={git['dirty']})")
    print(f"train cutoff : {cutoff}  ({int(np.asarray(ts >= cutoff).sum())} h of load masked)")

    t0 = time.perf_counter()
    matrix = build_design_matrix(frame, origins)
    print(f"design matrix: {matrix.shape[0]} rows x {matrix.shape[1]} cols")
    print(f"               built in {time.perf_counter() - t0:.1f}s")

    train = matrix[(matrix["origin"] < cutoff) & matrix["y_true"].notna()]
    n_val = int((train["origin"] >= cutoff - VAL_SPAN).sum())
    print(
        f"training     : {len(train)} rows ({len(train) - n_val} fit, {n_val} early-stopping), "
        f"origins {train['origin'].min().date()} -> {train['origin'].max().date()}"
    )

    cal = calibration_origins(origins, cutoff, cfg.calibration_window, exclusions)
    if len(cal) < cfg.min_calibration_origins:
        raise SystemExit(
            f"only {len(cal)} clean calibration origins (< {cfg.min_calibration_origins})"
        )
    print(f"calibration  : {len(cal)} clean origins, {cal[0].date()} -> {cal[-1].date()}")

    print("\nfitting final models ...")
    t0 = time.perf_counter()
    models, feats = fit_quantile_models(matrix, cutoff)
    iters = {n: int(models[n].best_iteration_ or 0) for n in NAMES}
    print(f"  done in {time.perf_counter() - t0:.0f}s, best iterations {iters}")

    print("\ncalibration forecasts (one retrain per origin) ...")
    pool = calibration_forecasts(matrix, cal)
    shifts = conformal_shifts(pool)
    y = pool["y_true"].to_numpy()
    lo, mid, hi = (pool[n].to_numpy() for n in NAMES)
    raw_cov = float(np.mean((y >= lo) & (y <= hi)))
    pool_mae = float(np.mean(np.abs(y - mid)))

    partial.mkdir(parents=True)
    paths = save_models(models, partial)
    boosters = load_boosters(paths)
    check = matrix[matrix["origin"] >= cutoff - cfg.calibration_window]
    rt = roundtrip_max_diff(models, boosters, check[feats])
    print(f"\nsave/load round trip: max |diff| {rt:.2e} MW over {len(check)} rows")
    if rt > ROUNDTRIP_ATOL_MW:
        raise SystemExit("round trip mismatch: saved models do not reproduce in-memory ones")

    ref = reference_forecast(boosters, feats, matrix, cutoff, shifts, raw, exclusions)
    pool.to_parquet(partial / "calibration_forecasts.parquet", index=False)
    ref.to_parquet(partial / "reference_forecast.parquet", index=False)

    manifest = {
        "artifact": cfg.name,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git": git,
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "train_cutoff": cutoff.isoformat(),
        "serve_rule": "origins >= train_cutoff only; earlier origins were in training",
        "quantiles": {n: DEFAULT_QUANTILES[n] for n in NAMES},
        "features": feats,
        "dropped_features": list(DROP_FEATURES),
        "training": {
            # per model: the shared base params with objective="quantile" and
            # alpha=tau, as trained (until 2026-10-01 this recorded the base
            # params alone, which showed objective "l1")
            "params": {n: quantile_params(DEFAULT_QUANTILES[n]) for n in NAMES},
            "first_origin": train["origin"].min().isoformat(),
            "last_origin": train["origin"].max().isoformat(),
            "n_rows": len(train),
            "n_early_stopping_rows": n_val,
            "validation_span": str(VAL_SPAN),
            "best_iteration": iters,
        },
        "calibration": {
            "method": "asymmetric conformal shift per quantile (conformal.conformal_shift)",
            "window_days": cfg.calibration_window.days,
            "n_origins": len(cal),
            "first_origin": cal[0].isoformat(),
            "last_origin": cal[-1].isoformat(),
            "shifts_mw": shifts,
            "pool_raw_interval_coverage": raw_cov,
            "pool_q50_mae_mw": pool_mae,
            "note": "pool forecasts are out-of-sample but NOT a sealed holdout",
        },
        "trust": {"flags": FLAG_COLS, "cold_pct": COLD_PCT, "shift_frac": SHIFT_FRAC},
        "roundtrip_max_abs_diff_mw": rt,
        "inputs": {
            "features_sha256": _sha256(Path(FEATURES_PATH)),
            "features_last_timestamp": ts.max().isoformat(),
            "data_as_of": load_ts.max().isoformat(),
            "provisional_days": provisional_days,
            "exclusions_sha256": _sha256(Path(EXCLUSIONS_PATH)),
            "serving_config_sha256": _sha256(CONFIG_PATH),
        },
        "files": {p.name: _sha256(p) for p in sorted(partial.iterdir())},
    }
    (partial / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    partial.rename(out_dir)

    print("\n=== conformal shifts (MW) ===")
    for n in NAMES:
        print(f"  {n}: {shifts[n]:+8.1f}")
    print(f"pool ({len(cal)} origins, out-of-sample, not sealed):")
    print(f"  raw 80% interval coverage {raw_cov:.1%}, q50 MAE {pool_mae:.1f} MW")

    view = ref[["timestamp", "q10_cal", "q50_cal", "q90_cal", "trust_reasons"]].copy()
    view["timestamp"] = view["timestamp"].dt.tz_convert(TZ).dt.strftime("%m-%d %H:%M")
    print(f"\n=== reference forecast, origin {cutoff} (local time) ===")
    print(view.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print(f"\nwrote {out_dir}")
    for name, digest in manifest["files"].items():
        print(f"  {name:32s} {digest[:12]}")


if __name__ == "__main__":
    main()
