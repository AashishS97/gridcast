"""Serve calibrated forecasts from a frozen artifact -- no FastAPI here.

The serving path is the export's reference path, step for step:
  features (exclusions applied, load masked from the origin on)
  -> build_design_matrix for ONE origin (the same builder as training)
  -> saved boosters -> rearrange -> + conformal shifts -> rearrange
  -> trust flags.
tests/test_serving_parity.py checks the result against the artifact's
reference_forecast.parquet: the same numbers, not approximately the same.

Every request passes gates first. A failed gate is a refusal with a reason
code, never a degraded forecast:
  not_midnight_utc     origins are midnight UTC (the evaluated contract)
  before_cutoff        origin < train_cutoff: the model trained on that day
  data_unavailable     lookback load missing (beyond the data, or gaps)
  excluded_data        the 336h lookback overlaps a data exclusion
  provisional_data     the lookback reaches into the last N days before the
                       data vintage: ENTSO-E publishes NL actuals ~20-35% low
                       for ~28 days and revises them later
  weather_unavailable  target-hour weather missing (weather features are
                       perfect-prognosis; the model never saw NaN weather
                       in volume)
Exclusions encode evidence about specific periods; the provisional rule
encodes the mechanism, so it also covers a window nobody has inspected yet.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast.models.backtest import FEATURES_PATH
from gridcast.models.exclusions import LOOKBACK_HOURS, Exclusion, apply_exclusions, load_exclusions
from gridcast.models.export_artifact import PROJECT_ROOT, load_config, mask_from
from gridcast.models.metrics import rearrange
from gridcast.models.ml_features import N_HORIZONS, TARGET_COL, WEATHER_COLS, build_design_matrix
from gridcast.models.trust import trust_flags

HOUR = pd.Timedelta("1h")
GATES_PATH = PROJECT_ROOT / "configs" / "serving_gates.toml"


class ArtifactError(RuntimeError):
    """The artifact on disk is incomplete, inconsistent or modified."""


class NotServable(ValueError):
    """A request failed a serving gate. `code` is stable and machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Artifact:
    name: str
    path: Path
    manifest: dict
    boosters: dict[str, lgb.Booster]
    names: list[str]  # quantile names in ascending tau
    features: list[str]
    shifts: dict[str, float]
    train_cutoff: pd.Timestamp


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifact(path: str | Path) -> Artifact:
    """Load and verify: every manifest-listed file must exist with its
    recorded sha256, and every model's feature order must match."""
    path = Path(path)
    mpath = path / "manifest.json"
    if not mpath.exists():
        raise ArtifactError(f"no manifest.json in {path}")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    for fname, digest in manifest["files"].items():
        f = path / fname
        if not f.exists():
            raise ArtifactError(f"{fname} is listed in the manifest but missing")
        if file_sha256(f) != digest:
            raise ArtifactError(f"{fname}: sha256 mismatch (modified after export)")
    quantiles = manifest["quantiles"]
    names = sorted(quantiles, key=quantiles.__getitem__)
    feats = list(manifest["features"])
    boosters: dict[str, lgb.Booster] = {}
    for n in names:
        booster = lgb.Booster(model_file=str(path / f"{n}.txt"))
        if booster.feature_name() != feats:
            raise ArtifactError(f"{n}.txt feature order differs from the manifest")
        boosters[n] = booster
    return Artifact(
        name=manifest["artifact"],
        path=path,
        manifest=manifest,
        boosters=boosters,
        names=names,
        features=feats,
        shifts={n: float(manifest["calibration"]["shifts_mw"][n]) for n in names},
        train_cutoff=pd.Timestamp(manifest["train_cutoff"]).tz_convert("UTC"),
    )


def load_provisional_days(path: str | Path = GATES_PATH) -> int:
    raw = tomllib.loads(Path(path).read_bytes().decode("utf-8-sig"))
    days = int(raw["gates"]["provisional_days"])
    if days < 0:
        raise ValueError("provisional_days must be >= 0")
    return days


class Forecaster:
    """Gated, calibrated forecasts for one artifact over one data vintage."""

    def __init__(
        self,
        artifact: Artifact,
        features: pd.DataFrame,
        exclusions: list[Exclusion],
        provisional_days: int,
        features_sha256: str | None = None,
    ) -> None:
        if provisional_days < 0:
            raise ValueError("provisional_days must be >= 0")
        raw = features.sort_values("timestamp").reset_index(drop=True)
        ts = pd.DatetimeIndex(raw["timestamp"])
        load = raw[TARGET_COL].to_numpy(dtype=float)
        has_load = ~np.isnan(load)
        if not has_load.any():
            raise ValueError("feature table has no actual load")

        self.artifact = artifact
        self.exclusions = exclusions
        self.provisional_days = provisional_days
        self.features_sha256 = features_sha256
        self._raw = raw
        self._frame = apply_exclusions(raw, exclusions)
        self._ts = ts
        self._load = pd.Series(load, index=ts)
        self._weather = raw.set_index("timestamp")[WEATHER_COLS]
        self.data_as_of = pd.Timestamp(ts[has_load].max())
        self.provisional_from = self.data_as_of - pd.Timedelta(days=provisional_days)

    @property
    def data_matches_artifact(self) -> bool | None:
        """Same features.parquet the artifact was exported from? None if unknown."""
        if self.features_sha256 is None:
            return None
        return self.features_sha256 == self.artifact.manifest["inputs"]["features_sha256"]

    def check(self, origin: str | pd.Timestamp) -> tuple[str, str] | None:
        """(code, message) of the first failed gate, or None if servable."""
        o = pd.Timestamp(origin)
        if o.tz is None:
            return ("not_midnight_utc", f"{origin!r} has no timezone; send UTC, e.g. ...T00:00:00Z")
        o = o.tz_convert("UTC")
        if o != o.floor("D"):
            return ("not_midnight_utc", f"{o} is not midnight UTC")

        cutoff = self.artifact.train_cutoff
        if o < cutoff:
            return (
                "before_cutoff",
                f"origin {o.date()} precedes the training cutoff {cutoff.date()}: "
                "the model was trained on that day",
            )

        lo = o - LOOKBACK_HOURS * HOUR
        window = self._load[(self._load.index >= lo) & (self._load.index < o)]
        observed = int(window.notna().sum())
        if observed < LOOKBACK_HOURS:
            return (
                "data_unavailable",
                f"{LOOKBACK_HOURS - observed} of {LOOKBACK_HOURS} lookback hours of load "
                f"missing (data ends {self.data_as_of})",
            )

        for ex in self.exclusions:
            if ex.start < o and (ex.end is None or lo < ex.end):
                end = "open-ended" if ex.end is None else str(ex.end)
                return (
                    "excluded_data",
                    f"lookback {lo} -> {o} overlaps exclusion {ex.start} -> {end}: {ex.reason}",
                )

        if o > self.provisional_from:
            return (
                "provisional_data",
                f"lookback ends {o - HOUR}, within {self.provisional_days} days of the data "
                f"vintage {self.data_as_of}: ENTSO-E actuals there are still provisional",
            )

        targets = pd.date_range(o, periods=N_HORIZONS, freq="h")
        w = self._weather.reindex(targets)
        missing = int(w.isna().any(axis=1).sum())
        if missing:
            return (
                "weather_unavailable",
                f"weather missing for {missing} of {N_HORIZONS} target hours",
            )
        return None

    def forecast(self, origin: str | pd.Timestamp) -> pd.DataFrame:
        """24 rows: raw, rearranged and calibrated quantiles, the conformal
        shifts, trust flags. Raises NotServable if a gate fails."""
        return self.forecast_with_inputs(origin)[0]

    def forecast_with_inputs(self, origin: str | pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The forecast plus the exact feature rows the boosters predicted from
        (origin, timestamp, horizon, then the artifact's other features in
        model order). One computation, so the logged inputs cannot differ from the
        ones used."""
        problem = self.check(origin)
        if problem is not None:
            raise NotServable(*problem)
        o = pd.Timestamp(origin).tz_convert("UTC")

        frame = mask_from(self._frame, o)
        rows = build_design_matrix(frame, pd.DatetimeIndex([o]))
        rows = rows.sort_values("horizon").reset_index(drop=True)
        a = self.artifact
        raw = np.column_stack([a.boosters[n].predict(rows[a.features]) for n in a.names])
        q = rearrange(raw)
        cal = rearrange(q + np.array([a.shifts[n] for n in a.names]))

        out = pd.DataFrame(
            {
                "origin": o,
                "timestamp": rows["timestamp"].to_numpy(),
                "horizon": rows["horizon"].astype(int).to_numpy(),
            }
        )
        for j, n in enumerate(a.names):
            out[f"{n}_raw"] = raw[:, j]
            out[n] = q[:, j]
            out[f"{n}_cal"] = cal[:, j]
            out[f"shift_{n}"] = a.shifts[n]
        inputs = out[["origin", "timestamp", "horizon"]].copy()
        for f in a.features:  # horizon is itself a feature: equal values, kept once
            if f not in inputs.columns:
                inputs[f] = rows[f].to_numpy()
        return trust_flags(out, self._raw, self.exclusions), inputs

    def final_actuals(self, origin: str | pd.Timestamp) -> np.ndarray | None:
        """Observed load for the origin's 24 target hours, or None while any of
        them is provisional, excluded or missing. Scoring waits for this: the
        lookback gate alone lets an origin be served while its own target day
        is still provisional, and scoring against those values would compare
        the forecast with numbers published 20-35% low."""
        o = pd.Timestamp(origin).tz_convert("UTC")
        end = o + N_HORIZONS * HOUR
        if end > self.provisional_from:
            return None
        for ex in self.exclusions:
            if ex.start < end and (ex.end is None or o < ex.end):
                return None
        values = self._load.reindex(pd.date_range(o, periods=N_HORIZONS, freq="h")).to_numpy()
        if np.isnan(values).any():
            return None
        return values

    def daily_ledger(self, days: int) -> pd.DataFrame:
        """What the upstream has published for the last `days` UTC calendar days
        up to the data vintage: hours of actual load present, their mean, and
        the mean temperature. Raw, exclusions NOT applied -- this records the
        source, not what the model may use. Logged every run, it is the only
        record of how a past day's published value changed between vintages."""
        if days < 1:
            raise ValueError("days must be >= 1")
        last = self.data_as_of.floor("D")
        days_idx = pd.date_range(last - pd.Timedelta(days=days - 1), last, freq="1D")
        lo, hi = days_idx[0], last + pd.Timedelta("1D")
        load = self._load[(self._load.index >= lo) & (self._load.index < hi)]
        temp = self._weather["temperature_2m"]
        temp = temp[(temp.index >= lo) & (temp.index < hi)]
        load_by_day = load.groupby(load.index.floor("D"))
        temp_by_day = temp.groupby(temp.index.floor("D"))
        return pd.DataFrame(
            {
                "day": days_idx,
                "load_hours": load_by_day.count().reindex(days_idx, fill_value=0).to_numpy(int),
                "load_mean_mw": load_by_day.mean().reindex(days_idx).to_numpy(),
                "temp_mean_c": temp_by_day.mean().reindex(days_idx).to_numpy(),
            }
        )

    def actual_load(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Published actual load at the given timestamps (NaN where absent).
        Raw: exclusions are not applied."""
        return self._load.reindex(pd.DatetimeIndex(timestamps)).to_numpy(dtype=float)

    def servable_origins(self) -> pd.DatetimeIndex:
        """Every origin from the cutoff to the end of the data that passes all gates."""
        start = self.artifact.train_cutoff
        end = self._ts.max().floor("D")
        if end < start:
            return pd.DatetimeIndex([], tz="UTC")
        cand = pd.date_range(start, end, freq="1D")
        return pd.DatetimeIndex([o for o in cand if self.check(o) is None])

    def status(self) -> dict:
        servable = self.servable_origins()
        return {
            "artifact": self.artifact.name,
            "train_cutoff": self.artifact.train_cutoff.isoformat(),
            "data_as_of": self.data_as_of.isoformat(),
            "provisional_days": self.provisional_days,
            "provisional_from": self.provisional_from.isoformat(),
            "data_matches_artifact": self.data_matches_artifact,
            "n_servable": len(servable),
            "first_servable": servable[0].isoformat() if len(servable) else None,
            "last_servable": servable[-1].isoformat() if len(servable) else None,
        }


def from_config(artifact_dir: Path | None = None) -> Forecaster:
    """The production wiring: configured artifact, current features, current
    exclusions, configured gates. artifact_dir overrides the configured
    artifact (a challenger in its trial); everything else stays the same, so
    both models see identical data and gates."""
    cfg = load_config()
    artifact = load_artifact(
        artifact_dir if artifact_dir is not None else cfg.output_root / cfg.name
    )
    features_path = Path(FEATURES_PATH)
    if not features_path.is_absolute():
        features_path = PROJECT_ROOT / features_path
    return Forecaster(
        artifact=artifact,
        features=pd.read_parquet(features_path),
        exclusions=load_exclusions(),
        provisional_days=load_provisional_days(),
        features_sha256=file_sha256(features_path),
    )
