"""Shared synthetic data: an hourly table with the real feature schema, and
a quantile artifact trained on it with the real training and export code."""

import json

import numpy as np
import pandas as pd
import pytest

from gridcast.models.export_artifact import mask_from, save_models
from gridcast.models.metrics import DEFAULT_QUANTILES
from gridcast.models.ml_features import build_design_matrix, daily_origins
from gridcast.models.quantile_lgbm import fit_quantile_models
from gridcast.serving.forecaster import file_sha256

SYNTH_CUTOFF = pd.Timestamp("2024-10-01", tz="UTC")
SYNTH_SHIFTS = {"q10": -50.0, "q50": 10.0, "q90": 60.0}


def make_frame(days: int = 300) -> pd.DataFrame:
    """Hourly from 2024-01-01 UTC: daily and weekly shape, temperature
    dependence, noise. 300 days ends 2024-10-26 23:00."""
    ts = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    n = len(ts)
    hour = ts.hour.to_numpy()
    doy = ts.dayofyear.to_numpy()
    weekend = np.asarray(ts.dayofweek >= 5)
    temp = 10 + 8 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 1.5, n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "hour": hour.astype("int8"),
            "dow": ts.dayofweek.to_numpy().astype("int8"),
            "is_weekend": weekend.astype("int8"),
            "month": ts.month.to_numpy().astype("int8"),
            "day_of_year": doy.astype("int16"),
            "is_holiday": np.zeros(n, dtype="int8"),
            "temperature_2m": temp,
            "shortwave_radiation": np.clip(600 * np.sin(np.pi * (hour - 6) / 12), 0, None),
            "wind_speed_10m": rng.gamma(2.0, 3.0, n),
        }
    )
    df["temp_mean_24h"] = df["temperature_2m"].rolling(24, min_periods=1).mean()
    load = (
        12000
        + 1500 * np.sin(2 * np.pi * (hour - 8) / 24)
        - 120 * (temp - 10)
        - 800 * weekend
        + rng.normal(0, 150, n)
    )
    df["load_mw"] = load
    df["da_forecast_mw"] = load * 0.8
    return df


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="session")
def synthetic_artifact_dir(synthetic_frame, tmp_path_factory):
    """Artifact frozen at SYNTH_CUTOFF with fixed SYNTH_SHIFTS. Trained once
    per test session."""
    out = tmp_path_factory.mktemp("artifact")
    origins = daily_origins(pd.DatetimeIndex(synthetic_frame["timestamp"]))
    origins = origins[np.asarray(origins <= SYNTH_CUTOFF)]
    matrix = build_design_matrix(mask_from(synthetic_frame, SYNTH_CUTOFF), origins)
    models, feats = fit_quantile_models(matrix, SYNTH_CUTOFF)
    save_models(models, out)
    manifest = {
        "artifact": "synthetic",
        "train_cutoff": SYNTH_CUTOFF.isoformat(),
        "quantiles": dict(DEFAULT_QUANTILES),
        "features": feats,
        "calibration": {"shifts_mw": SYNTH_SHIFTS},
        "inputs": {"features_sha256": "not-a-real-hash"},
        "files": {p.name: file_sha256(p) for p in sorted(out.glob("*.txt"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return out
