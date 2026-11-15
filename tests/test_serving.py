"""Forecaster on a synthetic artifact built with the REAL training and export
code: output contract, shift arithmetic, every gate at its boundary, a
serving leakage canary, and tamper detection."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from gridcast.models.exclusions import Exclusion
from gridcast.models.export_artifact import mask_from, save_models
from gridcast.models.metrics import DEFAULT_QUANTILES
from gridcast.models.ml_features import build_design_matrix, daily_origins
from gridcast.models.quantile_lgbm import fit_quantile_models
from gridcast.serving.forecaster import (
    ArtifactError,
    Forecaster,
    NotServable,
    file_sha256,
    load_artifact,
)

CUTOFF = pd.Timestamp("2024-10-01", tz="UTC")
SHIFTS = {"q10": -50.0, "q50": 10.0, "q90": 60.0}
NAMES = ["q10", "q50", "q90"]


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture(scope="module")
def artifact_dir(synthetic_frame, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifact")
    origins = daily_origins(pd.DatetimeIndex(synthetic_frame["timestamp"]))
    origins = origins[np.asarray(origins <= CUTOFF)]
    matrix = build_design_matrix(mask_from(synthetic_frame, CUTOFF), origins)
    models, feats = fit_quantile_models(matrix, CUTOFF)
    save_models(models, out)
    manifest = {
        "artifact": "synthetic",
        "train_cutoff": CUTOFF.isoformat(),
        "quantiles": dict(DEFAULT_QUANTILES),
        "features": feats,
        "calibration": {"shifts_mw": SHIFTS},
        "inputs": {"features_sha256": "not-a-real-hash"},
        "files": {p.name: file_sha256(p) for p in sorted(out.glob("*.txt"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def artifact(artifact_dir):
    return load_artifact(artifact_dir)


def make(artifact, frame, exclusions=(), provisional_days=0):
    return Forecaster(artifact, frame, list(exclusions), provisional_days)


def refusal(fc, origin) -> str:
    with pytest.raises(NotServable) as info:
        fc.forecast(origin)
    return info.value.code


def test_forecast_contract(artifact, synthetic_frame):
    out = make(artifact, synthetic_frame).forecast(utc("2024-10-10"))
    assert len(out) == 24
    assert out["horizon"].tolist() == list(range(1, 25))
    assert (out["timestamp"].iloc[0], out["timestamp"].iloc[-1]) == (
        utc("2024-10-10"),
        utc("2024-10-10 23:00"),
    )
    assert (out["q10_cal"] <= out["q50_cal"]).all() and (out["q50_cal"] <= out["q90_cal"]).all()
    for n in NAMES:
        assert (out[f"shift_{n}"] == SHIFTS[n]).all()
    assert {"trust_low", "trust_reasons"} <= set(out.columns)


def test_calibration_is_shift_of_rearranged(artifact, synthetic_frame):
    out = make(artifact, synthetic_frame).forecast(utc("2024-10-10"))
    # shifts widen the interval, so rearranging after shifting is a no-op here
    for n in NAMES:
        np.testing.assert_allclose(out[f"{n}_cal"], out[n] + SHIFTS[n], rtol=0, atol=1e-9)


def test_before_cutoff_refused(artifact, synthetic_frame):
    fc = make(artifact, synthetic_frame)
    assert refusal(fc, utc("2024-09-30")) == "before_cutoff"
    assert fc.check(CUTOFF) is None


@pytest.mark.parametrize("origin", ["2024-10-10T06:00:00Z", "2024-10-10"])
def test_not_midnight_utc_refused(artifact, synthetic_frame, origin):
    assert refusal(make(artifact, synthetic_frame), origin) == "not_midnight_utc"


def test_excluded_lookback_refused(artifact, synthetic_frame):
    ex = Exclusion(utc("2024-10-03"), utc("2024-10-04"), "test", "test")
    fc = make(artifact, synthetic_frame, exclusions=[ex])
    assert refusal(fc, utc("2024-10-10")) == "excluded_data"
    # lookback of Oct 18 starts exactly at the exclusion end: clean
    assert fc.check(utc("2024-10-18")) is None


def test_provisional_window_refused(artifact, synthetic_frame):
    fc = make(artifact, synthetic_frame, provisional_days=10)
    assert fc.data_as_of == utc("2024-10-26 23:00")
    assert refusal(fc, utc("2024-10-17")) == "provisional_data"
    assert fc.check(utc("2024-10-16")) is None


def test_missing_lookback_refused(artifact, synthetic_frame):
    fc = make(artifact, synthetic_frame)
    assert refusal(fc, utc("2024-10-28")) == "data_unavailable"
    gappy = synthetic_frame.copy()
    gappy.loc[gappy["timestamp"] == utc("2024-10-09 12:00"), "load_mw"] = np.nan
    assert refusal(make(artifact, gappy), utc("2024-10-10")) == "data_unavailable"


def test_missing_target_weather_refused(artifact, synthetic_frame):
    holes = synthetic_frame.copy()
    holes.loc[holes["timestamp"] == utc("2024-10-10 15:00"), "temperature_2m"] = np.nan
    assert refusal(make(artifact, holes), utc("2024-10-10")) == "weather_unavailable"


def test_future_load_cannot_change_forecast(artifact, synthetic_frame):
    """Serving canary: load at or after the origin must not reach the forecast."""
    origin = utc("2024-10-10")
    corrupted = synthetic_frame.copy()
    late = np.asarray(pd.DatetimeIndex(corrupted["timestamp"]) >= origin)
    corrupted.loc[late, ["load_mw", "da_forecast_mw"]] *= 3
    a = make(artifact, synthetic_frame).forecast(origin)
    b = make(artifact, corrupted).forecast(origin)
    for n in NAMES:
        np.testing.assert_array_equal(a[f"{n}_cal"], b[f"{n}_cal"])


def test_modified_model_file_rejected(artifact_dir, tmp_path):
    copy = tmp_path / "artifact"
    shutil.copytree(artifact_dir, copy)
    with open(copy / "q50.txt", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ArtifactError, match="sha256"):
        load_artifact(copy)


def test_servable_origins(artifact, synthetic_frame):
    ex = Exclusion(utc("2024-10-03"), utc("2024-10-04"), "test", "test")
    servable = make(artifact, synthetic_frame, exclusions=[ex]).servable_origins()
    # Oct 1-3: lookback clean (ends before the exclusion starts);
    # Oct 4-17: lookback overlaps it; Oct 18-26: clean again
    expected = pd.date_range("2024-10-01", "2024-10-03", freq="1D", tz="UTC").append(
        pd.date_range("2024-10-18", "2024-10-26", freq="1D", tz="UTC")
    )
    assert list(servable) == list(expected)
