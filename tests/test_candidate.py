"""Challenger checks on a copy of the synthetic artifact: a sound candidate
passes every check, each defect fails exactly its own check, the export
refuses a cutoff whose training data would be provisional, and the bundle
has the layout the workflows unpack."""

import json
import shutil
import tarfile

import pandas as pd
import pytest

from gridcast.models.candidate import (
    CandidateConfig,
    build_bundle,
    bundle_name,
    release_tag,
    validate,
)
from gridcast.models.export_artifact import check_final, parse_cutoff
from gridcast.models.metrics import DEFAULT_QUANTILES
from gridcast.models.quantile_lgbm import quantile_params
from gridcast.serving.forecaster import Forecaster, file_sha256, load_artifact

CFG = CandidateConfig(pool_coverage_min=0.5, pool_coverage_max=0.9, pool_mae_max_ratio=1.3)
GATE = {"metrics": {"mae_q50": 256.9}}


@pytest.fixture
def candidate(synthetic_artifact_dir, synthetic_frame, tmp_path):
    """A complete candidate: manifest as the new export writes it, the
    feature table it was 'exported from', and its reference forecast."""
    art = tmp_path / "lgbm_quantile_2024-10-01"
    shutil.copytree(synthetic_artifact_dir, art)
    features = tmp_path / "features.parquet"
    synthetic_frame.to_parquet(features, index=False)
    m = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    m["inputs"]["features_sha256"] = file_sha256(features)
    m["training"] = {"params": {n: quantile_params(t) for n, t in DEFAULT_QUANTILES.items()}}
    m["calibration"].update({"pool_raw_interval_coverage": 0.64, "pool_q50_mae_mw": 250.0})
    (art / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    fc = Forecaster(load_artifact(art), synthetic_frame, [], 0, file_sha256(features))
    fc.forecast(pd.Timestamp("2024-10-01", tz="UTC")).to_parquet(
        art / "reference_forecast.parquet", index=False
    )
    return art, features


def run(art, features):
    return {c: (ok, d) for c, ok, d in validate(art, features, CFG, GATE, 0)}


def edit_manifest(art, fn):
    m = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    fn(m)
    (art / "manifest.json").write_text(json.dumps(m), encoding="utf-8")


def test_sound_candidate_passes_everything(candidate):
    r = run(*candidate)
    assert set(r) == {"integrity", "data", "parity", "objective", "shifts", "pool"}
    assert all(ok for ok, _ in r.values()), r


def test_tampered_model_fails_integrity(candidate):
    art, features = candidate
    with open(art / "q50.txt", "a", encoding="utf-8") as f:
        f.write("\n")
    r = run(art, features)
    assert list(r) == ["integrity"] and not r["integrity"][0]


def test_other_data_fails_data_and_skips_parity(candidate, synthetic_frame, tmp_path):
    art, _ = candidate
    other = tmp_path / "other.parquet"
    synthetic_frame.iloc[:-24].to_parquet(other, index=False)
    r = run(art, other)
    assert not r["data"][0] and not r["parity"][0] and "skipped" in r["parity"][1]


def test_wrong_objective_fails(candidate):
    art, features = candidate
    edit_manifest(art, lambda m: m["training"].update({"params": {"q10": {"objective": "l1"}}}))
    assert not run(art, features)["objective"][0]


def test_non_increasing_shifts_fail(candidate):
    art, features = candidate
    edit_manifest(art, lambda m: m["calibration"]["shifts_mw"].update({"q50": -60.0}))
    r = run(art, features)
    assert not r["shifts"][0]


def test_pool_out_of_bounds_fails(candidate):
    art, features = candidate
    edit_manifest(art, lambda m: m["calibration"].update({"pool_raw_interval_coverage": 0.3}))
    assert not run(art, features)["pool"][0]


def test_bundle_layout(candidate, tmp_path):
    art, features = candidate
    path = build_bundle(art, features, tmp_path / "dist" / "b.tar.gz")
    with tarfile.open(path) as tar:
        names = tar.getnames()
    assert "data/processed/features.parquet" in names
    assert f"models/{art.name}/manifest.json" in names
    assert f"models/{art.name}/q50.txt" in names


def test_names_and_cutoff_parsing():
    c = parse_cutoff("2026-10-22")
    assert c == pd.Timestamp("2026-10-22", tz="UTC")
    assert release_tag(c) == "serving-2026-10-22"
    assert bundle_name(c) == "serving_bundle_2026-10-22.tar.gz"
    with pytest.raises(ValueError):
        parse_cutoff("2026-10-22T06:00:00Z")


def test_export_refuses_a_provisional_cutoff():
    load_ts = pd.date_range("2026-08-01", "2026-11-20 06:00", freq="h", tz="UTC")
    check_final(pd.Timestamp("2026-10-22", tz="UTC"), load_ts, 29)  # line 10-22 06:00
    with pytest.raises(SystemExit, match="provisional line"):
        check_final(pd.Timestamp("2026-10-23", tz="UTC"), load_ts, 29)
