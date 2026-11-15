"""Parity: serving the REAL artifact at its cutoff must reproduce the
reference forecast the export computed. Skips when the artifact or data are
absent (e.g. CI) or features.parquet is a different vintage."""

import numpy as np
import pandas as pd
import pytest

from gridcast.models.export_artifact import load_config
from gridcast.serving.forecaster import from_config


@pytest.fixture(scope="module")
def forecaster():
    cfg = load_config()
    if not (cfg.output_root / cfg.name).exists():
        pytest.skip(f"artifact {cfg.name} not built")
    fc = from_config()
    if not fc.data_matches_artifact:
        pytest.skip("features.parquet differs from the artifact's data vintage")
    return fc


def test_serving_reproduces_reference_forecast(forecaster):
    ref = pd.read_parquet(forecaster.artifact.path / "reference_forecast.parquet")
    got = forecaster.forecast(forecaster.artifact.train_cutoff)
    assert len(got) == len(ref) == 24
    assert list(pd.DatetimeIndex(got["timestamp"])) == list(pd.DatetimeIndex(ref["timestamp"]))
    numeric = [c for c in ref.columns if ref[c].dtype.kind == "f"]
    assert {"q10_cal", "q50_cal", "q90_cal", "temp_pct"} <= set(numeric)
    for col in numeric:
        np.testing.assert_allclose(got[col], ref[col], rtol=0, atol=1e-9, err_msg=col)
    for col in [
        "flag_cold",
        "flag_calendar",
        "flag_instability",
        "flag_integrity",
        "trust_reasons",
    ]:
        assert got[col].tolist() == ref[col].tolist(), col
