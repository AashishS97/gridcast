"""HTTP layer on the synthetic artifact: routes, response contract, refusal
mapping, input validation, and fail-fast startup."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridcast.api.app import create_app
from gridcast.serving.forecaster import ArtifactError, Forecaster, load_artifact


@pytest.fixture(scope="module")
def client(synthetic_artifact_dir, synthetic_frame):
    fc = Forecaster(load_artifact(synthetic_artifact_dir), synthetic_frame, [], 0)
    with TestClient(create_app(lambda: fc)) as c:
        yield c, fc


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["artifact"] == "synthetic"
    assert body["n_servable"] == 26


def test_root_redirects_to_docs(client):
    c, _ = client
    r = c.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/docs"


def test_model_returns_manifest(client):
    c, fc = client
    assert c.get("/model").json() == fc.artifact.manifest


def test_origins(client):
    c, _ = client
    origins = c.get("/origins").json()["origins"]
    assert len(origins) == 26
    assert origins[0].startswith("2024-10-01T00:00:00")


def test_forecast_equals_forecaster_output(client):
    c, fc = client
    r = c.get("/forecast", params={"origin": "2024-10-10T00:00:00Z"})
    assert r.status_code == 200
    body = r.json()
    assert body["artifact"] == "synthetic"
    assert body["units"] == "MW"
    hours = body["hours"]
    assert [h["horizon"] for h in hours] == list(range(1, 25))
    ref = fc.forecast(pd.Timestamp("2024-10-10", tz="UTC"))
    for h, row in zip(hours, ref.itertuples(index=False), strict=True):
        assert (h["p10"], h["p50"], h["p90"]) == (row.q10_cal, row.q50_cal, row.q90_cal)
    assert hours[0]["timestamp_local"] == "2024-10-10T02:00:00+02:00"
    assert body["trust_low_hours"] == sum(h["trust_low"] for h in hours)


@pytest.mark.parametrize(
    "origin, code",
    [
        ("2024-09-30T00:00:00Z", "before_cutoff"),
        ("2024-10-10T06:00:00Z", "not_midnight_utc"),
        ("2024-10-28T00:00:00Z", "data_unavailable"),
    ],
)
def test_refusals_are_422_with_code(client, origin, code):
    c, _ = client
    r = c.get("/forecast", params={"origin": origin})
    assert r.status_code == 422
    assert r.json()["code"] == code


@pytest.mark.parametrize("params", [{}, {"origin": "2024-10-10"}, {"origin": "not-a-date"}])
def test_malformed_input_is_validation_error(client, params):
    c, _ = client
    r = c.get("/forecast", params=params)
    assert r.status_code == 422
    assert "code" not in r.json()  # FastAPI validation error, not a gate refusal


def test_broken_artifact_stops_startup():
    def broken() -> Forecaster:
        raise ArtifactError("q50.txt: sha256 mismatch")

    with pytest.raises(ArtifactError), TestClient(create_app(broken)):
        pass
