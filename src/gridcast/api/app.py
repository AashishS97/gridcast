"""HTTP API for calibrated day-ahead load forecasts.

Thin by design: every decision lives in gridcast.serving.forecaster (and is
covered by its parity test); this module only validates input, maps refusals
to HTTP and shapes JSON.

  GET /health    liveness + what is being served (artifact, data vintage,
                 provisional boundary, servable range)
  GET /model     the artifact manifest (hashes, versions, git commit, shifts)
  GET /origins   origins that pass every serving gate
  GET /forecast  ?origin=2026-10-05T00:00:00Z -> 24 hourly p10/p50/p90 (MW,
                 80% interval, conformally calibrated) with trust flags

Two kinds of 422: malformed input gets FastAPI's validation error; a
well-formed request the model cannot honestly serve gets {"code", "detail"}
with a stable gate code (before_cutoff, provisional_data, ...).

The forecaster loads once at startup; a missing or modified artifact stops
the process from starting rather than failing every request later.

Run:  uv run uvicorn gridcast.api.app:app --port 8000   (docs at /docs)
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import AwareDatetime, BaseModel

from gridcast.serving.forecaster import Forecaster, NotServable, from_config

TZ = "Europe/Amsterdam"


class HourForecast(BaseModel):
    timestamp_utc: datetime
    timestamp_local: str
    horizon: int
    p10: float
    p50: float
    p90: float
    trust_low: bool
    trust_reasons: list[str]


class ForecastResponse(BaseModel):
    artifact: str
    origin: datetime
    data_as_of: datetime
    units: str = "MW"
    interval: str = "80% prediction interval (p10-p90), conformally calibrated"
    trust_low_hours: int
    hours: list[HourForecast]


class Refusal(BaseModel):
    code: str
    detail: str


def create_app(factory: Callable[[], Forecaster] = from_config) -> FastAPI:
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.forecaster = factory()  # fail fast: a bad artifact stops startup
        yield

    app = FastAPI(
        title="GridCast",
        version="0.1.0",
        description="Calibrated day-ahead forecasts of Dutch hourly electricity load.",
        lifespan=lifespan,
    )

    def get_forecaster(request: Request) -> Forecaster:
        return request.app.state.forecaster

    @app.exception_handler(NotServable)
    async def refused(request: Request, exc: NotServable) -> JSONResponse:
        return JSONResponse(status_code=422, content={"code": exc.code, "detail": exc.message})

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health(request: Request) -> dict:
        return {"status": "ok", **get_forecaster(request).status()}

    @app.get("/model")
    def model(request: Request) -> dict:
        return get_forecaster(request).artifact.manifest

    @app.get("/origins")
    def origins(request: Request) -> dict:
        servable = get_forecaster(request).servable_origins()
        return {"origins": [o.isoformat() for o in servable]}

    @app.get("/forecast", responses={422: {"model": Refusal}})
    def forecast(
        request: Request,
        origin: Annotated[
            AwareDatetime,
            Query(description="Forecast origin, midnight UTC, e.g. 2026-10-05T00:00:00Z"),
        ],
    ) -> ForecastResponse:
        fc = get_forecaster(request)
        o = pd.Timestamp(origin).tz_convert("UTC")
        with lock:
            df = fc.forecast(o)
        hours = [
            HourForecast(
                timestamp_utc=row.timestamp.to_pydatetime(),
                timestamp_local=row.timestamp.tz_convert(TZ).isoformat(),
                horizon=int(row.horizon),
                p10=float(row.q10_cal),
                p50=float(row.q50_cal),
                p90=float(row.q90_cal),
                trust_low=bool(row.trust_low),
                trust_reasons=[r for r in str(row.trust_reasons).split(",") if r],
            )
            for row in df.itertuples(index=False)
        ]
        return ForecastResponse(
            artifact=fc.artifact.name,
            origin=o.to_pydatetime(),
            data_as_of=fc.data_as_of.to_pydatetime(),
            trust_low_hours=sum(h.trust_low for h in hours),
            hours=hours,
        )

    return app


app = create_app()
