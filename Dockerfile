# syntax=docker/dockerfile:1
# GridCast serving image: code + one frozen artifact + one data vintage.
# The image is the unit of reproducibility; /health reports the artifact, the
# data vintage, and whether they match (data_matches_artifact).
#
#   docker build -t gridcast:local .
#   docker run --rm -p 8000:8000 gridcast:local      -> http://localhost:8000/docs

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# ---- builder: the locked environment, resolved by uv ----------------------
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# dependencies first: this layer is reused until pyproject.toml/uv.lock change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
# then the project, installed editable: PROJECT_ROOT = parents[3] of a module
# file must resolve to /app, so the source has to live at /app/src
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime: no uv, no build cache, non-root ----------------------------
FROM ${PYTHON_IMAGE} AS runtime
# LightGBM needs the OpenMP runtime, which slim images do not ship
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 gridcast
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY configs ./configs
COPY models ./models
COPY data/processed/features.parquet ./data/processed/features.parquet
USER gridcast
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --start-interval=2s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]
CMD ["uvicorn", "gridcast.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
