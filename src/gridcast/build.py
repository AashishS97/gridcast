"""Reproducible build: raw -> clean -> features -> parquet, one command.

    uv run python -m gridcast.build --years 3

Idempotent: immutable ENTSO-E months and cached weather cities are
skipped, so re-runs only refresh recent data. --offline rebuilds
processed outputs from the existing raw cache without any network.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from gridcast.data import weather
from gridcast.data.base import LoadKind
from gridcast.data.clean import PROCESSED_DIR, clean_series
from gridcast.data.entsoe import EntsoeClient
from gridcast.data.fetch import fetch_and_cache, load_raw
from gridcast.features.build import build_feature_table

logger = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument(
        "--offline", action="store_true", help="rebuild from raw cache, no network calls"
    )
    args = parser.parse_args()

    if not args.offline:
        load_dotenv()
        api_key = os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            raise SystemExit("ENTSOE_API_KEY not set — put it in .env")
        client = EntsoeClient(api_key=api_key)
        for kind in LoadKind:
            fetch_and_cache(client, kind, n_years=args.years)

        end_date = (
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=weather.ARCHIVE_LAG_DAYS)
        ).strftime("%Y-%m-%d")
        weather.fetch_all("2023-07-01", end_date)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    hourly = {}
    for kind in LoadKind:
        hourly[kind] = clean_series(load_raw(kind), kind)
        hourly[kind].to_parquet(PROCESSED_DIR / f"{kind.value}_hourly.parquet", index=False)

    frames = {name: pd.read_parquet(weather.RAW_DIR / f"{name}.parquet") for name in weather.CITIES}
    national = weather.build_national(frames)
    national.to_parquet(weather.PROCESSED_PATH, index=False)

    features = build_feature_table(hourly[LoadKind.ACTUAL], hourly[LoadKind.DAY_AHEAD], national)
    features.to_parquet(FEATURES_PATH, index=False)
    logger.info(
        "wrote %s: %d rows x %d cols, %s -> %s",
        FEATURES_PATH,
        len(features),
        features.shape[1],
        features["timestamp"].min(),
        features["timestamp"].max(),
    )


if __name__ == "__main__":
    main()
