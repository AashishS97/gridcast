"""Open-Meteo historical weather: fetch, cache, national aggregate.

Five cities, population-weighted (approximate, Randstad-heavy) into one
national hourly series. timezone=UTC is requested explicitly: joining
local-time weather onto a UTC load index would misalign by 1-2 hours and
smear exactly the morning-ramp signal temperature is meant to capture.

Archive = ERA5 reanalysis (actual weather), lagging ~5 days. Known
train/serve skew vs production forecasts — documented, accepted for now.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
RAW_DIR = Path("data/raw/weather")
PROCESSED_PATH = Path("data/processed/weather_hourly.parquet")

HOURLY_VARS = ["temperature_2m", "shortwave_radiation", "wind_speed_10m"]

# name: (lat, lon, weight). Weights sum to 1.0.
CITIES: dict[str, tuple[float, float, float]] = {
    "amsterdam": (52.37, 4.89, 0.30),
    "rotterdam": (51.92, 4.48, 0.25),
    "utrecht": (52.09, 5.12, 0.15),
    "eindhoven": (51.44, 5.47, 0.15),
    "groningen": (53.22, 6.57, 0.15),
}

ARCHIVE_LAG_DAYS = 6  # ERA5 trails reality by ~5 days


def fetch_city(name: str, start_date: str, end_date: str) -> pd.DataFrame:
    lat, lon, _ = CITIES[name]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    for attempt in range(3):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
            resp.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == 2:
                raise
            logger.warning("%s: retrying after %s", name, exc)
            time.sleep(2 * (attempt + 1))
    payload = resp.json()["hourly"]
    df = pd.DataFrame(payload)
    # timezone=UTC makes these UTC wall times; utc=True attaches the zone.
    df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True)
    return df[["timestamp", *HOURLY_VARS]]


def fetch_all(start_date: str, end_date: str, force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in CITIES:
        path = RAW_DIR / f"{name}.parquet"
        if path.exists() and not force:
            logger.info("%s cached, skipping (--force to refetch)", name)
            continue
        df = fetch_city(name, start_date, end_date)
        df.to_parquet(path, index=False)
        logger.info("wrote %s (%d rows)", path, len(df))
        time.sleep(0.3)


def build_national(city_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Population-weighted national series + trailing 24h temperature mean."""
    weights = {name: CITIES[name][2] for name in city_frames}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"city weights sum to {total}, not 1.0")

    acc: pd.DataFrame | None = None
    for name, df in city_frames.items():
        contrib = df.set_index("timestamp")[HOURLY_VARS] * weights[name]
        acc = contrib if acc is None else acc.add(contrib, fill_value=None)

    national = acc.sort_index()
    # Trailing window: value at t uses hours (t-23h .. t] only — no future.
    national["temp_mean_24h"] = national["temperature_2m"].rolling(window=24, min_periods=24).mean()
    return national.rename_axis("timestamp").reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-07-01")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    end_date = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=ARCHIVE_LAG_DAYS)).strftime(
        "%Y-%m-%d"
    )
    fetch_all(args.start, end_date, force=args.force)

    frames = {name: pd.read_parquet(RAW_DIR / f"{name}.parquet") for name in CITIES}
    national = build_national(frames)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    national.to_parquet(PROCESSED_PATH, index=False)
    logger.info(
        "wrote %s: %d rows, %s -> %s",
        PROCESSED_PATH,
        len(national),
        national["timestamp"].min(),
        national["timestamp"].max(),
    )
    logger.info(
        "\nsanity — national temperature by month (degC):\n%s",
        national.assign(month=national["timestamp"].dt.month)
        .groupby("month")["temperature_2m"]
        .mean()
        .round(1)
        .to_string(),
    )


if __name__ == "__main__":
    main()
