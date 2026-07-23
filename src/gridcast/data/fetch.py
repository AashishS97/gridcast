"""Fetch raw NL load from ENTSO-E and cache it as monthly parquet files.

Cache layout: data/raw/entsoe/{kind}/{YYYY-MM}.parquet, at native (15-min)
resolution, exactly as received — raw means raw.

Months ending more than REFRESH_HORIZON_DAYS ago are immutable and skipped
if cached; recent months are always refetched because ENTSO-E revises
recent actuals.

Run:  uv run python -m gridcast.data.fetch --years 3
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from gridcast.data.base import LoadKind, LoadSource, NoDataError
from gridcast.data.entsoe import EntsoeClient

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/entsoe")
REFRESH_HORIZON_DAYS = 30
POLITE_SLEEP_S = 0.5


def month_starts(n_years: int, now: pd.Timestamp) -> pd.DatetimeIndex:
    """First-of-month UTC timestamps covering the last n_years up to now."""
    start = (now - pd.DateOffset(years=n_years)).normalize().replace(day=1)
    return pd.date_range(start, now, freq="MS")


def fetch_and_cache(source: LoadSource, kind: LoadKind, n_years: int) -> None:
    now = pd.Timestamp.now(tz="UTC")
    out_dir = RAW_DIR / kind.value
    out_dir.mkdir(parents=True, exist_ok=True)

    fetched = skipped = 0
    for m_start in month_starts(n_years, now):
        m_end = m_start + pd.DateOffset(months=1)
        path = out_dir / f"{m_start:%Y-%m}.parquet"

        immutable = m_end < now - pd.Timedelta(days=REFRESH_HORIZON_DAYS)
        if path.exists() and immutable:
            skipped += 1
            continue

        try:
            df = source.fetch_load(kind, m_start.to_pydatetime(), m_end.to_pydatetime())
        except NoDataError as exc:
            logger.warning("%s %s: %s", kind.value, f"{m_start:%Y-%m}", exc)
            continue

        df.to_parquet(path, index=False)
        fetched += 1
        logger.info("wrote %s (%d rows)", path, len(df))
        time.sleep(POLITE_SLEEP_S)

    logger.info("%s: fetched %d months, skipped %d cached months", kind.value, fetched, skipped)


def load_raw(kind: LoadKind) -> pd.DataFrame:
    """Concatenate all cached months for one series into a single DataFrame."""
    paths = sorted((RAW_DIR / kind.value).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no cached data for {kind.value!r} — run `python -m gridcast.data.fetch` first"
        )
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")  # refetched-month overlaps
        .reset_index(drop=True)
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    load_dotenv()
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise SystemExit("ENTSOE_API_KEY not set — put it in .env (see .env.example)")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=3, help="years of history to fetch")
    args = parser.parse_args()

    client = EntsoeClient(api_key=api_key)
    for kind in LoadKind:
        fetch_and_cache(client, kind, n_years=args.years)

    # Sanity summary — eyeball this before we build quality checks on top.
    for kind in LoadKind:
        df = load_raw(kind)
        logger.info(
            "%s: %d rows, %s -> %s, median step %s",
            kind.value,
            len(df),
            df["timestamp"].min(),
            df["timestamp"].max(),
            df["timestamp"].diff().median(),
        )


if __name__ == "__main__":
    main()
