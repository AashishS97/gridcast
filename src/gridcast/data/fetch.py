"""Fetch raw NL load from ENTSO-E and cache it as monthly parquet files.

Cache layout: data/raw/entsoe/{kind}/{YYYY-MM}.parquet, at native (15-min)
resolution, exactly as received — raw means raw.

Months ending more than REFRESH_HORIZON_DAYS ago are immutable and skipped
if cached; recent months are always refetched because ENTSO-E revises
recent actuals.

History starts at DATA_START, a fixed date: the recipe trains on an
expanding window from there (configs/evaluation.toml train_start). Until
2026-11-23 the start was "3 years before today", which silently moved: a
challenger built on CI on 2026-11-23 trained from 2023-11-01 while the
champion had trained from 2023-07-01.

Run:  uv run python -m gridcast.data.fetch
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
# Months ending within this many days are always refetched. It must cover the
# vintage ledger's window (gridcast.serving.daily_log.VINTAGE_DAYS = 60): with
# a cached raw store (CI keeps one across runs), a month treated as immutable
# is never asked again, so a revision inside the ledger window would be
# invisible to the late-revision check. 30 left days aged 31-61 unrefetched
# at the start of each month.
REFRESH_HORIZON_DAYS = 62
POLITE_SLEEP_S = 0.5
DATA_START = pd.Timestamp("2023-07-01", tz="UTC")


def month_starts(start: pd.Timestamp, now: pd.Timestamp) -> pd.DatetimeIndex:
    """First-of-month UTC timestamps from start's month up to now."""
    first = start.tz_convert("UTC").normalize().replace(day=1)
    return pd.date_range(first, now, freq="MS")


def fetch_and_cache(source: LoadSource, kind: LoadKind, start: pd.Timestamp = DATA_START) -> None:
    now = pd.Timestamp.now(tz="UTC")
    out_dir = RAW_DIR / kind.value
    out_dir.mkdir(parents=True, exist_ok=True)

    fetched = skipped = 0
    for m_start in month_starts(start, now):
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
    parser.parse_args()

    client = EntsoeClient(api_key=api_key)
    for kind in LoadKind:
        fetch_and_cache(client, kind)

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
