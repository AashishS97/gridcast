"""ENTSO-E Transparency Platform client for NL load data.

The API is a single GET endpoint speaking XML (GL_MarketDocument).
API guide: https://transparency.entsoe.eu/ -> Web API documentation.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import pandas as pd
import requests

from gridcast.data.base import DataSourceError, LoadKind, NoDataError

logger = logging.getLogger(__name__)

BASE_URL = "https://web-api.tp.entsoe.eu/api"
NL_BIDDING_ZONE = "10YNL----------L"

_PROCESS_TYPE = {LoadKind.ACTUAL: "A16", LoadKind.DAY_AHEAD: "A01"}

_RESOLUTIONS = {
    "PT15M": pd.Timedelta(minutes=15),
    "PT30M": pd.Timedelta(minutes=30),
    "PT60M": pd.Timedelta(hours=1),
}

_RETRYABLE_STATUS = {500, 502, 503, 504}


def _fmt_utc(dt: datetime) -> str:
    """Format a tz-aware datetime the way ENTSO-E wants it (UTC, yyyyMMddHHmm)."""
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime {dt!r} — GridCast rule: every datetime is tz-aware")
    return dt.astimezone(UTC).strftime("%Y%m%d%H%M")


def _parse_load_xml(xml_text: str) -> pd.DataFrame:
    """Parse a GL_MarketDocument into (timestamp, load_mw) rows.

    Timestamps are reconstructed as period_start + (position - 1) * resolution,
    entirely in UTC — DST does not exist at this layer.
    """
    root = ET.fromstring(xml_text)
    doc_type = root.tag.split("}")[-1]  # strip the xml namespace

    if doc_type == "Acknowledgement_MarketDocument":
        reason = root.findtext(".//{*}Reason/{*}text", default="(no reason given)")
        raise NoDataError(f"ENTSO-E returned no data: {reason}")
    if doc_type != "GL_MarketDocument":
        raise DataSourceError(f"unexpected document type: {doc_type}")

    records: list[dict] = []
    for period in root.iterfind(".//{*}TimeSeries/{*}Period"):
        start_text = period.findtext("./{*}timeInterval/{*}start")
        res_text = period.findtext("./{*}resolution")
        if start_text is None or res_text is None:
            raise DataSourceError("Period missing timeInterval/start or resolution")
        if res_text not in _RESOLUTIONS:
            raise DataSourceError(f"unsupported resolution {res_text!r}")
        period_start = pd.Timestamp(start_text)  # '...T23:00Z' -> tz-aware UTC
        step = _RESOLUTIONS[res_text]
        for point in period.iterfind("./{*}Point"):
            position = int(point.findtext("./{*}position"))
            quantity = float(point.findtext("./{*}quantity"))
            records.append(
                {
                    "timestamp": period_start + (position - 1) * step,
                    "load_mw": quantity,
                }
            )

    if not records:
        raise NoDataError("document parsed but contained no data points")

    df = pd.DataFrame.from_records(records)
    # A document can contain multiple TimeSeries covering overlapping windows
    # (revisions); keep the last occurrence per timestamp.
    return (
        df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    )


class EntsoeClient:
    """Thin ENTSO-E client. Implements the LoadSource protocol."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        max_retries: int = 5,
        timeout_s: float = 30.0,
        rate_limit_sleep_s: float = 65.0,
    ) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.rate_limit_sleep_s = rate_limit_sleep_s

    def fetch_load(self, kind: LoadKind, start: datetime, end: datetime) -> pd.DataFrame:
        params = {
            "documentType": "A65",  # system total load
            "processType": _PROCESS_TYPE[kind],
            "outBiddingZone_Domain": NL_BIDDING_ZONE,
            "periodStart": _fmt_utc(start),
            "periodEnd": _fmt_utc(end),
        }
        logger.info(
            "fetching %s load %s -> %s",
            kind.value,
            params["periodStart"],
            params["periodEnd"],
        )
        return _parse_load_xml(self._request(params))

    def _request(self, params: dict[str, str]) -> str:
        full_params = {**params, "securityToken": self.api_key}
        backoff_s = 2.0
        last_error = "no attempts made"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(BASE_URL, params=full_params, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"network error: {exc}"
                logger.warning("attempt %d/%d failed: %s", attempt, self.max_retries, last_error)
                time.sleep(backoff_s)
                backoff_s *= 2
                continue

            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 401:
                raise DataSourceError("ENTSO-E rejected the API key (401) — check ENTSOE_API_KEY")
            if resp.status_code == 429:
                logger.warning(
                    "rate limited (429); sleeping %.0fs — ENTSO-E temp-bans pushy clients",
                    self.rate_limit_sleep_s,
                )
                time.sleep(self.rate_limit_sleep_s)
                continue
            if resp.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {resp.status_code}"
                logger.warning("attempt %d/%d failed: %s", attempt, self.max_retries, last_error)
                time.sleep(backoff_s)
                backoff_s *= 2
                continue

            # Anything else (e.g. 400 bad params): retrying will not help.
            raise DataSourceError(f"ENTSO-E HTTP {resp.status_code}: {resp.text[:300]}")

        raise DataSourceError(f"giving up after {self.max_retries} attempts ({last_error})")
