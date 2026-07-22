"""Fetch Dutch electricity load data from the ENTSO-E Transparency Platform."""

import os
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

# ENTSO-E area code for the Netherlands (bidding zone)
AREA_CODE_NL = "10YNL----------L"

# ENTSO-E Transparency Platform REST API base URL
BASE_URL = "https://web-api.tp.entsoe.eu/api"


def fetch_load(start: datetime, end: datetime) -> str:
    """Fetch actual total load from ENTSO-E for a given time range.

    Parameters
    ----------
    start : datetime
        Start of the period (UTC).
    end : datetime
        End of the period (UTC).

    Returns
    -------
    str
        Raw XML response from the API.

    Raises
    ------
    ValueError
        If the API key is not set.
    requests.HTTPError
        If the API returns a non-200 status code.
    """
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError(
            "ENTSOE_API_KEY not found. "
            "Copy .env.example to .env and add your key from transparency.entsoe.eu"
        )

    params = {
        "securityToken": api_key,
        "documentType": "A65",  # System total load
        "processType": "A16",  # Realised
        "outBiddingZone_Domain": AREA_CODE_NL,
        "periodStart": start.strftime("%Y%m%d%H00"),
        "periodEnd": end.strftime("%Y%m%d%H00"),
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.text


if __name__ == "__main__":
    # Quick smoke test: fetch yesterday's load data
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=1)

    print(f"Fetching Dutch load data: {start.date()} to {end.date()}")
    xml_data = fetch_load(start, end)

    # Print first 500 chars to verify we got real data
    print(f"\nResponse length: {len(xml_data)} chars")
    print(f"\nFirst 500 chars:\n{xml_data[:500]}")
