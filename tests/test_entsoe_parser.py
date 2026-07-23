"""Unit tests for the ENTSO-E XML parser (no network)."""

import pandas as pd
import pytest
from gridcast.data.base import NoDataError
from gridcast.data.entsoe import _parse_load_xml

# 2025-03-30 is the NL spring-forward day. In UTC: nothing special happens.
GL_DOC = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2025-03-30T00:00Z</start>
        <end>2025-03-30T01:00Z</end>
      </timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>11000.25</quantity></Point>
      <Point><position>2</position><quantity>10950.00</quantity></Point>
      <Point><position>3</position><quantity>10900.75</quantity></Point>
      <Point><position>4</position><quantity>10880.00</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>
"""

ACK_DOC = """<?xml version="1.0" encoding="UTF-8"?>
<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:8:1">
  <Reason>
    <code>999</code>
    <text>No matching data found</text>
  </Reason>
</Acknowledgement_MarketDocument>
"""


def test_parses_points_with_utc_timestamps():
    df = _parse_load_xml(GL_DOC)
    assert len(df) == 4
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["timestamp"].iloc[0] == pd.Timestamp("2025-03-30T00:00Z")
    assert df["timestamp"].iloc[3] == pd.Timestamp("2025-03-30T00:45Z")
    assert df["load_mw"].iloc[1] == 10950.0


def test_acknowledgement_raises_no_data():
    with pytest.raises(NoDataError):
        _parse_load_xml(ACK_DOC)
