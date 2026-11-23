"""Retry behaviour of the ENTSO-E client against a fake session: transient
gateway failures (including ENTSO-E's non-standard 599) are retried, a bad
request is not, and retries are bounded."""

import pytest

from gridcast.data import entsoe
from gridcast.data.base import DataSourceError
from gridcast.data.entsoe import EntsoeClient


class FakeResponse:
    def __init__(self, status: int, text: str = "") -> None:
        self.status_code = status
        self.text = text


class FakeSession:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = list(statuses)
        self.calls = 0

    def get(self, url, params, timeout):
        self.calls += 1
        return FakeResponse(self.statuses.pop(0), "<xml/>")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(entsoe.time, "sleep", lambda s: None)


@pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
def test_transient_status_is_retried(status):
    session = FakeSession([status, status, 200])
    client = EntsoeClient("key", session=session)
    assert client._request({}) == "<xml/>"
    assert session.calls == 3


def test_bad_request_is_not_retried():
    session = FakeSession([400])
    with pytest.raises(DataSourceError, match="HTTP 400"):
        EntsoeClient("key", session=session)._request({})
    assert session.calls == 1


def test_retries_are_bounded():
    session = FakeSession([599] * 10)
    with pytest.raises(DataSourceError, match="giving up after 5 attempts"):
        EntsoeClient("key", session=session, max_retries=5)._request({})
    assert session.calls == 5
