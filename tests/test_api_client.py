import json
import pathlib
import pytest
from data.api_client import PolymarketClient

FIX = pathlib.Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self._payload


@pytest.fixture
def client(tmp_path, monkeypatch):
    c = PolymarketClient(cache_dir=str(tmp_path), sleep_s=0.0)
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        if "prices-history" in url:
            return FakeResponse(json.loads((FIX / "prices_history.json").read_text()))
        return FakeResponse(json.loads((FIX / "gamma_market.json").read_text()))

    monkeypatch.setattr("data.api_client.requests.get", fake_get)
    c._calls = calls
    return c


def test_price_history_maps_to_candles(client):
    candles = client.get_price_history("111", 1699000000, 1700000000)
    assert len(candles) == 2 and candles[0].price_yes == 0.61


def test_get_market_returns_first(client):
    m = client.get_market("0xabc")
    assert m["conditionId"] == "0xabc"


def test_cache_prevents_second_network_call(client):
    client.get_price_history("111", 1699000000, 1700000000)
    client.get_price_history("111", 1699000000, 1700000000)
    assert len(client._calls) == 1


@pytest.mark.network
def test_live_smoke_known_market():
    # Well-known resolved market: 2024 US Presidential Election winner.
    c = PolymarketClient(cache_dir="cache")
    page = c.fetch_markets_page(offset=0, limit=5)
    assert isinstance(page, list) and len(page) > 0
