import json
import pathlib
import pytest
import requests
from data.api_client import PolymarketClient, MAX_RETRIES

FIX = pathlib.Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload, status_code=200, raise_status=False):
        self._payload = payload
        self.status_code = status_code
        self._raise_status = raise_status
    def raise_for_status(self):
        if self._raise_status:
            err = requests.exceptions.HTTPError(f"{self.status_code} Server Error")
            err.response = self
            raise err
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
        if "keyset" in url:
            return FakeResponse(json.loads((FIX / "gamma_market_keyset.json").read_text()))
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


def test_fetch_markets_keyset_returns_markets_and_cursor(client):
    markets, next_cursor = client.fetch_markets_keyset()
    assert len(markets) == 1 and markets[0]["conditionId"] == "0xabc"
    assert next_cursor == "abc123"


def test_fetch_markets_keyset_sends_after_cursor_param(client):
    client.fetch_markets_keyset(cursor="prevcursor")
    assert any("after_cursor=prevcursor" in u for u in client._calls)


def test_cache_prevents_second_network_call(client):
    client.get_price_history("111", 1699000000, 1700000000)
    client.get_price_history("111", 1699000000, 1700000000)
    assert len(client._calls) == 1


def test_get_retries_on_5xx_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("data.api_client.time.sleep", lambda s: None)
    calls = []
    responses = [
        FakeResponse(None, status_code=500, raise_status=True),
        FakeResponse(None, status_code=500, raise_status=True),
        FakeResponse({"ok": True}, status_code=200),
    ]

    def fake_get(url, timeout):
        calls.append(url)
        return responses[len(calls) - 1]

    monkeypatch.setattr("data.api_client.requests.get", fake_get)
    c = PolymarketClient(cache_dir=str(tmp_path), sleep_s=0.0)

    result = c._get("https://gamma-api.polymarket.com/markets/keyset?closed=true&limit=100")

    assert result == {"ok": True}
    assert len(calls) == 3


def test_get_raises_after_max_retries_on_persistent_5xx(tmp_path, monkeypatch):
    monkeypatch.setattr("data.api_client.time.sleep", lambda s: None)
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return FakeResponse(None, status_code=500, raise_status=True)

    monkeypatch.setattr("data.api_client.requests.get", fake_get)
    c = PolymarketClient(cache_dir=str(tmp_path), sleep_s=0.0)

    with pytest.raises(requests.exceptions.HTTPError):
        c._get("https://gamma-api.polymarket.com/markets/keyset?closed=true&limit=100")

    assert len(calls) == MAX_RETRIES


@pytest.mark.network
def test_live_smoke_known_market():
    # Well-known resolved market: 2024 US Presidential Election winner.
    c = PolymarketClient(cache_dir="cache")
    page = c.fetch_markets_page(offset=0, limit=5)
    assert isinstance(page, list) and len(page) > 0


@pytest.mark.network
def test_live_keyset_pagination_advances():
    # Offset pagination hard-caps around offset~2000 (verified live, Task 13)
    # -- this confirms the keyset cursor mechanism actually advances instead
    # of re-serving the same page (an easy mistake: passing back the wrong
    # param name silently no-ops and just returns page 1 again).
    c = PolymarketClient(cache_dir="cache")
    page1, cursor1 = c.fetch_markets_keyset(limit=5)
    assert len(page1) > 0 and cursor1
    page2, _ = c.fetch_markets_keyset(cursor=cursor1, limit=5)
    assert len(page2) > 0
    assert {m["conditionId"] for m in page1}.isdisjoint({m["conditionId"] for m in page2})
