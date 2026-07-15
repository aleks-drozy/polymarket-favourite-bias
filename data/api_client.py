"""Read-only PolyMarket client: Gamma (metadata) + CLOB (price history).
No auth needed for reads. Disk-cached, throttled (~60 req/min unauthenticated)."""
from __future__ import annotations
import hashlib
import json
import pathlib
import time
import requests
from data.schema import Candle

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# A single transient 5xx/connection blip must not kill an hours-long backtest
# run. Retry those with exponential backoff; a 4xx is a real error (bad
# request/not found) and must surface immediately, not be retried.
MAX_RETRIES = 5
RETRY_BACKOFF_S = [2, 4, 8, 16]


class PolymarketClient:
    def __init__(self, cache_dir: str = "cache", sleep_s: float = 1.1):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_s = sleep_s

    def _get(self, url: str) -> dict | list:
        key = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if key.exists():
            return json.loads(key.read_text())
        last_exc: requests.exceptions.RequestException | None = None
        for attempt in range(MAX_RETRIES):
            time.sleep(self.sleep_s)  # throttle, separate from retry backoff
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
            except requests.exceptions.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise  # real client error -- surface immediately, no retry
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S[attempt])
                continue
            payload = resp.json()
            key.write_text(json.dumps(payload))
            return payload
        raise last_exc

    def get_market(self, condition_id: str) -> dict:
        # NOTE: the live Gamma API defaults `condition_ids` lookups to closed=false
        # (i.e. active markets only) and silently returns [] for any resolved market
        # without an explicit closed=true. This project only ever looks up resolved
        # markets, so closed=true is hardcoded here. See task-10-report.md.
        payload = self._get(f"{GAMMA}/markets?condition_ids={condition_id}&closed=true")
        return payload[0]

    def fetch_markets_page(self, offset: int, limit: int = 100) -> list[dict]:
        return self._get(f"{GAMMA}/markets?closed=true&limit={limit}&offset={offset}")

    def fetch_markets_keyset(self, cursor: str | None = None, limit: int = 100,
                             end_date_min: str | None = None,
                             end_date_max: str | None = None
                             ) -> tuple[list[dict], str | None]:
        # NOTE (Task 13, live-verified): `fetch_markets_page`'s plain offset
        # pagination hard-caps around offset~2000 -- HTTP 422
        # {"error":"offset too large, use /markets/keyset for deeper
        # pagination"} starting somewhere in [2000, 2031] (binary-searched
        # live). That's nowhere near enough historical depth for this study
        # (see data/dataset_loader.py's "Task 13 real-API discovery" note for why deep
        # pagination matters here), so load_from_gamma uses this instead.
        # The request param is `after_cursor` -- found via the live
        # /openapi.json spec; the response's own cursor field is named
        # `next_cursor`, a different name than what you send back, which is
        # easy to get wrong (a bare `next_cursor=...` request param is
        # silently ignored and just re-serves page 1). Server also silently
        # caps `limit` at 100 regardless of the requested value (verified
        # live with limit=500 -> got 100 back), same as fetch_markets_page.
        url = f"{GAMMA}/markets/keyset?closed=true&limit={limit}"
        if cursor:
            url += f"&after_cursor={cursor}"
        if end_date_min:
            url += f"&end_date_min={end_date_min}"
        if end_date_max:
            url += f"&end_date_max={end_date_max}"
        payload = self._get(url)
        return payload.get("markets") or [], payload.get("next_cursor")

    def get_price_history(self, clob_token_id: str, start_ts: int, end_ts: int) -> list[Candle]:
        payload = self._get(
            f"{CLOB}/prices-history?market={clob_token_id}"
            f"&startTs={start_ts}&endTs={end_ts}&fidelity=720")
        return [Candle(t=int(h["t"]), price_yes=float(h["p"]))
                for h in payload.get("history", [])
                if 0.0 < float(h["p"]) < 1.0]
