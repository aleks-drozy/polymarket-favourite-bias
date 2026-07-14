"""Bulk metadata ingestion. Primary: Gamma pagination. Secondary: CSV loader for a
Gamma-shaped export (NOT the third-party dataset -- see below).

Branch taken: gamma, because the free tier of the third-party dataset lacks genuine
resolution outcomes.

Evidence (see .superpowers/sdd/task-11-report.md for the full writeup):
manja316/polymarket-historical-data (github.com/manja316/polymarket-historical-data,
shallow-cloned into dataset_tmp/ for inspection only, never committed) ships
markets.csv with 9,550 rows, 7,552 of them "closed" (active=0). ZERO of those 7,552
have an exact {0.0, 1.0} outcome_prices pair -- the maximum observed certainty is
[0.9995, 0.0005], which matches the row's own best_bid=0.999/best_ask=1.000/
spread=0.001, i.e. a last-polled order-book quote, not an on-chain settlement price.
There is also no dedicated resolved-timestamp field: `end_date` is the *scheduled*
resolution date, and 139 of the 7,552 "closed" rows still have `end_date` in the
future. Given a favourite-bias backtest needs ground-truth resolution, this dataset
is unusable as the primary metadata source, so `load_from_gamma` (already built in
Task 10) is what Task 13's pipeline calls.

`load_dataset_csv` is still implemented per the brief's "implement both" requirement,
but -- since the third-party dataset's own columns were ruled out above -- it targets
a Gamma-shaped CSV export (conditionId/category/createdAt/closedTime/closed/
outcomes/outcomePrices/volumeNum columns, i.e. what you'd get by dumping Gamma API
JSON responses to CSV) rather than the incompatible third-party schema. It shares
its outcome-classification logic with `load_from_gamma` via `_classify_outcome`.
"""
from __future__ import annotations
import json
import pandas as pd
from data.schema import MarketRecord, Exclusion
from data.api_client import PolymarketClient

# A resolved binary market's winning outcomePrices entry is almost never exactly
# "1" in real Gamma data (string/float encoding noise from the AMM/UMA settlement
# pipeline). Live sample of 206 real binary closed markets: 0 hit exactly 1.0, but
# 173 were >=0.999 and the rest of the genuinely-resolved ones ranged down to
# ~0.897. Separately, some closed markets are degenerate/void with prices near
# ["0","0"] or ~0.50/0.50 (never resolved to either side) -- 28 such cases seen in
# the same sample, cleanly separated from the resolved cluster (max gap: 0.50 to
# 0.897). 0.9 sits in that gap, so it's used as the "this side won" threshold.
RESOLVED_PRICE_THRESHOLD = 0.9

COLUMN_MAP = {
    # ours -> Gamma field/CSV-export column names. (Not the third-party dataset's
    # columns -- see module docstring for why.)
    "market_id": "conditionId",
    "category": "category",
    "created_ts": "createdAt",
    "resolved_ts": "closedTime",
    "closed": "closed",
    "outcomes": "outcomes",
    "outcome_prices": "outcomePrices",
    "volume": "volumeNum",
}


def _classify_outcome(outcomes: list, prices: list) -> tuple[str | None, str | None]:
    """Returns (resolved_outcome, exclusion_reason) -- exactly one is None."""
    if [str(o).strip().lower() for o in outcomes] != ["yes", "no"]:
        return None, "not_binary"
    if len(prices) != 2:
        return None, "bad_outcome"
    p0, p1 = float(prices[0]), float(prices[1])
    if p0 >= RESOLVED_PRICE_THRESHOLD and p1 < RESOLVED_PRICE_THRESHOLD:
        return "YES", None
    if p1 >= RESOLVED_PRICE_THRESHOLD and p0 < RESOLVED_PRICE_THRESHOLD:
        return "NO", None
    return None, "bad_outcome"


def _row_to_record(row: dict) -> MarketRecord | Exclusion:
    mid = str(row.get(COLUMN_MAP["market_id"], "unknown"))
    for ours, theirs in COLUMN_MAP.items():
        if theirs not in row or pd.isna(row[theirs]):
            return Exclusion(market_id=mid, reason=f"missing_field:{ours}")
    closed = row[COLUMN_MAP["closed"]]
    if closed in (False, 0, "False", "false", "0"):
        return Exclusion(market_id=mid, reason="not_resolved")
    try:
        outcomes = json.loads(row[COLUMN_MAP["outcomes"]])
        prices = json.loads(row[COLUMN_MAP["outcome_prices"]])
        outcome, reason = _classify_outcome(outcomes, prices)
    except (TypeError, ValueError, KeyError, AttributeError):
        return Exclusion(market_id=mid, reason="bad_outcome")
    if outcome is None:
        return Exclusion(market_id=mid, reason=reason)
    try:
        return MarketRecord(
            market_id=mid,
            category=str(row[COLUMN_MAP["category"]]).lower(),
            created_ts=int(pd.Timestamp(row[COLUMN_MAP["created_ts"]]).timestamp()),
            resolved_ts=int(pd.Timestamp(row[COLUMN_MAP["resolved_ts"]]).timestamp()),
            resolved_outcome=outcome,
            volume=float(row[COLUMN_MAP["volume"]]),
        )
    except (ValueError, TypeError) as exc:
        return Exclusion(market_id=mid, reason=f"missing_field:{exc.__class__.__name__}")


def load_dataset_csv(path: str) -> tuple[list[MarketRecord], list[Exclusion]]:
    df = pd.read_csv(path)
    records, exclusions = [], []
    for row in df.to_dict(orient="records"):
        r = _row_to_record(row)
        (records if isinstance(r, MarketRecord) else exclusions).append(r)
    return records, exclusions


def _gamma_to_record(m: dict) -> MarketRecord | Exclusion:
    mid = str(m.get(COLUMN_MAP["market_id"], "unknown"))
    for ours, theirs in COLUMN_MAP.items():
        if m.get(theirs) in (None, ""):
            return Exclusion(market_id=mid, reason=f"missing_field:{ours}")
    if not m.get(COLUMN_MAP["closed"]):
        return Exclusion(market_id=mid, reason="not_resolved")
    try:
        outcomes = json.loads(m.get(COLUMN_MAP["outcomes"], "[]") or "[]")
        prices = json.loads(m.get(COLUMN_MAP["outcome_prices"], "[]") or "[]")
        outcome, reason = _classify_outcome(outcomes, prices)
    except (TypeError, ValueError, KeyError, AttributeError):
        return Exclusion(market_id=mid, reason="bad_outcome")
    if outcome is None:
        return Exclusion(market_id=mid, reason=reason)
    try:
        return MarketRecord(
            market_id=mid,
            category=str(m.get(COLUMN_MAP["category"], "unknown")).lower(),
            created_ts=int(pd.Timestamp(m[COLUMN_MAP["created_ts"]]).timestamp()),
            resolved_ts=int(pd.Timestamp(m[COLUMN_MAP["resolved_ts"]]).timestamp()),
            resolved_outcome=outcome,
            volume=float(m.get(COLUMN_MAP["volume"], 0.0)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        return Exclusion(market_id=mid, reason=f"missing_field:{exc.__class__.__name__}")


def load_from_gamma(client: PolymarketClient,
                    max_markets: int | None = None) -> tuple[list[MarketRecord], list[Exclusion]]:
    records, exclusions, offset = [], [], 0
    while True:
        page = client.fetch_markets_page(offset=offset)
        if not page:
            break
        for m in page:
            r = _gamma_to_record(m)
            (records if isinstance(r, MarketRecord) else exclusions).append(r)
        offset += len(page)
        if max_markets and len(records) >= max_markets:
            break
    return records, exclusions
