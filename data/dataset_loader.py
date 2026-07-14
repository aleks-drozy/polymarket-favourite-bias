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

Resolution-truth rule (tightened in review round 1, see task-11-report.md "Fix
round 1"): Gamma exposes no usable authoritative settlement field for this
study's purposes -- `umaResolutionStatuses` exists but is empty for ~81% of
markets closed more than a year ago (checked live), which is most of the
history this backtest needs, so requiring it would gut the sample far worse
than any price threshold. Instead, `_classify_outcome` requires one side's
outcomePrices to be >=0.999 (RESOLVED_PRICE_THRESHOLD, see its own comment for
the live-sample evidence); anything short of that -- including the degenerate
near-["0","0"]/near-0.5-0.5 void-market cluster -- is excluded as
`bad_outcome`. Identical rule on both `load_from_gamma` and `load_dataset_csv`.
"""
from __future__ import annotations
import json
import pandas as pd
from data.schema import MarketRecord, Exclusion
from data.api_client import PolymarketClient

# --- Task 11 review round 1: threshold tightened 0.9 -> 0.999, evidence below ---
#
# Investigated whether Gamma exposes an authoritative settlement-truth field
# instead of relying on outcomePrices at all. Live `GET /markets?closed=true`
# responses do carry `umaResolutionStatuses` (a JSON-string-encoded history like
# '["proposed","resolved"]' or '["proposed","disputed","proposed"]' -- values
# seen: proposed / disputed / resolved). No `umaResolutionStatus` (singular),
# `resolvedBy`, or `hasResolutionData` field exists anywhere in the ~70-key
# market object (checked via a live full-object dump).
#
# That field is NOT usable as a required gate for this study: sampled 837 real
# closed markets spanning 2020-2026 and bucketed by age at query time --
# markets closed <365 days ago were 100% populated (216/216), but markets
# closed >365 days ago were only 18.9% populated (101/535); the rest returned
# `[]`. A favourite-bias backtest needs multi-year history, and >365-day-old
# markets are the bulk of that history, so requiring umaResolutionStatuses ==
# "resolved" would silently drop ~81% of an otherwise-usable historical sample
# -- far worse than any price-threshold tightening below. (Separately, prices
# already snap to their near-final value the moment a market closes, before
# umaResolutionStatuses ever reaches "resolved" -- e.g. a market closed same-day
# already showed exact outcomePrices=["1","0"] while still `["proposed"]` -- so
# even where present, the field doesn't add confidence beyond what the price
# itself already encodes for markets old enough to matter here.) Conclusion:
# treat the field as not reliably usable for this study and tighten the price
# rule instead, sharing the tightened rule across both loader paths.
#
# A resolved binary market's winning outcomePrices entry is almost never exactly
# "1" in real Gamma data (string/float encoding noise from the AMM/UMA settlement
# pipeline), though markets closed within roughly the last year commonly do show
# a clean "1"/"0". Re-ran the live-sample investigation at n=1,465 binary closed
# markets (offsets 0-2,950, spanning 2020-2026 -- 7x the original n=206 sample):
# 1,433 resolved one-sided under the old >=0.9 rule, of which 1,407 (98.2%) were
# already >=0.999; only 26 (1.8%) sit in [0.9, 0.999) and are newly excluded by
# this tightening (down from the original ~16%-loss estimate, which was based on
# a much smaller and evidently less representative sample). Degenerate/void
# markets (never resolved to either side, e.g. outcomePrices near ["0","0"] or
# ~0.50/0.50) topped out at 0.897 in the same sample -- still a clean gap below
# 0.9, so nothing in that cluster is newly misclassified as resolved by raising
# the bar to 0.999. Net trade-off: ~1.8% additional sample loss for meaningfully
# tighter label purity on the remaining markets, given no authoritative field is
# available to corroborate price instead. See task-11-report.md ("Fix round 1")
# for the raw JSON evidence.
RESOLVED_PRICE_THRESHOLD = 0.999

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
