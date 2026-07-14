"""Dataset-vs-official-API agreement check (spec §4). Agreement % is reported
in the WRITEUP regardless of outcome."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from data.schema import MarketRecord


def _api_outcome(market: dict) -> str | None:
    prices = json.loads(market.get("outcomePrices", "[]") or "[]")
    if len(prices) != 2:
        return None
    return "YES" if float(prices[0]) == 1.0 else "NO"


def cross_validate(records: list[MarketRecord], client, n_sample: int = 100,
                   seed: int = 0, tolerance_ts: int = 6 * 3600) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(n_sample, len(records)), replace=False)
    n_checked = n_outcome = n_ts = n_err = 0
    mismatches = []
    for i in idx:
        r = records[int(i)]
        try:
            m = client.get_market(r.market_id)
        except Exception:
            n_err += 1
            continue
        n_checked += 1
        api_out = _api_outcome(m)
        if api_out == r.resolved_outcome:
            n_outcome += 1
        else:
            mismatches.append({"market_id": r.market_id, "field": "resolved_outcome",
                               "ours": r.resolved_outcome, "theirs": api_out})
        api_ts = int(pd.Timestamp(m["closedTime"]).timestamp()) if m.get("closedTime") else None
        if api_ts is not None and abs(api_ts - r.resolved_ts) <= tolerance_ts:
            n_ts += 1
        else:
            mismatches.append({"market_id": r.market_id, "field": "resolved_ts",
                               "ours": r.resolved_ts, "theirs": api_ts})
    pct = (100.0 * n_outcome / n_checked) if n_checked else 0.0
    return {"n_checked": n_checked, "n_outcome_match": n_outcome, "n_ts_match": n_ts,
            "n_api_errors": n_err, "agreement_pct": pct, "mismatches": mismatches}
