"""End-to-end backtest pipeline. Full run: python run_backtest.py
Smoke run: python run_backtest.py --max-markets 50"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import pathlib
import time
from data.schema import MarketRecord, Exclusion
from data.api_client import PolymarketClient
from data.crossval import cross_validate
from backtest.engine import run_backtest

FLOORS = [100_000, 50_000, 10_000, 5_000, 1_000, 0]
TARGET_N = 2_000
H = 3600


def calibrate_volume_floor(records: list[MarketRecord]) -> tuple[float, int]:
    for floor in FLOORS:
        kept = sum(1 for r in records if r.volume >= floor)
        if kept >= TARGET_N:
            return float(floor), kept
    return 0.0, len(records)


def _yes_token_id(client, market_id: str) -> str | None:
    try:
        m = client.get_market(market_id)
        tokens = json.loads(m.get("clobTokenIds", "[]") or "[]")
        return tokens[0] if tokens else None
    except Exception:
        return None


def run_pipeline(records: list[MarketRecord], client, results_dir: str = "results",
                 n_crossval: int = 100) -> dict:
    out_dir = pathlib.Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    floor, kept = calibrate_volume_floor(records)
    filtered = [r for r in records if r.volume >= floor]

    # Ledger watch item (Task 9/11 progress notes): track distinct lowercased
    # category strings and their market counts so the writeup can check
    # fee-slug alignment against backtest/fees.py's CATEGORY_RATES keys --
    # any category here that's absent from CATEGORY_RATES silently falls to
    # the _default rate, which is worth knowing about even if it's fine.
    categories: dict[str, int] = collections.Counter(r.category.lower() for r in filtered)

    pairs, exclusions = [], []
    for r in filtered:
        token = _yes_token_id(client, r.market_id)
        if token is None:
            exclusions.append(Exclusion(market_id=r.market_id, reason="no_token_id"))
            continue
        candles = client.get_price_history(token, r.resolved_ts - 48 * H,
                                           r.resolved_ts - 24 * H + H)
        pairs.append((r, candles))

    bets, engine_exclusions = run_backtest(pairs)
    exclusions.extend(engine_exclusions)

    payload = {"config": {"volume_floor": floor, "n_metadata": len(records),
                          "n_after_floor": kept, "n_bets": len(bets),
                          "n_excluded": len(exclusions),
                          "categories": dict(categories),
                          "run_ts": int(time.time())},
               "bets": [b.model_dump() for b in bets]}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))

    with open(out_dir / "exclusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["market_id", "reason"])
        for e in exclusions:
            w.writerow([e.market_id, e.reason])

    report = cross_validate(filtered, client, n_sample=n_crossval)
    (out_dir / "crossval_report.json").write_text(json.dumps(report, indent=1))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-markets", type=int, default=None)
    args = ap.parse_args()
    client = PolymarketClient()
    # Metadata source per Task 11's branch decision: Gamma API is PRIMARY
    # (third-party dataset outcomes are order-book quotes, not settlements).
    from data.dataset_loader import load_from_gamma
    records, load_exclusions = load_from_gamma(client, max_markets=args.max_markets)
    print(f"metadata: {len(records)} records, {len(load_exclusions)} load exclusions")
    payload = run_pipeline(records, client)
    print(json.dumps(payload["config"], indent=2))


if __name__ == "__main__":
    main()
