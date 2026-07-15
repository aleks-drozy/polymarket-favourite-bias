"""Per-market flat-stake favourite bet simulation (spec §5). Nothing silently dropped."""
from __future__ import annotations
from pydantic import BaseModel
from data.schema import MarketRecord, Candle, Exclusion
from backtest.favourite import label_favourite
from backtest.snapshot import select_snapshot
from backtest.fees import taker_fee


class BetResult(BaseModel):
    market_id: str
    category: str
    side: str
    entry_price: float
    is_coinflip: bool
    resolved_ts: int
    won: bool
    pnl: float
    stake: float = 1.0


def simulate_market(record: MarketRecord, candles: list[Candle],
                    stake: float = 1.0) -> BetResult | Exclusion:
    snap = select_snapshot(candles, record.resolved_ts)
    if snap is None:
        return Exclusion(market_id=record.market_id, reason="no_candle_in_window")
    side, entry_price, is_coinflip = label_favourite(snap.price_yes)
    shares = stake / entry_price
    fee = taker_fee(shares, entry_price, record.category)
    won = side == record.resolved_outcome
    pnl = (shares - stake - fee) if won else (-stake - fee)
    return BetResult(market_id=record.market_id, category=record.category, side=side,
                     entry_price=entry_price, is_coinflip=is_coinflip,
                     resolved_ts=record.resolved_ts, won=won, pnl=pnl, stake=stake)


def run_backtest(records_with_candles: list[tuple[MarketRecord, list[Candle]]]
                 ) -> tuple[list[BetResult], list[Exclusion]]:
    results, exclusions = [], []
    for record, candles in records_with_candles:
        r = simulate_market(record, candles)
        (exclusions if isinstance(r, Exclusion) else results).append(r)
    return results, exclusions
