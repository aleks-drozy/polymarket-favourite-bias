"""Deterministic snapshot rule (spec §3): latest candle >=24h before resolution, <=48h."""
from __future__ import annotations
from data.schema import Candle

H24 = 24 * 3600
H48 = 48 * 3600


def select_snapshot(candles: list[Candle], resolved_ts: int) -> Candle | None:
    lo, hi = resolved_ts - H48, resolved_ts - H24
    eligible = [c for c in candles if lo <= c.t <= hi]
    return max(eligible, key=lambda c: c.t) if eligible else None
