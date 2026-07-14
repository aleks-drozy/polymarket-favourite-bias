"""Pure scoring functions over per-market P&L series (mirrors sibling repo mc/metrics.py)."""
from __future__ import annotations
from typing import Sequence


def total_pnl(pnls: Sequence[float]) -> float:
    return float(sum(pnls))


def roi(pnls: Sequence[float], stakes: Sequence[float]) -> float:
    staked = float(sum(stakes))
    return total_pnl(pnls) / staked if staked > 0 else 0.0


def win_rate(pnls: Sequence[float]) -> float:
    if len(pnls) == 0:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls)


def max_drawdown(pnls: Sequence[float]) -> float:
    peak, worst, running = 0.0, 0.0, 0.0
    for p in pnls:
        running += p
        peak = max(peak, running)
        worst = max(worst, peak - running)
    return worst
