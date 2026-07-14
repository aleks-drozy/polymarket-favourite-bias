"""Random-side null: does picking the FAVOURITE beat picking sides at random
on the same markets, same prices, same fees? (spec §6)"""
from __future__ import annotations
import numpy as np
from backtest.engine import BetResult
from backtest.fees import taker_fee
from mc.metrics import roi


def _pnl_for_side(r: BetResult, pick_favourite_side: bool, stake: float = 1.0) -> float:
    price = r.entry_price if pick_favourite_side else 1.0 - r.entry_price
    won = r.won if pick_favourite_side else not r.won
    shares = stake / price
    fee = taker_fee(shares, price, r.category)
    return (shares - stake - fee) if won else (-stake - fee)


def random_side_null(results: list[BetResult], n_sims: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    observed = roi([r.pnl for r in results], [r.stake for r in results])
    n = len(results)
    null_rois = np.empty(n_sims)
    for i in range(n_sims):
        picks = rng.random(n) < 0.5  # True = take the favourite side
        pnls = [_pnl_for_side(r, bool(p)) for r, p in zip(results, picks)]
        null_rois[i] = roi(pnls, [1.0] * n)
    p_value = (1.0 + float((null_rois >= observed).sum())) / (n_sims + 1.0)
    return {"null_rois": null_rois, "observed_roi": observed, "p_value": p_value}
