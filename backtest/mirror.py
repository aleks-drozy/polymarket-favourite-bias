"""Mirror a favourite-side BetResult onto the underdog (complement) side.

Pure mirror of the sample used for the favourite-bias study: same market,
same snapshot price, same $1 flat stake, same fee model -- side flips to
whichever side was NOT the favourite. P&L reuses mc.reshuffle._pnl_for_side
(pick_favourite_side=False) directly rather than re-deriving the payout
formula here, so the two stay in lockstep by construction.
"""
from __future__ import annotations
from backtest.engine import BetResult
from mc.reshuffle import _pnl_for_side

_FLIP = {"YES": "NO", "NO": "YES"}


def mirror_to_underdog(bet: BetResult) -> BetResult:
    return BetResult(
        market_id=bet.market_id,
        category=bet.category,
        side=_FLIP[bet.side],
        entry_price=1.0 - bet.entry_price,
        is_coinflip=bet.is_coinflip,
        resolved_ts=bet.resolved_ts,
        won=not bet.won,
        pnl=_pnl_for_side(bet, pick_favourite_side=False, stake=bet.stake),
        stake=bet.stake,
    )
