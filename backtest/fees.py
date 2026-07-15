"""Taker fee per docs.polymarket.com/trading/fees.

FORMULA (verified 2026-07-14 via https://docs.polymarket.com/trading/fees, HTTP 200):
    fee = C x feeRate x p x (1 - p)   [C = shares, p = price; matches
    fee = rate * price * (1 - price) * shares used below, multiplication is
    commutative]. Docs confirm: taker-only (makers are never charged fees),
    and the fee amount in USDC is symmetric around 50% probability (a trade
    at 30c incurs the same dollar fee as a trade at 70c).

Rates (verified 2026-07-14, Taker Fee Rate column): crypto 0.07; sports 0.05;
finance 0.04; politics 0.04; economics 0.05; culture 0.05; weather 0.05;
other/general 0.05; mentions 0.04; tech 0.04; geopolitics 0. The live docs
list more categories than this project's research assumption (which only
named crypto/politics/finance/tech/geopolitics) -- all rates for those five
matched exactly, but "sports" (used elsewhere in this project's fixtures)
and several others were only visible on the live page, so CATEGORY_RATES
below was expanded to include them. "_default" is set to 0.05 to match the
docs' "Other / General" catch-all category rather than the 0.04 originally
assumed.

Not modeled here (documented but out of scope for this float-based
backtest): fees are rounded to 5 decimal places on-chain, with the smallest
chargeable fee being 0.00001 USDC (sub-cent trades near price extremes may
round to zero). This has no material effect on backtest-scale trades.

Update this comment + CATEGORY_RATES if the docs change.
"""
from __future__ import annotations

CATEGORY_RATES: dict[str, float] = {
    "crypto": 0.07,
    "sports": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "mentions": 0.04,
    "tech": 0.04,
    "geopolitics": 0.0,
    "_default": 0.05,
}


def taker_fee(shares: float, price: float, category: str) -> float:
    rate = CATEGORY_RATES.get(category.lower(), CATEGORY_RATES["_default"])
    return rate * price * (1.0 - price) * shares
