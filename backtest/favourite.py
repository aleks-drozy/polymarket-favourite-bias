"""Favourite = the side with implied probability > 50% at snapshot (spec §3)."""
from __future__ import annotations


def label_favourite(price_yes: float) -> tuple[str, float, bool]:
    if price_yes > 0.5:
        return "YES", price_yes, False
    if price_yes < 0.5:
        return "NO", 1.0 - price_yes, False
    return "YES", 0.5, True  # exact tie: deterministic YES, flagged coinflip
