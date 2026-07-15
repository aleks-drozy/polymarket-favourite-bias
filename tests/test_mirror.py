import pytest
from backtest.engine import BetResult
from backtest.mirror import mirror_to_underdog
from mc.reshuffle import _pnl_for_side


def bet(side="YES", entry=0.8, won=True, pnl=0.242, cat="politics",
        market_id="m", resolved_ts=1_700_000_000, is_coinflip=False, stake=1.0):
    return BetResult(market_id=market_id, category=cat, side=side, entry_price=entry,
                     is_coinflip=is_coinflip, resolved_ts=resolved_ts, won=won, pnl=pnl,
                     stake=stake)


def test_favourite_won_mirrors_to_underdog_lost():
    # favourite at 0.8 politics won -> underdog at 0.2 lost
    # pnl = -1 - fee(5 shares, 0.2, politics) = -1 - 0.04*0.2*0.8*5.0 = -1.032
    fav = bet(side="YES", entry=0.8, won=True, cat="politics")
    underdog = mirror_to_underdog(fav)
    assert underdog.side == "NO"
    assert underdog.entry_price == pytest.approx(0.2)
    assert underdog.won is False
    assert underdog.pnl == pytest.approx(-1.032)


def test_favourite_lost_mirrors_to_underdog_won():
    # favourite entry 0.7 lost -> underdog at 0.3 won
    # shares = 1/0.3, fee = 0.04*0.3*0.7*shares, pnl = shares - 1 - fee
    fav = bet(side="NO", entry=0.7, won=False, cat="politics")
    underdog = mirror_to_underdog(fav)
    shares = 1.0 / 0.3
    fee = 0.04 * 0.3 * 0.7 * shares
    expected_pnl = shares - 1.0 - fee
    assert underdog.side == "YES"
    assert underdog.entry_price == pytest.approx(0.3)
    assert underdog.won is True
    assert underdog.pnl == pytest.approx(expected_pnl)


def test_side_flips_no_to_yes():
    fav = bet(side="NO", entry=0.65, won=True, cat="sports")
    underdog = mirror_to_underdog(fav)
    assert underdog.side == "YES"
    assert underdog.entry_price == pytest.approx(0.35)


def test_coinflip_preserved():
    fav = bet(side="YES", entry=0.5, won=True, cat="politics", is_coinflip=True)
    underdog = mirror_to_underdog(fav)
    assert underdog.is_coinflip is True

    fav2 = bet(side="YES", entry=0.5, won=True, cat="politics", is_coinflip=False)
    underdog2 = mirror_to_underdog(fav2)
    assert underdog2.is_coinflip is False


def test_market_id_category_resolved_ts_stake_preserved():
    fav = bet(market_id="0xabc123", cat="crypto", resolved_ts=1_650_000_000, stake=1.0)
    underdog = mirror_to_underdog(fav)
    assert underdog.market_id == "0xabc123"
    assert underdog.category == "crypto"
    assert underdog.resolved_ts == 1_650_000_000
    assert underdog.stake == pytest.approx(1.0)


def test_pnl_equals_pnl_for_side_false_across_varied_bets():
    bets = [
        bet(side="YES", entry=0.8, won=True, cat="politics"),
        bet(side="NO", entry=0.7, won=False, cat="sports"),
        bet(side="YES", entry=0.55, won=False, cat="crypto"),
        bet(side="NO", entry=0.999, won=True, cat="geopolitics"),
        bet(side="YES", entry=0.5, won=True, cat="finance", is_coinflip=True),
    ]
    for fav in bets:
        underdog = mirror_to_underdog(fav)
        assert underdog.pnl == pytest.approx(_pnl_for_side(fav, False))
