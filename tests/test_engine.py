import pytest
from data.schema import MarketRecord, Candle, Exclusion
from backtest.engine import simulate_market, run_backtest

RES = 1_700_000_000
IN_WINDOW = RES - 30 * 3600


def rec(outcome="YES", category="politics"):
    return MarketRecord(market_id="0xabc", category=category, created_ts=RES - 90 * 24 * 3600,
                        resolved_ts=RES, resolved_outcome=outcome, volume=50000.0)


def test_winning_favourite_pnl():
    r = simulate_market(rec("YES"), [Candle(t=IN_WINDOW, price_yes=0.8)])
    assert r.won and r.side == "YES"
    assert r.pnl == pytest.approx(0.242)


def test_losing_favourite_pnl():
    r = simulate_market(rec("NO"), [Candle(t=IN_WINDOW, price_yes=0.8)])
    assert not r.won
    assert r.pnl == pytest.approx(-1.008)


def test_no_favourite_side_wins_when_no_resolves():
    r = simulate_market(rec("NO"), [Candle(t=IN_WINDOW, price_yes=0.3)])
    assert r.side == "NO" and r.won


def test_no_candle_returns_exclusion():
    r = simulate_market(rec(), [Candle(t=RES - 3600, price_yes=0.8)])
    assert isinstance(r, Exclusion) and r.reason == "no_candle_in_window"


def test_run_backtest_partitions_results_and_exclusions():
    pairs = [(rec(), [Candle(t=IN_WINDOW, price_yes=0.8)]),
             (rec(), [])]
    results, exclusions = run_backtest(pairs)
    assert len(results) == 1 and len(exclusions) == 1
