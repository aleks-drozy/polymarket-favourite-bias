from data.schema import Candle
from backtest.snapshot import select_snapshot

H = 3600
RES = 1_700_000_000


def c(hours_before: float, p: float = 0.6) -> Candle:
    return Candle(t=int(RES - hours_before * H), price_yes=p)


def test_picks_latest_candle_in_window():
    candles = [c(47), c(36, 0.65), c(25, 0.7), c(23)]  # 23h is inside excluded final day
    got = select_snapshot(candles, RES)
    assert got is not None and got.price_yes == 0.7  # 25h wins: latest >= 24h


def test_exactly_24h_and_48h_are_inclusive():
    assert select_snapshot([c(24, 0.55)], RES).price_yes == 0.55
    assert select_snapshot([c(48, 0.58)], RES).price_yes == 0.58


def test_none_when_no_candle_qualifies():
    assert select_snapshot([c(60), c(12)], RES) is None
    assert select_snapshot([], RES) is None
