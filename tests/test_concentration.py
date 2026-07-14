from backtest.engine import BetResult
from mc.concentration import concentration_check, quarter_of


def br(cat, ts, pnl):
    return BetResult(market_id="m", category=cat, side="YES", entry_price=0.8,
                     is_coinflip=False, resolved_ts=ts, won=pnl > 0, pnl=pnl)


Q4_2024 = 1_730_000_000  # 2024-10-27
Q1_2025 = 1_738_000_000  # 2025-01-27


def test_quarter_of():
    assert quarter_of(Q4_2024) == "2024Q4"
    assert quarter_of(Q1_2025) == "2025Q1"


def test_concentrated_when_edge_lives_in_one_category():
    # politics: 30 strong winners; sports: 30 small losers -> full CI positive,
    # excluding politics flips it negative
    results = ([br("politics", Q4_2024, 0.24) for _ in range(30)]
               + [br("sports", Q1_2025, -0.05) for _ in range(30)])
    out = concentration_check(results, n_sims=500, seed=0)
    assert out["concentrated"] is True


def test_not_concentrated_when_edge_is_uniform():
    results = ([br("politics", Q4_2024, 0.2) for _ in range(30)]
               + [br("sports", Q1_2025, 0.2) for _ in range(30)])
    out = concentration_check(results, n_sims=500, seed=0)
    assert out["concentrated"] is False
