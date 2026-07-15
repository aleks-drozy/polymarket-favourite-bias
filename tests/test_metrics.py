import pytest
from mc.metrics import roi, total_pnl, win_rate, max_drawdown


def test_roi_hand_computed():
    assert roi([0.25, -1.0, 0.5], [1.0, 1.0, 1.0]) == pytest.approx(-0.25 / 3.0)


def test_roi_empty_is_zero():
    assert roi([], []) == 0.0


def test_total_and_winrate():
    assert total_pnl([1.0, -0.5]) == pytest.approx(0.5)
    assert win_rate([1.0, -0.5, 2.0, -0.1]) == 0.5
    assert win_rate([]) == 0.0


def test_max_drawdown():
    # equity: 1, 0.2, 1.2, 0.1 -> worst drop 1.2 - 0.1 = 1.1
    assert max_drawdown([1.0, -0.8, 1.0, -1.1]) == pytest.approx(1.1)
    assert max_drawdown([0.5, 0.5]) == 0.0
