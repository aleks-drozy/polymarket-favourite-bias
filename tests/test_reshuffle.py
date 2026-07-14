import numpy as np
import pytest
from backtest.engine import BetResult
from mc.reshuffle import random_side_null


def br(side="YES", entry=0.8, won=True, pnl=0.242, cat="politics"):
    return BetResult(market_id="m", category=cat, side=side, entry_price=entry,
                     is_coinflip=False, resolved_ts=1_700_000_000, won=won, pnl=pnl)


def test_deterministic_with_seed():
    results = [br(), br(side="NO", entry=0.7, won=False, pnl=-1.0084)]
    a = random_side_null(results, n_sims=300, seed=0)
    b = random_side_null(results, n_sims=300, seed=0)
    assert np.allclose(a["null_rois"], b["null_rois"])
    assert a["null_rois"].shape == (300,)


def test_observed_roi_matches_results():
    results = [br(pnl=0.242), br(pnl=-1.008, won=False)]
    out = random_side_null(results, n_sims=100, seed=0)
    assert out["observed_roi"] == pytest.approx((0.242 - 1.008) / 2.0)


def test_p_value_in_open_unit_interval():
    results = [br() for _ in range(20)]
    out = random_side_null(results, n_sims=500, seed=1)
    assert 0.0 < out["p_value"] <= 1.0
