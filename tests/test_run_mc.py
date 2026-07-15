import json
import pathlib
from backtest.engine import BetResult
from run_mc import compute_mc, verdict


def br(pnl, cat="politics", ts=1_700_000_000):
    return BetResult(market_id="m", category=cat, side="YES", entry_price=0.8,
                     is_coinflip=False, resolved_ts=ts, won=pnl > 0, pnl=pnl)


def test_verdict_requires_both_gate_parts():
    assert verdict(True, True) == "PROFITABLE"
    assert verdict(True, False) == "NOT PROVEN"
    assert verdict(False, True) == "NOT PROVEN"


def test_compute_mc_structure(tmp_path):
    bets = [br(0.2), br(-1.0), br(0.25), br(0.2, cat="sports", ts=1_738_000_000)] * 10
    out = compute_mc([b.model_dump() for b in bets], n_sims=200)
    assert set(out) >= {"observed", "bootstrap_ci", "null", "concentration",
                        "fee_sensitivity", "gate", "coinflip_slice"}
    assert out["gate"]["verdict"] in ("PROFITABLE", "NOT PROVEN")
    assert 0.0 < out["null"]["p_value"] <= 1.0
