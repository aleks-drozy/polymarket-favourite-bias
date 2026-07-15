import json
from backtest.engine import BetResult
from backtest.mirror import mirror_to_underdog
from mc.metrics import roi
from run_underdog import main


def bet(mid, entry, won, cat="politics", ts=1_700_000_000, side="YES", is_coinflip=False):
    shares = 1.0 / entry
    from backtest.fees import taker_fee
    fee = taker_fee(shares, entry, cat)
    pnl = (shares - 1.0 - fee) if won else (-1.0 - fee)
    return BetResult(market_id=mid, category=cat, side=side, entry_price=entry,
                     is_coinflip=is_coinflip, resolved_ts=ts, won=won, pnl=pnl)


def test_run_underdog_writes_expected_structure(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    bets = [
        bet("m1", 0.8, True, cat="politics"),
        bet("m2", 0.7, False, cat="sports", side="NO"),
        bet("m3", 0.55, True, cat="crypto"),
        bet("m4", 0.9, False, cat="finance", side="NO"),
    ]
    payload = {"config": {"n_bets": len(bets)}, "bets": [b.model_dump() for b in bets]}
    (results_dir / "results.json").write_text(json.dumps(payload))

    monkeypatch.chdir(tmp_path)
    out = main(n_sims=200)

    out_path = tmp_path / "results" / "mc_results_underdog.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text())

    assert set(written) >= {"observed", "bootstrap_ci", "null", "concentration",
                            "fee_sensitivity", "gate", "coinflip_slice"}
    assert written == out

    mirrored = [mirror_to_underdog(b) for b in bets]
    expected_roi = roi([m.pnl for m in mirrored], [m.stake for m in mirrored])
    assert written["observed"]["roi"] == expected_roi
