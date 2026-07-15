"""Monte Carlo validation + pre-registered gate verdict. Run after run_backtest.py:
python run_mc.py"""
from __future__ import annotations
import json
import pathlib
from backtest.engine import BetResult
from backtest.fees import taker_fee
from mc.metrics import roi, win_rate, total_pnl, max_drawdown
from mc.bootstrap import bootstrap_roi, ci
from mc.reshuffle import random_side_null
from mc.concentration import concentration_check


def verdict(ci_lower_positive: bool, not_concentrated: bool) -> str:
    return "PROFITABLE" if (ci_lower_positive and not_concentrated) else "NOT PROVEN"


def _roi_at_fee_multiple(bets: list[BetResult], mult: float) -> float:
    pnls = []
    for b in bets:
        shares = b.stake / b.entry_price
        base_fee = taker_fee(shares, b.entry_price, b.category)
        gross = (shares - b.stake) if b.won else -b.stake
        pnls.append(gross - mult * base_fee)
    return roi(pnls, [b.stake for b in bets])


def compute_mc(bet_dicts: list[dict], n_sims: int = 10000, seed: int = 0) -> dict:
    bets = [BetResult(**d) for d in bet_dicts]
    pnls = [b.pnl for b in bets]
    stakes = [b.stake for b in bets]

    boot = ci(bootstrap_roi(pnls, stakes, n_sims=n_sims, seed=seed))
    null = random_side_null(bets, n_sims=n_sims, seed=seed)
    conc = concentration_check(bets, n_sims=max(2000, n_sims // 5), seed=seed)

    coin = [b for b in bets if b.is_coinflip]
    gate_ci = boot["lower"] > 0.0
    gate_conc = not conc["concentrated"]
    return {
        "observed": {"roi": roi(pnls, stakes), "win_rate": win_rate(pnls),
                     "total_pnl": total_pnl(pnls), "max_drawdown": max_drawdown(pnls),
                     "n_bets": len(bets)},
        "bootstrap_ci": boot,
        "null": {"p_value": null["p_value"],
                 "null_mean_roi": float(null["null_rois"].mean())},
        "concentration": {k: v for k, v in conc.items() if k != "groups"} | {
            "groups": {g: {"n_excluded": d["n_excluded"], "ci_lower": d["ci"]["lower"]}
                       for g, d in conc["groups"].items()}},
        "fee_sensitivity": {"roi_no_fees": _roi_at_fee_multiple(bets, 0.0),
                            "roi_double_fees": _roi_at_fee_multiple(bets, 2.0)},
        "coinflip_slice": {"n": len(coin),
                           "roi": roi([b.pnl for b in coin], [b.stake for b in coin])},
        "gate": {"ci_lower_positive": gate_ci, "not_concentrated": gate_conc,
                 "verdict": verdict(gate_ci, gate_conc)},
    }


def main():
    payload = json.loads(pathlib.Path("results/results.json").read_text())
    out = compute_mc(payload["bets"])
    pathlib.Path("results/mc_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"gate": out["gate"], "observed_roi": out["observed"]["roi"],
                      "ci": out["bootstrap_ci"], "p": out["null"]["p_value"]}, indent=2))


if __name__ == "__main__":
    main()
