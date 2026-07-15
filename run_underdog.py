"""The other side: always bet the underdog (complement of the favourite).

Pure mirror of run_mc.py's pipeline over the same 2,418 bets in
results/results.json -- same snapshot prices, same $1 flat stake, same fee
model, side = complement of the favourite. Same pre-registered gate, same
Monte Carlo machinery (compute_mc), same seeds, 10,000 sims by default.

Run after run_backtest.py has produced results/results.json:
    python run_underdog.py
"""
from __future__ import annotations
import json
import pathlib
from backtest.engine import BetResult
from backtest.mirror import mirror_to_underdog
from run_mc import compute_mc


def main(n_sims: int = 10000, seed: int = 0) -> dict:
    payload = json.loads(pathlib.Path("results/results.json").read_text())
    fav_bets = [BetResult(**d) for d in payload["bets"]]
    underdog_bets = [mirror_to_underdog(b) for b in fav_bets]

    out = compute_mc([b.model_dump() for b in underdog_bets], n_sims=n_sims, seed=seed)
    pathlib.Path("results/mc_results_underdog.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"gate": out["gate"], "observed_roi": out["observed"]["roi"],
                      "ci": out["bootstrap_ci"], "p": out["null"]["p_value"]}, indent=2))
    return out


if __name__ == "__main__":
    main()
