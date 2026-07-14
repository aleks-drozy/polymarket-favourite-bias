# PolyMarket Favourite-Bias Backtest — Design Spec

**Date:** 2026-07-14
**Status:** Approved (brainstorm)

## 1. Goal

Test, historically, whether always betting on the market favourite (the side with implied probability > 50%) in PolyMarket's binary Yes/No markets would have been profitable — and whether any edge found is statistically real or just noise. This is a research study for a public portfolio piece (GitHub, LinkedIn, CV), not a live trading system.

**Not in scope for this project:** placing real bets, paper trading, or any execution layer. If the study finds a real edge, going live is a separate, later project with its own design.

## 2. Background

This mirrors the academic "favourite-longshot bias" literature (well documented in horse racing and sports betting: favourites tend to be underpriced relative to their true win probability). PolyMarket differs from a bookmaker — no house vig, but there is a bid-ask spread and a taker fee — so it's an open question whether the bias replicates there.

This project follows the same rigor bar as the sibling project `Trading-Strategy-Monte-Carlo-Simulation` (vault: `14-monte-carlo`): pre-registered significance gate, Monte Carlo validation before trusting any point estimate, and an honest null result reported as-is if that's what the data shows.

## 3. Scope

- **Markets:** resolved, binary Yes/No PolyMarket markets only, above a liquidity/volume floor calibrated empirically once the data is in hand (target: land in the thousands-of-markets range — research confirmed a "Standard Binary + liquidity" filter yields a sample this size).
- **Category** (politics, sports, crypto, etc.) is a *breakdown dimension* in the results, not a filter — the full binary-market sample stays intact so category-level patterns can be reported without shrinking the core sample.
- **Favourite definition:** whichever side (YES or NO) has implied probability > 50% at the snapshot time.
- **Snapshot timing:** price sampled in the window 24–48 hours before market resolution — the closest a backtest can get to "a bet you could realistically have placed" while avoiding most late-window "the outcome is now obvious" leakage. **Candle selection rule (deterministic):** take the latest available candle that is still ≥24h before resolution (the freshest price still outside the excluded final-24h zone), as long as it is ≤48h before resolution. If no candle exists in that band, the market is excluded (see §9).
- **Bet sizing:** flat $1 stake per market. This isolates whether the favourite-selection rule itself has edge, before any bankroll-management layer (Kelly, liquidity-aware sizing) gets added — those are explicitly out of scope for v1.

## 4. Data Sourcing (hybrid)

1. **Bulk load** from a pre-built historical dataset (e.g. the `manja316/polymarket-historical-data` GitHub dataset: 13,964 markets, 15-minute snapshots, resolution outcomes included) into the canonical schema below.
2. **Cross-validate** a random sample (~100 markets) against PolyMarket's official APIs — Gamma API (market metadata, free, no auth) and CLOB API `/prices-history` (historical prices, free, no auth for reads) — and log the agreement %.
3. **Backfill** any gaps the dataset doesn't cover using the same API client.

**Canonical schema:** `market_id, category, created_ts, resolved_ts, resolved_outcome, snapshot_ts, price_yes, price_no, volume`.

**Known constraint:** CLOB `/prices-history` only returns 12h+ candles for resolved markets (fine-grained 5s bars are for active markets only) — acceptable given the 24-48h snapshot window doesn't need finer resolution than that.

## 5. Payout & Fee Math

Each qualifying market becomes one simulated bet: spend $1 buying shares of the favourite at its snapshot price. If it resolves correctly, shares redeem at $1 each; if not, they're worth $0. PolyMarket's taker fee (category-based — politics/finance/tech ≈ 0.04, crypto ≈ 0.07, geopolitics ≈ 0%, symmetric around the 50% price point per current research) is subtracted per trade. The exact fee formula is implemented directly from `docs.polymarket.com/trading/fees` at build time rather than assumed from secondhand research. Gas/settlement cost is treated as negligible (PolyMarket's relayer absorbs it for retail users under normal conditions).

This produces one net P&L number per market. The full P&L series is the input to everything downstream.

## 6. Statistical Validation

- **Bootstrap** (resample-with-replacement, 10,000 sims) → 95% confidence interval on aggregate ROI.
- **Reshuffle/permutation null** (10,000 sims) → tests whether specifically picking the favourite beats picking sides some other consistent way on the same markets (not just "the platform is profitable to trade on generally").
- **Pre-registered significance gate** (decided now, before any result exists): the strategy is called "profitable" only if the bootstrap CI's lower bound clears breakeven (> 0% ROI after fees) **and** the edge isn't concentrated in one category or time window (e.g. entirely explained by the 2024 US election cycle). Anything short of that is reported as null/marginal, not massaged.

## 7. Conditional ML Phase

Only triggered if the Monte Carlo result is borderline or shows a suspicious subgroup pattern worth investigating — mirrors the `14-monte-carlo` repo's conditional ML phase. If triggered: a walk-forward-validated classifier testing whether features (category, price level, market duration, volume) predict which favourites to skip, evaluated against a same-universe random-skip baseline, reported honestly even if null. A clean strong "yes" or clean "no" from the Monte Carlo phase does not need this phase.

## 8. Architecture

```
polymarket-favourite-bias/
├── data/          # dataset loader, API client, cross-validation script → canonical dataset
├── backtest/      # favourite labeling, snapshot selection, payout/fee simulation
├── mc/            # bootstrap.py, reshuffle.py (mirrors Trading-Strategy-Monte-Carlo-Simulation)
├── ml/            # conditional phase 2 only, model.py + evaluate.py
├── tests/         # unit tests for fee math, favourite/snapshot selection, payout math
├── README.md
└── WRITEUP.md     # methodology + results + honest verdict, recruiter-readable
```

## 9. Error Handling

Nothing is silently dropped — every exclusion is logged with a reason so the final sample size is fully accounted for in the writeup:
- No candle satisfies the §3 candle-selection rule (e.g. market resolved too fast to have any snapshot ≥24h before resolution) → excluded, reason logged.
- Dataset-vs-live-API disagreement above a sanity threshold on the cross-validated sample → flagged as a data-quality blocker before trusting the full backtest; the agreement % is reported in the writeup regardless of outcome.
- A market at ~50/50 at snapshot (no real favourite) → included, not excluded (excluding it would bias the favourite definition), but callable out as a diagnostic slice.

## 10. Testing Strategy

TDD throughout: fee math, favourite-selection/tie-breaking, snapshot-window selection, and per-market payout math all get unit tests with hand-computed expected values before running against real data. Bootstrap/reshuffle get a determinism test (fixed seed) and a sanity case (an all-same-outcome series should show ~zero variance). Ingestion gets a smoke test against a couple of well-known real markets as an end-to-end wiring check.

## 11. Deliverables & Conventions

- **Code repo:** `C:\Users\Alex\Projects\polymarket-favourite-bias` (this repo).
- **Vault folder:** `16-polymarket-favourites` (next number per `VAULT_INDEX.md`), seeded with the standard 6-doc set once implementation kicks off.
- **README.md** (quick summary + how to run) + **WRITEUP.md** (full methodology + honest verdict, recruiter-readable) — same convention as `Trading-Strategy-Monte-Carlo-Simulation`.
- **Publish timing:** stays local until the honest result is in and reviewed — same discipline as the FYP Monte Carlo project (local until Alex's explicit go-ahead). Not pushed to GitHub or posted about until then.

## 12. Out of Scope / Future Work

- Live or paper-trading execution (deferred; would be its own design if the backtest shows a real edge).
- Kelly-criterion or liquidity-aware bet sizing (v1 uses flat stake to isolate signal from bankroll management).
- Multi-outcome (non-binary) markets.
