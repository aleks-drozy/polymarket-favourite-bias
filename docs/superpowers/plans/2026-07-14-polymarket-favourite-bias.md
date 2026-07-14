# PolyMarket Favourite-Bias Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a historically-grounded backtest answering whether always betting the favourite on PolyMarket binary markets is profitable, validated by bootstrap + random-side-null Monte Carlo against a pre-registered significance gate.

**Architecture:** A Python pipeline in four layers — `data/` (dataset bulk-load + official-API snapshot fetch + cross-validation), `backtest/` (favourite labeling, snapshot selection, fee/payout math, exclusion logging), `mc/` (bootstrap CI, random-side permutation null, concentration check), and two orchestration scripts (`run_backtest.py`, `run_mc.py`) that emit JSON results consumed by the WRITEUP. Mirrors the structure and testing conventions of `C:\Users\Alex\Projects\Trading-Strategy-Monte-Carlo-Simulation`.

**Tech Stack:** Python 3.12, numpy, pandas, pydantic v2, requests, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-14-polymarket-favourite-bias-design.md` — this plan implements it exactly.
- **Local only:** never push to GitHub, never create a remote. Publishing waits for Alex's explicit go.
- **Pre-registered gate (verbatim from spec §6):** the strategy is called "profitable" only if the bootstrap CI's lower bound clears breakeven (> 0% ROI after fees) AND the edge isn't concentrated in one category or time window. Operationalized in Task 9. Never weaken this gate to fit a result.
- **ML phase is NOT built in this plan.** Trigger condition (spec §7): only if the MC verdict is borderline (CI straddles 0 within ±1 percentage point) or the concentration check reveals a suspicious subgroup pattern. If triggered, that's a new plan.
- All Monte Carlo runs: 10,000 sims, fixed seed 0, deterministic.
- Every excluded market is logged with a reason; nothing silently dropped.
- Flat $1 stake per market. ROI = total P&L / total staked.
- TDD: failing test → run → minimal code → run → commit. Frequent commits, conventional messages.
- Network-dependent tests are marked `@pytest.mark.network` and excluded from the default run (`pytest -m "not network"`); offline tests use checked-in fixtures.
- Windows environment: paths under `C:\Users\Alex\Projects\polymarket-favourite-bias`; use `python -m pytest`.

## File Structure

```
polymarket-favourite-bias/
├── data/
│   ├── __init__.py
│   ├── schema.py            # MarketRecord + Candle + Exclusion models
│   ├── api_client.py        # Gamma + CLOB read-only client with disk cache
│   ├── dataset_loader.py    # bulk metadata load (third-party dataset or Gamma fallback)
│   └── crossval.py          # dataset-vs-API agreement check
├── backtest/
│   ├── __init__.py
│   ├── fees.py              # category-based taker fee
│   ├── favourite.py         # favourite labeling (>50% side)
│   ├── snapshot.py          # 24-48h candle selection rule
│   └── engine.py            # per-market P&L + exclusion collection
├── mc/
│   ├── __init__.py
│   ├── metrics.py           # roi, win_rate, total_pnl, max_drawdown
│   ├── bootstrap.py         # 10k-sim bootstrap, CI summary
│   ├── reshuffle.py         # random-side permutation null
│   └── concentration.py     # leave-one-out category/quarter gate check
├── tests/
│   ├── __init__.py
│   ├── fixtures/            # small checked-in CSV/JSON fixtures
│   └── test_*.py            # one per module
├── results/                 # gitignored except .gitkeep: results.json, exclusions.csv, mc_results.json, crossval_report.json
├── cache/                   # gitignored: API response cache
├── run_backtest.py
├── run_mc.py
├── requirements.txt
├── README.md
└── WRITEUP.md
```

---

### Task 1: Scaffold + canonical schema

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.gitignore`, `data/__init__.py`, `backtest/__init__.py`, `mc/__init__.py`, `tests/__init__.py`, `results/.gitkeep`
- Create: `data/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: `MarketRecord` (pydantic model, fields: `market_id: str, category: str, created_ts: int, resolved_ts: int, resolved_outcome: str` ("YES"|"NO")`, volume: float`), `Candle` (`t: int, price_yes: float`), `Exclusion` (`market_id: str, reason: str`). All timestamps are unix seconds UTC. `price_no` is always computed as `1 - price_yes`, never stored.

- [ ] **Step 1: Write scaffolding files**

`requirements.txt`:
```
numpy==2.1.3
pandas==2.2.3
pydantic==2.9.2
requests==2.32.3
pytest==8.3.3
```

`pytest.ini`:
```ini
[pytest]
markers =
    network: hits live PolyMarket APIs (deselect with -m "not network")
addopts = -m "not network"
```

`.gitignore`:
```
__pycache__/
.pytest_cache/
cache/
results/*
!results/.gitkeep
*.egg-info/
```

Empty `__init__.py` in `data/`, `backtest/`, `mc/`, `tests/`. Empty `results/.gitkeep`.

- [ ] **Step 2: Write the failing test**

`tests/test_schema.py`:
```python
import pytest
from pydantic import ValidationError
from data.schema import MarketRecord, Candle, Exclusion


def make_record(**over):
    base = dict(market_id="0xabc", category="politics", created_ts=1700000000,
                resolved_ts=1705000000, resolved_outcome="YES", volume=50000.0)
    base.update(over)
    return MarketRecord(**base)


def test_valid_record_roundtrips():
    r = make_record()
    assert r.market_id == "0xabc" and r.resolved_outcome == "YES"


def test_outcome_must_be_yes_or_no():
    with pytest.raises(ValidationError):
        make_record(resolved_outcome="INVALID")


def test_resolved_must_be_after_created():
    with pytest.raises(ValidationError):
        make_record(resolved_ts=1600000000)


def test_candle_price_bounds():
    assert Candle(t=1700000000, price_yes=0.62).price_yes == 0.62
    with pytest.raises(ValidationError):
        Candle(t=1700000000, price_yes=1.5)


def test_exclusion_holds_reason():
    e = Exclusion(market_id="0xabc", reason="no_candle_in_window")
    assert e.reason == "no_candle_in_window"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_schema.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'data.schema'`

- [ ] **Step 4: Write minimal implementation**

`data/schema.py`:
```python
"""Canonical market/candle/exclusion models. Timestamps are unix seconds UTC."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class MarketRecord(BaseModel):
    market_id: str
    category: str
    created_ts: int
    resolved_ts: int
    resolved_outcome: Literal["YES", "NO"]
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def _resolved_after_created(self):
        if self.resolved_ts <= self.created_ts:
            raise ValueError("resolved_ts must be after created_ts")
        return self


class Candle(BaseModel):
    t: int
    price_yes: float = Field(gt=0, lt=1)


class Exclusion(BaseModel):
    market_id: str
    reason: str
```

- [ ] **Step 5: Run tests, verify pass**

Run: `python -m pytest tests/test_schema.py -v`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: scaffold project + canonical schema models"
```

---

### Task 2: Fee module (verify formula from official docs first)

**Files:**
- Create: `backtest/fees.py`
- Test: `tests/test_fees.py`

**Interfaces:**
- Produces: `taker_fee(shares: float, price: float, category: str) -> float` and `CATEGORY_RATES: dict[str, float]` with key `"_default"`.

- [ ] **Step 1: Verify the live fee formula**

Fetch `https://docs.polymarket.com/trading/fees` (WebFetch or `requests.get` + read). Record in a comment at the top of `backtest/fees.py`: the exact formula, the per-category base rates, and the fetch date. Research (2026-07-14) indicates: taker-only, symmetric around 50¢, crypto ≈ 0.07, politics/finance/tech ≈ 0.04, geopolitics 0. **If the live docs differ from the research or from the default formula below, update `FORMULA`, `CATEGORY_RATES`, and recompute every hand-computed constant in the test before writing it.** If the docs are unreachable, proceed with the defaults below and log that in `WRITEUP.md`'s limitations section (Task 15) — the fee-sensitivity run in Task 14 bounds the impact either way.

- [ ] **Step 2: Write the failing test** (constants assume `fee = rate * price * (1 - price) * shares`; recompute if Step 1 found a different formula)

`tests/test_fees.py`:
```python
import pytest
from backtest.fees import taker_fee, CATEGORY_RATES


def test_politics_fee_hand_computed():
    # rate 0.04, p=0.6, 1.25 shares: 0.04 * 0.6 * 0.4 * 1.25 = 0.012
    assert taker_fee(1.25, 0.6, "politics") == pytest.approx(0.012)


def test_fee_symmetric_around_half():
    assert taker_fee(1.0, 0.3, "crypto") == pytest.approx(taker_fee(1.0, 0.7, "crypto"))


def test_geopolitics_free():
    assert taker_fee(2.0, 0.55, "geopolitics") == 0.0


def test_unknown_category_uses_default():
    assert taker_fee(1.0, 0.5, "weird-new-category") == pytest.approx(
        CATEGORY_RATES["_default"] * 0.25)


def test_zero_shares_zero_fee():
    assert taker_fee(0.0, 0.6, "crypto") == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_fees.py -v`
Expected: FAIL — `No module named 'backtest.fees'`

- [ ] **Step 4: Write minimal implementation**

`backtest/fees.py`:
```python
"""Taker fee per docs.polymarket.com/trading/fees.

FORMULA (verified <fetch-date>): fee = rate * price * (1 - price) * shares
Rates (verified <fetch-date>): crypto 0.07; politics/finance/tech 0.04; geopolitics 0.
Update this comment + CATEGORY_RATES if the docs change.
"""
from __future__ import annotations

CATEGORY_RATES: dict[str, float] = {
    "crypto": 0.07,
    "politics": 0.04,
    "finance": 0.04,
    "tech": 0.04,
    "geopolitics": 0.0,
    "_default": 0.04,
}


def taker_fee(shares: float, price: float, category: str) -> float:
    rate = CATEGORY_RATES.get(category.lower(), CATEGORY_RATES["_default"])
    return rate * price * (1.0 - price) * shares
```

- [ ] **Step 5: Run tests, verify pass**

Run: `python -m pytest tests/test_fees.py -v`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: category-based taker fee verified against official docs"
```

---

### Task 3: Favourite labeling

**Files:**
- Create: `backtest/favourite.py`
- Test: `tests/test_favourite.py`

**Interfaces:**
- Produces: `label_favourite(price_yes: float) -> tuple[str, float, bool]` returning `(side, entry_price, is_coinflip)` where side ∈ {"YES","NO"}, `entry_price` is the favourite side's price, and `is_coinflip` is True when `price_yes == 0.5` exactly (tie-break: bet YES, flagged for the diagnostic slice — spec §9 says ~50/50 markets are included, never excluded).

- [ ] **Step 1: Write the failing test**

`tests/test_favourite.py`:
```python
from backtest.favourite import label_favourite


def test_yes_is_favourite():
    assert label_favourite(0.72) == ("YES", 0.72, False)


def test_no_is_favourite():
    side, price, coin = label_favourite(0.31)
    assert side == "NO" and price == 0.69 and coin is False


def test_exact_coinflip_ties_to_yes_and_flags():
    assert label_favourite(0.5) == ("YES", 0.5, True)


def test_price_no_is_complement():
    _, price, _ = label_favourite(0.4)
    assert price == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_favourite.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

`backtest/favourite.py`:
```python
"""Favourite = the side with implied probability > 50% at snapshot (spec §3)."""
from __future__ import annotations


def label_favourite(price_yes: float) -> tuple[str, float, bool]:
    if price_yes > 0.5:
        return "YES", price_yes, False
    if price_yes < 0.5:
        return "NO", 1.0 - price_yes, False
    return "YES", 0.5, True  # exact tie: deterministic YES, flagged coinflip
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_favourite.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: favourite labeling with deterministic coinflip tie-break"
```

---

### Task 4: Snapshot selection

**Files:**
- Create: `backtest/snapshot.py`
- Test: `tests/test_snapshot.py`

**Interfaces:**
- Consumes: `Candle` from `data.schema`.
- Produces: `select_snapshot(candles: list[Candle], resolved_ts: int) -> Candle | None` — the latest candle with `resolved_ts - 48h <= t <= resolved_ts - 24h`; None if no candle qualifies (caller logs the exclusion).

- [ ] **Step 1: Write the failing test**

`tests/test_snapshot.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_snapshot.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

`backtest/snapshot.py`:
```python
"""Deterministic snapshot rule (spec §3): latest candle >=24h before resolution, <=48h."""
from __future__ import annotations
from data.schema import Candle

H24 = 24 * 3600
H48 = 48 * 3600


def select_snapshot(candles: list[Candle], resolved_ts: int) -> Candle | None:
    lo, hi = resolved_ts - H48, resolved_ts - H24
    eligible = [c for c in candles if lo <= c.t <= hi]
    return max(eligible, key=lambda c: c.t) if eligible else None
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_snapshot.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: 24-48h snapshot candle selection rule"
```

---

### Task 5: Payout engine

**Files:**
- Create: `backtest/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `MarketRecord`, `Candle`, `Exclusion` (schema); `label_favourite`; `select_snapshot`; `taker_fee`.
- Produces:
  - `BetResult` (pydantic): `market_id, category, side, entry_price, is_coinflip, resolved_ts, won: bool, pnl: float, stake: float = 1.0`.
  - `simulate_market(record: MarketRecord, candles: list[Candle], stake: float = 1.0) -> BetResult | Exclusion`.
  - `run_backtest(records_with_candles: list[tuple[MarketRecord, list[Candle]]]) -> tuple[list[BetResult], list[Exclusion]]`.
- Payout math (spec §5): shares = stake / entry_price; fee = taker_fee(shares, entry_price, category); win → pnl = shares − stake − fee; lose → pnl = −stake − fee.

- [ ] **Step 1: Write the failing test** (hand-computed: p=0.8, politics, $1: shares=1.25, fee=0.04·0.8·0.2·1.25=0.008; win pnl=0.25−0.008=0.242; lose pnl=−1.008)

`tests/test_engine.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_engine.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

`backtest/engine.py`:
```python
"""Per-market flat-stake favourite bet simulation (spec §5). Nothing silently dropped."""
from __future__ import annotations
from pydantic import BaseModel
from data.schema import MarketRecord, Candle, Exclusion
from backtest.favourite import label_favourite
from backtest.snapshot import select_snapshot
from backtest.fees import taker_fee


class BetResult(BaseModel):
    market_id: str
    category: str
    side: str
    entry_price: float
    is_coinflip: bool
    resolved_ts: int
    won: bool
    pnl: float
    stake: float = 1.0


def simulate_market(record: MarketRecord, candles: list[Candle],
                    stake: float = 1.0) -> BetResult | Exclusion:
    snap = select_snapshot(candles, record.resolved_ts)
    if snap is None:
        return Exclusion(market_id=record.market_id, reason="no_candle_in_window")
    side, entry_price, is_coinflip = label_favourite(snap.price_yes)
    shares = stake / entry_price
    fee = taker_fee(shares, entry_price, record.category)
    won = side == record.resolved_outcome
    pnl = (shares - stake - fee) if won else (-stake - fee)
    return BetResult(market_id=record.market_id, category=record.category, side=side,
                     entry_price=entry_price, is_coinflip=is_coinflip,
                     resolved_ts=record.resolved_ts, won=won, pnl=pnl, stake=stake)


def run_backtest(records_with_candles: list[tuple[MarketRecord, list[Candle]]]
                 ) -> tuple[list[BetResult], list[Exclusion]]:
    results, exclusions = [], []
    for record, candles in records_with_candles:
        r = simulate_market(record, candles)
        (exclusions if isinstance(r, Exclusion) else results).append(r)
    return results, exclusions
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_engine.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: payout engine with fee-aware P&L and exclusion logging"
```

---

### Task 6: MC metrics

**Files:**
- Create: `mc/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `roi(pnls, stakes) -> float` (total pnl / total staked), `total_pnl(pnls) -> float`, `win_rate(pnls) -> float`, `max_drawdown(pnls) -> float` (same semantics as the sibling repo: largest peak-to-trough drop in cumulative equity from implicit peak 0.0, non-negative).

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

`mc/metrics.py`:
```python
"""Pure scoring functions over per-market P&L series (mirrors sibling repo mc/metrics.py)."""
from __future__ import annotations
from typing import Sequence


def total_pnl(pnls: Sequence[float]) -> float:
    return float(sum(pnls))


def roi(pnls: Sequence[float], stakes: Sequence[float]) -> float:
    staked = float(sum(stakes))
    return total_pnl(pnls) / staked if staked > 0 else 0.0


def win_rate(pnls: Sequence[float]) -> float:
    if len(pnls) == 0:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls)


def max_drawdown(pnls: Sequence[float]) -> float:
    peak, worst, running = 0.0, 0.0, 0.0
    for p in pnls:
        running += p
        peak = max(peak, running)
        worst = max(worst, peak - running)
    return worst
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: MC metrics (roi, win_rate, total_pnl, max_drawdown)"
```

---

### Task 7: Bootstrap

**Files:**
- Create: `mc/bootstrap.py`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Produces: `bootstrap_roi(pnls: Sequence[float], stakes: Sequence[float], n_sims: int = 10000, seed: int = 0) -> np.ndarray` (per-sim ROI, resampling (pnl, stake) pairs with replacement), and `ci(values: np.ndarray, level: float = 0.95) -> dict` with keys `mean, lower, upper` (percentile CI).

- [ ] **Step 1: Write the failing test**

`tests/test_bootstrap.py`:
```python
import numpy as np
import pytest
from mc.bootstrap import bootstrap_roi, ci


def test_determinism_with_seed():
    pnls, stakes = [0.2, -1.0, 0.3, 0.25], [1.0] * 4
    a = bootstrap_roi(pnls, stakes, n_sims=500, seed=0)
    b = bootstrap_roi(pnls, stakes, n_sims=500, seed=0)
    assert a.shape == (500,) and np.allclose(a, b)


def test_all_same_outcome_has_zero_variance():
    vals = bootstrap_roi([0.25] * 50, [1.0] * 50, n_sims=200, seed=0)
    assert np.allclose(vals, 0.25)


def test_ci_bounds_ordered_and_hand_computed():
    vals = np.arange(1001.0) / 1000.0  # uniform 0..1
    c = ci(vals)
    assert c["lower"] == pytest.approx(0.025, abs=0.002)
    assert c["upper"] == pytest.approx(0.975, abs=0.002)
    assert c["lower"] < c["mean"] < c["upper"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bootstrap.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

`mc/bootstrap.py`:
```python
"""Bootstrap (resample-with-replacement) distribution of aggregate ROI (spec §6)."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from mc.metrics import roi


def bootstrap_roi(pnls: Sequence[float], stakes: Sequence[float],
                  n_sims: int = 10000, seed: int = 0) -> np.ndarray:
    pn = np.asarray(pnls, dtype=float)
    st = np.asarray(stakes, dtype=float)
    n = len(pn)
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims)
    for i in range(n_sims):
        idx = rng.integers(0, n, size=n)
        out[i] = roi(pn[idx], st[idx])
    return out


def ci(values: np.ndarray, level: float = 0.95) -> dict:
    alpha = (1.0 - level) / 2.0
    return {"mean": float(values.mean()),
            "lower": float(np.percentile(values, 100 * alpha)),
            "upper": float(np.percentile(values, 100 * (1 - alpha)))}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_bootstrap.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: seeded bootstrap ROI with percentile CI"
```

---

### Task 8: Random-side permutation null

**Files:**
- Create: `mc/reshuffle.py`
- Test: `tests/test_reshuffle.py`

**Interfaces:**
- Consumes: `BetResult` from `backtest.engine`; `taker_fee`; `roi`.
- Produces: `random_side_null(results: list[BetResult], n_sims: int = 10000, seed: int = 0) -> dict` with keys `null_rois: np.ndarray`, `observed_roi: float`, `p_value: float`. Null model (spec §6): per sim, per market, pick YES or NO uniformly at random, buy that side at its snapshot price (favourite side price if picked side is the favourite, else its complement), same $1 stake and fee model, aggregate ROI. p-value = (1 + #{null ≥ observed}) / (n_sims + 1).

- [ ] **Step 1: Write the failing test**

`tests/test_reshuffle.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reshuffle.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

`mc/reshuffle.py`:
```python
"""Random-side null: does picking the FAVOURITE beat picking sides at random
on the same markets, same prices, same fees? (spec §6)"""
from __future__ import annotations
import numpy as np
from backtest.engine import BetResult
from backtest.fees import taker_fee
from mc.metrics import roi


def _pnl_for_side(r: BetResult, pick_favourite_side: bool, stake: float = 1.0) -> float:
    price = r.entry_price if pick_favourite_side else 1.0 - r.entry_price
    won = r.won if pick_favourite_side else not r.won
    shares = stake / price
    fee = taker_fee(shares, price, r.category)
    return (shares - stake - fee) if won else (-stake - fee)


def random_side_null(results: list[BetResult], n_sims: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    observed = roi([r.pnl for r in results], [r.stake for r in results])
    n = len(results)
    null_rois = np.empty(n_sims)
    for i in range(n_sims):
        picks = rng.random(n) < 0.5  # True = take the favourite side
        pnls = [_pnl_for_side(r, bool(p)) for r, p in zip(results, picks)]
        null_rois[i] = roi(pnls, [1.0] * n)
    p_value = (1.0 + float((null_rois >= observed).sum())) / (n_sims + 1.0)
    return {"null_rois": null_rois, "observed_roi": observed, "p_value": p_value}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_reshuffle.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: random-side permutation null with seeded p-value"
```

---

### Task 9: Concentration check (gate part 2)

**Files:**
- Create: `mc/concentration.py`
- Test: `tests/test_concentration.py`

**Interfaces:**
- Consumes: `BetResult`; `bootstrap_roi`, `ci`.
- Produces: `concentration_check(results: list[BetResult], n_sims: int = 2000, seed: int = 0) -> dict` — leave-one-out over categories and calendar quarters (quarter derived from `resolved_ts` UTC, format `"2024Q4"`). For each group g: recompute bootstrap CI on the sample excluding g. Returns `{"groups": {name: {"n_excluded": int, "ci": {...}}}, "concentrated": bool}` where `concentrated` is True iff the FULL-sample CI lower bound > 0 but ANY leave-one-out CI lower bound <= 0. Groups smaller than 2% of the sample are skipped (excluding them can't meaningfully move the estimate); skipped groups are listed under `"skipped_small_groups"`.

- [ ] **Step 1: Write the failing test**

`tests/test_concentration.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_concentration.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

`mc/concentration.py`:
```python
"""Gate part 2 (spec §6): the edge must not be concentrated in one category
or time window. Leave-one-group-out bootstrap CIs."""
from __future__ import annotations
from datetime import datetime, timezone
from backtest.engine import BetResult
from mc.bootstrap import bootstrap_roi, ci

MIN_GROUP_FRACTION = 0.02


def quarter_of(ts: int) -> str:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _loo_ci(results: list[BetResult], excluded: set[int], n_sims: int, seed: int) -> dict:
    kept = [r for i, r in enumerate(results) if i not in excluded]
    vals = bootstrap_roi([r.pnl for r in kept], [r.stake for r in kept],
                         n_sims=n_sims, seed=seed)
    return ci(vals)


def concentration_check(results: list[BetResult], n_sims: int = 2000, seed: int = 0) -> dict:
    n = len(results)
    full = _loo_ci(results, set(), n_sims, seed)
    groups: dict[str, set[int]] = {}
    for i, r in enumerate(results):
        groups.setdefault(f"category:{r.category}", set()).add(i)
        groups.setdefault(f"quarter:{quarter_of(r.resolved_ts)}", set()).add(i)

    out_groups, skipped = {}, []
    concentrated = False
    for name, idx in sorted(groups.items()):
        if len(idx) < MIN_GROUP_FRACTION * n:
            skipped.append(name)
            continue
        loo = _loo_ci(results, idx, n_sims, seed)
        out_groups[name] = {"n_excluded": len(idx), "ci": loo}
        if full["lower"] > 0 and loo["lower"] <= 0:
            concentrated = True
    return {"full_ci": full, "groups": out_groups,
            "skipped_small_groups": skipped, "concentrated": concentrated}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_concentration.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: leave-one-out concentration check for the significance gate"
```

---

### Task 10: API client (Gamma + CLOB, cached)

**Files:**
- Create: `data/api_client.py`
- Create: `tests/fixtures/gamma_market.json`, `tests/fixtures/prices_history.json`
- Test: `tests/test_api_client.py`

**Interfaces:**
- Produces:
  - `PolymarketClient(cache_dir: str = "cache", sleep_s: float = 1.1)` with methods:
    - `get_market(condition_id: str) -> dict` — Gamma `GET https://gamma-api.polymarket.com/markets?condition_ids={id}`, returns the first market dict.
    - `fetch_markets_page(offset: int, limit: int = 100) -> list[dict]` — Gamma `GET /markets?closed=true&limit={limit}&offset={offset}` (the dataset-independent fallback path).
    - `get_price_history(clob_token_id: str, start_ts: int, end_ts: int) -> list[Candle]` — CLOB `GET https://clob.polymarket.com/prices-history?market={token}&startTs=..&endTs=..&fidelity=720`, mapping response `{"history": [{"t":..,"p":..}]}` to `Candle(t, price_yes=p)`.
  - Every GET result is cached to `{cache_dir}/{sha1(url)}.json`; cache hit skips network and sleep. `sleep_s` throttle before each real network call (respects ~60/min unauthenticated limits).

- [ ] **Step 1: Create offline fixtures**

`tests/fixtures/prices_history.json`:
```json
{"history": [{"t": 1699900000, "p": 0.61}, {"t": 1699943200, "p": 0.64}]}
```

`tests/fixtures/gamma_market.json`:
```json
[{"conditionId": "0xabc", "question": "Will X happen?", "category": "Politics",
  "closed": true, "volumeNum": 123456.0,
  "clobTokenIds": "[\"111\", \"222\"]",
  "outcomes": "[\"Yes\", \"No\"]", "outcomePrices": "[\"1\", \"0\"]",
  "createdAt": "2023-10-01T00:00:00Z", "closedTime": "2023-11-15 12:00:00+00"}]
```

- [ ] **Step 2: Write the failing test** (network calls monkeypatched to fixtures; one real smoke test marked `network`)

`tests/test_api_client.py`:
```python
import json
import pathlib
import pytest
from data.api_client import PolymarketClient

FIX = pathlib.Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self._payload


@pytest.fixture
def client(tmp_path, monkeypatch):
    c = PolymarketClient(cache_dir=str(tmp_path), sleep_s=0.0)
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        if "prices-history" in url:
            return FakeResponse(json.loads((FIX / "prices_history.json").read_text()))
        return FakeResponse(json.loads((FIX / "gamma_market.json").read_text()))

    monkeypatch.setattr("data.api_client.requests.get", fake_get)
    c._calls = calls
    return c


def test_price_history_maps_to_candles(client):
    candles = client.get_price_history("111", 1699000000, 1700000000)
    assert len(candles) == 2 and candles[0].price_yes == 0.61


def test_get_market_returns_first(client):
    m = client.get_market("0xabc")
    assert m["conditionId"] == "0xabc"


def test_cache_prevents_second_network_call(client):
    client.get_price_history("111", 1699000000, 1700000000)
    client.get_price_history("111", 1699000000, 1700000000)
    assert len(client._calls) == 1


@pytest.mark.network
def test_live_smoke_known_market():
    # Well-known resolved market: 2024 US Presidential Election winner.
    c = PolymarketClient(cache_dir="cache")
    page = c.fetch_markets_page(offset=0, limit=5)
    assert isinstance(page, list) and len(page) > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_api_client.py -v`
Expected: FAIL — module not found (network test deselected by default)

- [ ] **Step 4: Write minimal implementation**

`data/api_client.py`:
```python
"""Read-only PolyMarket client: Gamma (metadata) + CLOB (price history).
No auth needed for reads. Disk-cached, throttled (~60 req/min unauthenticated)."""
from __future__ import annotations
import hashlib
import json
import pathlib
import time
import requests
from data.schema import Candle

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(self, cache_dir: str = "cache", sleep_s: float = 1.1):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_s = sleep_s

    def _get(self, url: str) -> dict | list:
        key = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if key.exists():
            return json.loads(key.read_text())
        time.sleep(self.sleep_s)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        key.write_text(json.dumps(payload))
        return payload

    def get_market(self, condition_id: str) -> dict:
        payload = self._get(f"{GAMMA}/markets?condition_ids={condition_id}")
        return payload[0]

    def fetch_markets_page(self, offset: int, limit: int = 100) -> list[dict]:
        return self._get(f"{GAMMA}/markets?closed=true&limit={limit}&offset={offset}")

    def get_price_history(self, clob_token_id: str, start_ts: int, end_ts: int) -> list[Candle]:
        payload = self._get(
            f"{CLOB}/prices-history?market={clob_token_id}"
            f"&startTs={start_ts}&endTs={end_ts}&fidelity=720")
        return [Candle(t=int(h["t"]), price_yes=float(h["p"]))
                for h in payload.get("history", [])
                if 0.0 < float(h["p"]) < 1.0]
```

- [ ] **Step 5: Run offline tests, verify pass**

Run: `python -m pytest tests/test_api_client.py -v`
Expected: 3 PASS, 1 deselected

- [ ] **Step 6: Run the live smoke test once**

Run: `python -m pytest tests/test_api_client.py -m network -v`
Expected: PASS (requires internet). If PolyMarket's response shape differs from the fixtures (field names, types), update the fixtures AND the parsing code to match reality, re-run offline tests, and note the discrepancy in the commit message.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: cached read-only Gamma+CLOB API client"
```

---

### Task 11: Dataset loader (inspect-first, Gamma fallback)

**Files:**
- Create: `data/dataset_loader.py`
- Create: `tests/fixtures/dataset_sample.csv` (built in Step 1 from the real download)
- Test: `tests/test_dataset_loader.py`

**Interfaces:**
- Consumes: `MarketRecord`, `PolymarketClient`.
- Produces:
  - `load_dataset_csv(path: str) -> tuple[list[MarketRecord], list[Exclusion]]` — parses the third-party metadata CSV into records; rows that are non-binary, unresolved, or missing required fields become `Exclusion` entries (reasons: `"not_binary"`, `"not_resolved"`, `"missing_field:<name>"`, `"bad_outcome"`).
  - `load_from_gamma(client: PolymarketClient, max_markets: int | None = None) -> tuple[list[MarketRecord], list[Exclusion]]` — the fallback path: paginate `fetch_markets_page` until an empty page, same mapping rules.
  - Both paths map outcome to "YES"/"NO" by which outcome token's final `outcomePrices` entry is "1".

- [ ] **Step 1: Download and inspect the real dataset**

```bash
cd /c/Users/Alex/Projects/polymarket-favourite-bias
git clone --depth 1 https://github.com/manja316/polymarket-historical-data dataset_tmp
python -c "import pandas as pd, glob; [print(f, '\n', pd.read_csv(f, nrows=3).columns.tolist()) for f in glob.glob('dataset_tmp/**/*.csv', recursive=True)[:10]]"
```

Inspect the printed columns. **Decision point:** if the free tier contains usable per-market metadata (id, category, close/resolution time, outcome, volume) → write `COLUMN_MAP` in `dataset_loader.py` mapping their names to ours, copy the first ~20 rows into `tests/fixtures/dataset_sample.csv`, and proceed. If the free tier lacks resolution outcomes or is otherwise unusable → **skip the third-party dataset entirely and use `load_from_gamma` as the primary metadata source** (it's already built); record whichever branch was taken in `DECISIONS`-style comment at the top of `dataset_loader.py` and in the Task 15 WRITEUP. Do not keep `dataset_tmp/` in git (`echo dataset_tmp/ >> .gitignore`).

- [ ] **Step 2: Write the failing test** (adjust fixture column names to the real download; the test asserts BEHAVIOR, which is fixed)

`tests/test_dataset_loader.py`:
```python
import pathlib
from data.dataset_loader import load_dataset_csv

FIX = pathlib.Path(__file__).parent / "fixtures" / "dataset_sample.csv"


def test_loads_valid_binary_resolved_rows():
    records, exclusions = load_dataset_csv(str(FIX))
    assert len(records) > 0
    r = records[0]
    assert r.resolved_outcome in ("YES", "NO")
    assert r.resolved_ts > r.created_ts
    assert r.volume >= 0


def test_non_binary_rows_become_exclusions_with_reason():
    _, exclusions = load_dataset_csv(str(FIX))
    reasons = {e.reason.split(":")[0] for e in exclusions}
    assert reasons <= {"not_binary", "not_resolved", "missing_field", "bad_outcome"}


def test_nothing_silently_dropped():
    import pandas as pd
    total_rows = len(pd.read_csv(FIX))
    records, exclusions = load_dataset_csv(str(FIX))
    assert len(records) + len(exclusions) == total_rows
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_loader.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write the implementation** (skeleton below; `COLUMN_MAP` and `_parse_row` adapt to the inspected columns — keep the return contract and exclusion reasons exactly as tested)

`data/dataset_loader.py`:
```python
"""Bulk metadata ingestion. Primary: third-party dataset CSV. Fallback: Gamma pagination.
Branch taken (fill in at build time): <dataset|gamma>, because <reason>."""
from __future__ import annotations
import json
import pandas as pd
from data.schema import MarketRecord, Exclusion
from data.api_client import PolymarketClient

COLUMN_MAP = {
    # ours -> theirs (ADJUST after Step 1 inspection)
    "market_id": "condition_id",
    "category": "category",
    "created_ts": "created_time",
    "resolved_ts": "closed_time",
    "outcome": "outcome",
    "volume": "volume",
}


def _row_to_record(row: dict) -> MarketRecord | Exclusion:
    mid = str(row.get(COLUMN_MAP["market_id"], "unknown"))
    for ours, theirs in COLUMN_MAP.items():
        if theirs not in row or pd.isna(row[theirs]):
            return Exclusion(market_id=mid, reason=f"missing_field:{ours}")
    outcome = str(row[COLUMN_MAP["outcome"]]).strip().upper()
    if outcome not in ("YES", "NO"):
        return Exclusion(market_id=mid, reason="bad_outcome")
    try:
        return MarketRecord(
            market_id=mid,
            category=str(row[COLUMN_MAP["category"]]).lower(),
            created_ts=int(pd.Timestamp(row[COLUMN_MAP["created_ts"]]).timestamp()),
            resolved_ts=int(pd.Timestamp(row[COLUMN_MAP["resolved_ts"]]).timestamp()),
            resolved_outcome=outcome,
            volume=float(row[COLUMN_MAP["volume"]]),
        )
    except (ValueError, TypeError) as exc:
        return Exclusion(market_id=mid, reason=f"missing_field:{exc.__class__.__name__}")


def load_dataset_csv(path: str) -> tuple[list[MarketRecord], list[Exclusion]]:
    df = pd.read_csv(path)
    records, exclusions = [], []
    for row in df.to_dict(orient="records"):
        r = _row_to_record(row)
        (records if isinstance(r, MarketRecord) else exclusions).append(r)
    return records, exclusions


def _gamma_to_record(m: dict) -> MarketRecord | Exclusion:
    mid = str(m.get("conditionId", "unknown"))
    outcomes = json.loads(m.get("outcomes", "[]") or "[]")
    if [o.lower() for o in outcomes] != ["yes", "no"]:
        return Exclusion(market_id=mid, reason="not_binary")
    if not m.get("closed"):
        return Exclusion(market_id=mid, reason="not_resolved")
    prices = json.loads(m.get("outcomePrices", "[]") or "[]")
    if len(prices) != 2 or {float(prices[0]), float(prices[1])} != {0.0, 1.0}:
        return Exclusion(market_id=mid, reason="bad_outcome")
    try:
        return MarketRecord(
            market_id=mid,
            category=str(m.get("category", "unknown")).lower(),
            created_ts=int(pd.Timestamp(m["createdAt"]).timestamp()),
            resolved_ts=int(pd.Timestamp(m["closedTime"]).timestamp()),
            resolved_outcome="YES" if float(prices[0]) == 1.0 else "NO",
            volume=float(m.get("volumeNum", 0.0)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        return Exclusion(market_id=mid, reason=f"missing_field:{exc.__class__.__name__}")


def load_from_gamma(client: PolymarketClient,
                    max_markets: int | None = None) -> tuple[list[MarketRecord], list[Exclusion]]:
    records, exclusions, offset = [], [], 0
    while True:
        page = client.fetch_markets_page(offset=offset)
        if not page:
            break
        for m in page:
            r = _gamma_to_record(m)
            (records if isinstance(r, MarketRecord) else exclusions).append(r)
        offset += len(page)
        if max_markets and len(records) >= max_markets:
            break
    return records, exclusions
```

- [ ] **Step 5: Run tests, verify pass**

Run: `python -m pytest tests/test_dataset_loader.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: dataset loader with Gamma fallback and exclusion accounting"
```

---

### Task 12: Cross-validation

**Files:**
- Create: `data/crossval.py`
- Test: `tests/test_crossval.py`

**Interfaces:**
- Consumes: `MarketRecord`, `PolymarketClient`.
- Produces: `cross_validate(records: list[MarketRecord], client: PolymarketClient, n_sample: int = 100, seed: int = 0, tolerance_ts: int = 6 * 3600) -> dict` — seeded random sample (without replacement, capped at len(records)); for each sampled market, fetch Gamma metadata and compare `resolved_outcome` and `resolved_ts` (within `tolerance_ts`). Returns `{"n_checked": int, "n_outcome_match": int, "n_ts_match": int, "n_api_errors": int, "agreement_pct": float, "mismatches": [{"market_id", "field", "ours", "theirs"}]}`. `agreement_pct` = outcome matches / successfully-checked. API errors are counted, not fatal.

- [ ] **Step 1: Write the failing test**

`tests/test_crossval.py`:
```python
from data.schema import MarketRecord
from data.crossval import cross_validate


class StubClient:
    def __init__(self, outcome="YES", closed_time="2023-11-15 12:00:00+00", fail=False):
        self.outcome, self.closed_time, self.fail = outcome, closed_time, fail

    def get_market(self, condition_id):
        if self.fail:
            raise RuntimeError("api down")
        return {"conditionId": condition_id,
                "outcomes": "[\"Yes\", \"No\"]",
                "outcomePrices": "[\"1\", \"0\"]" if self.outcome == "YES" else "[\"0\", \"1\"]",
                "closedTime": self.closed_time}


def rec(mid="0xabc", outcome="YES"):
    return MarketRecord(market_id=mid, category="politics", created_ts=1690000000,
                        resolved_ts=1700049600, resolved_outcome=outcome, volume=1000.0)
    # 1700049600 == 2023-11-15 12:00:00 UTC


def test_full_agreement():
    out = cross_validate([rec()], StubClient(), n_sample=1, seed=0)
    assert out["n_checked"] == 1 and out["agreement_pct"] == 100.0
    assert out["mismatches"] == []


def test_outcome_mismatch_recorded():
    out = cross_validate([rec(outcome="NO")], StubClient(outcome="YES"), n_sample=1, seed=0)
    assert out["agreement_pct"] == 0.0
    assert out["mismatches"][0]["field"] == "resolved_outcome"


def test_api_errors_counted_not_fatal():
    out = cross_validate([rec()], StubClient(fail=True), n_sample=1, seed=0)
    assert out["n_api_errors"] == 1 and out["n_checked"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_crossval.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

`data/crossval.py`:
```python
"""Dataset-vs-official-API agreement check (spec §4). Agreement % is reported
in the WRITEUP regardless of outcome."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from data.schema import MarketRecord


def _api_outcome(market: dict) -> str | None:
    prices = json.loads(market.get("outcomePrices", "[]") or "[]")
    if len(prices) != 2:
        return None
    return "YES" if float(prices[0]) == 1.0 else "NO"


def cross_validate(records: list[MarketRecord], client, n_sample: int = 100,
                   seed: int = 0, tolerance_ts: int = 6 * 3600) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(n_sample, len(records)), replace=False)
    n_checked = n_outcome = n_ts = n_err = 0
    mismatches = []
    for i in idx:
        r = records[int(i)]
        try:
            m = client.get_market(r.market_id)
        except Exception:
            n_err += 1
            continue
        n_checked += 1
        api_out = _api_outcome(m)
        if api_out == r.resolved_outcome:
            n_outcome += 1
        else:
            mismatches.append({"market_id": r.market_id, "field": "resolved_outcome",
                               "ours": r.resolved_outcome, "theirs": api_out})
        api_ts = int(pd.Timestamp(m["closedTime"]).timestamp()) if m.get("closedTime") else None
        if api_ts is not None and abs(api_ts - r.resolved_ts) <= tolerance_ts:
            n_ts += 1
        else:
            mismatches.append({"market_id": r.market_id, "field": "resolved_ts",
                               "ours": r.resolved_ts, "theirs": api_ts})
    pct = (100.0 * n_outcome / n_checked) if n_checked else 0.0
    return {"n_checked": n_checked, "n_outcome_match": n_outcome, "n_ts_match": n_ts,
            "n_api_errors": n_err, "agreement_pct": pct, "mismatches": mismatches}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_crossval.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: dataset-vs-API cross-validation with agreement reporting"
```

---

### Task 13: Backtest pipeline (`run_backtest.py`)

**Files:**
- Create: `run_backtest.py`
- Test: `tests/test_run_backtest.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `calibrate_volume_floor(records: list[MarketRecord]) -> tuple[float, int]` — the largest floor in `[100_000, 50_000, 10_000, 5_000, 1_000, 0]` that keeps ≥ 2,000 markets; returns `(floor, n_kept)`. (With < 2,000 total records it returns `(0, len(records))` — floor 0 keeps everything.)
  - `main()` pipeline: load metadata (dataset or Gamma per Task 11's branch) → calibrate floor → filter → fetch each market's snapshot-window candles via `client.get_price_history(token, resolved_ts - 48*3600, resolved_ts - 24*3600 + 3600)` (needs the market's YES `clobTokenIds[0]` from Gamma metadata; markets whose token id can't be resolved are excluded with reason `"no_token_id"`) → `run_backtest` → write `results/results.json` (list of BetResult dicts + run config: floor, n_markets, n_excluded, timestamp) and `results/exclusions.csv`, then run `cross_validate` and write `results/crossval_report.json`.
  - CLI: `python run_backtest.py [--max-markets N]` (the cap is for smoke runs; full run omits it).

- [ ] **Step 1: Write the failing test** (pipeline glue is tested via the floor calibration + an end-to-end run over stubbed client/records)

`tests/test_run_backtest.py`:
```python
import json
from data.schema import MarketRecord
from run_backtest import calibrate_volume_floor, run_pipeline


def rec(mid, vol, res_ts=1_700_000_000):
    return MarketRecord(market_id=mid, category="politics", created_ts=res_ts - 90 * 86400,
                        resolved_ts=res_ts, resolved_outcome="YES", volume=vol)


def test_floor_keeps_at_least_2000_when_possible():
    records = [rec(str(i), 2000.0) for i in range(2500)]
    floor, kept = calibrate_volume_floor(records)
    assert floor == 1000 and kept == 2500


def test_floor_falls_to_zero_on_small_samples():
    records = [rec(str(i), 500.0) for i in range(100)]
    floor, kept = calibrate_volume_floor(records)
    assert floor == 0 and kept == 100


class StubClient:
    def get_price_history(self, token, start, end):
        from data.schema import Candle
        return [Candle(t=start + 3600, price_yes=0.8)]
    def get_market(self, condition_id):
        return {"conditionId": condition_id, "clobTokenIds": "[\"tok1\", \"tok2\"]",
                "outcomes": "[\"Yes\", \"No\"]", "outcomePrices": "[\"1\", \"0\"]",
                "closedTime": "2023-11-14 22:13:20+00"}


def test_pipeline_writes_results(tmp_path):
    records = [rec("0x1", 5000.0), rec("0x2", 5000.0)]
    out = run_pipeline(records, StubClient(), results_dir=str(tmp_path), n_crossval=1)
    results = json.loads((tmp_path / "results.json").read_text())
    assert len(results["bets"]) == 2
    assert results["config"]["volume_floor"] == 0
    assert (tmp_path / "exclusions.csv").exists()
    assert (tmp_path / "crossval_report.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_backtest.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the implementation**

`run_backtest.py`:
```python
"""End-to-end backtest pipeline. Full run: python run_backtest.py
Smoke run: python run_backtest.py --max-markets 50"""
from __future__ import annotations
import argparse
import csv
import json
import pathlib
import time
from data.schema import MarketRecord, Exclusion
from data.api_client import PolymarketClient
from data.crossval import cross_validate
from backtest.engine import run_backtest

FLOORS = [100_000, 50_000, 10_000, 5_000, 1_000, 0]
TARGET_N = 2_000
H = 3600


def calibrate_volume_floor(records: list[MarketRecord]) -> tuple[float, int]:
    for floor in FLOORS:
        kept = sum(1 for r in records if r.volume >= floor)
        if kept >= TARGET_N:
            return float(floor), kept
    return 0.0, len(records)


def _yes_token_id(client, market_id: str) -> str | None:
    try:
        m = client.get_market(market_id)
        tokens = json.loads(m.get("clobTokenIds", "[]") or "[]")
        return tokens[0] if tokens else None
    except Exception:
        return None


def run_pipeline(records: list[MarketRecord], client, results_dir: str = "results",
                 n_crossval: int = 100) -> dict:
    out_dir = pathlib.Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    floor, kept = calibrate_volume_floor(records)
    filtered = [r for r in records if r.volume >= floor]

    pairs, exclusions = [], []
    for r in filtered:
        token = _yes_token_id(client, r.market_id)
        if token is None:
            exclusions.append(Exclusion(market_id=r.market_id, reason="no_token_id"))
            continue
        candles = client.get_price_history(token, r.resolved_ts - 48 * H,
                                           r.resolved_ts - 24 * H + H)
        pairs.append((r, candles))

    bets, engine_exclusions = run_backtest(pairs)
    exclusions.extend(engine_exclusions)

    payload = {"config": {"volume_floor": floor, "n_metadata": len(records),
                          "n_after_floor": kept, "n_bets": len(bets),
                          "n_excluded": len(exclusions),
                          "run_ts": int(time.time())},
               "bets": [b.model_dump() for b in bets]}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))

    with open(out_dir / "exclusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["market_id", "reason"])
        for e in exclusions:
            w.writerow([e.market_id, e.reason])

    report = cross_validate(filtered, client, n_sample=n_crossval)
    (out_dir / "crossval_report.json").write_text(json.dumps(report, indent=1))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-markets", type=int, default=None)
    args = ap.parse_args()
    client = PolymarketClient()
    # Metadata source per Task 11's branch decision:
    from data.dataset_loader import load_from_gamma
    records, load_exclusions = load_from_gamma(client, max_markets=args.max_markets)
    print(f"metadata: {len(records)} records, {len(load_exclusions)} load exclusions")
    payload = run_pipeline(records, client)
    print(json.dumps(payload["config"], indent=2))


if __name__ == "__main__":
    main()
```

(If Task 11 chose the dataset branch, swap the `main()` metadata source to `load_dataset_csv` with the downloaded CSV path — `run_pipeline` is source-agnostic either way.)

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_run_backtest.py -v`
Expected: 3 PASS

- [ ] **Step 5: Smoke-run the real pipeline (network)**

Run: `python run_backtest.py --max-markets 50`
Expected: prints metadata counts + config JSON; `results/results.json`, `results/exclusions.csv`, `results/crossval_report.json` all exist and are well-formed. Fix any real-API shape mismatches now (update fixtures + parsers together).

- [ ] **Step 6: Full run (network, long)**

Run: `python run_backtest.py`
Expected: completes (may take 1–2h+ from API throttling on first run; the disk cache makes re-runs fast). Record `config` counts. **Sanity checks before proceeding:** `n_bets` in the thousands; exclusion reasons distribution looks sane (spot-read `exclusions.csv`); `crossval_report.json` agreement_pct ≥ 95 — if below, STOP and investigate the data source before running MC (spec §9 data-quality blocker).

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: end-to-end backtest pipeline with volume-floor calibration"
```

---

### Task 14: MC runner + verdict (`run_mc.py`)

**Files:**
- Create: `run_mc.py`
- Test: `tests/test_run_mc.py`

**Interfaces:**
- Consumes: `results/results.json` (Task 13), `bootstrap_roi`/`ci`, `random_side_null`, `concentration_check`.
- Produces: `results/mc_results.json` with keys: `observed` (roi, win_rate, total_pnl, max_drawdown, n_bets), `bootstrap_ci`, `null` (p_value, null mean), `concentration` (full output), `fee_sensitivity` (observed ROI recomputed at 0x and 2x fees), `gate` (`{"ci_lower_positive": bool, "not_concentrated": bool, "verdict": "PROFITABLE"|"NOT PROVEN"}`), plus `coinflip_slice` (n and ROI of is_coinflip bets, diagnostic only).
- `verdict(gate_inputs) -> str`: "PROFITABLE" iff `ci_lower_positive AND not_concentrated`, else "NOT PROVEN". The pre-registered gate — never modified to fit results.

- [ ] **Step 1: Write the failing test**

`tests/test_run_mc.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_mc.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the implementation**

`run_mc.py`:
```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/test_run_mc.py -v`
Expected: 2 PASS

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -v`
Expected: ALL PASS (network deselected)

- [ ] **Step 6: Run MC on the real backtest output**

Run: `python run_mc.py`
Expected: prints gate verdict + CI + p-value; `results/mc_results.json` written. **Record the honest numbers — whatever they are.** Check the ML trigger condition (Global Constraints): if borderline/suspicious-subgroup, note it for Alex; do NOT build ML.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: MC runner with pre-registered gate verdict and fee sensitivity"
```

---

### Task 15: README + WRITEUP

**Files:**
- Create: `README.md`, `WRITEUP.md`

**Interfaces:**
- Consumes: `results/results.json`, `results/mc_results.json`, `results/crossval_report.json`, `results/exclusions.csv` — every number in both docs comes from these files, none invented.

- [ ] **Step 1: Write README.md** — sections: one-paragraph summary (question + verdict, from `mc_results.json`); Quick start (`pip install -r requirements.txt`, `python -m pytest`, `python run_backtest.py`, `python run_mc.py`); Project structure (the tree from this plan); link to WRITEUP.md. Keep under 80 lines.

- [ ] **Step 2: Write WRITEUP.md** — recruiter-readable, mirrors the sibling repo's WRITEUP.md conventions. Required sections, all populated from the results files:
  1. **The question** — favourite-longshot bias background, why PolyMarket is an interesting venue (no vig, but spread + taker fee).
  2. **Method** — market universe + volume floor (actual calibrated value + n), snapshot rule, favourite rule, fee model (formula + verification date), flat $1 stake. State the **pre-registered gate verbatim** and that it was fixed before results existed.
  3. **Data quality** — source branch actually taken (dataset vs Gamma), cross-validation agreement % (reported regardless of value), full exclusion accounting table (reason → count, from exclusions.csv).
  4. **Results** — observed ROI/win-rate/n, bootstrap CI, null p-value, concentration table, fee sensitivity (0x/1x/2x), coinflip diagnostic slice.
  5. **Verdict** — the gate output, stated plainly. If NOT PROVEN, say so without hedging or spin — same honesty bar as the FYP writeup.
  6. **Limitations** — 12h+ candle granularity on resolved markets, taker-only fee model, spread not modeled (entry at mid/last rather than ask — flag as optimistic bias), category labels as-reported, any fee-docs verification caveat from Task 2.
  7. **What would change my mind / future work** — ML trigger status (fired or not, per the Global Constraints condition), Kelly sizing, spread modeling.

- [ ] **Step 3: Verify every number in the docs against the results files** — grep each figure cited in WRITEUP.md and confirm it appears in the corresponding results/*.json. No unsourced numbers.

- [ ] **Step 4: Commit**

```bash
git add README.md WRITEUP.md && git commit -m "docs: README + honest results writeup"
```

---

### Task 16: Vault project folder

**Files:**
- Create: `C:\Users\Alex\ObsidianVault\claude-memory\16-polymarket-favourites\_INDEX.md`, `PROJECT_OVERVIEW.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `ENV_VARS.md`, `KNOWN_ISSUES.md`
- Modify: `C:\Users\Alex\ObsidianVault\claude-memory\VAULT_INDEX.md` (register the project)
- Modify: `C:\Users\Alex\ObsidianVault\claude-memory\11-ideas\2026-07-08 Ideas for Projects.md` and `11-ideas\_INDEX.md` (item 3 status: not started → in progress/done with link)

**Interfaces:**
- Consumes: the templates at `C:\Users\Alex\ObsidianVault\claude-memory\00-meta\templates\` and this repo's spec/plan/results.

- [ ] **Step 1: Seed the 6-doc set from templates** — follow `00-meta/CONVENTIONS.md`: frontmatter with `project: polymarket-favourites`, `updated:` = today; `_INDEX.md` links the other five + codebase path `C:\Users\Alex\Projects\polymarket-favourite-bias`; `PROJECT_OVERVIEW.md` = the question, method summary, verdict, repo path, LOCAL-ONLY status; `ARCHITECTURE.md` = the module tree + data flow from this plan; `DECISIONS.md` = dated entries for every locked decision (binary-only universe, 24-48h snapshot + tie-break rule, flat $1 stake, hybrid data sourcing + which branch was taken, fee formula + verification date, volume floor calibrated value, pre-registered gate, coinflip tie-break, ML trigger fired/not); `ENV_VARS.md` = "none — public read-only APIs, no keys"; `KNOWN_ISSUES.md` = the Limitations list from WRITEUP.md.

- [ ] **Step 2: Register in VAULT_INDEX.md** — add row 16 to the folder table + an Active-projects entry (status: results in, local-only pending publish decision).

- [ ] **Step 3: Update the ideas tracker** — item 3 in the 2026-07-08 ideas note + the `_INDEX.md` status-snapshot table: not started → done (research study built; verdict recorded; publish pending Alex's go).

- [ ] **Step 4: Final repo commit** (spec/plan checkboxes updated)

```bash
git add -A && git commit -m "chore: mark plan complete"
```

---

## Completion

After Task 16: report to Alex — the honest verdict (gate output + key numbers), the cross-validation agreement %, whether the ML trigger condition fired, and that the repo is local-only awaiting his publish decision. Do NOT push anywhere.
