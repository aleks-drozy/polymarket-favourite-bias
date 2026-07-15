"""Gate part 2 (spec §6): the edge must not be concentrated in one category
or time window. Leave-one-group-out bootstrap CIs."""
from __future__ import annotations
from datetime import datetime, timezone
from backtest.engine import BetResult
from mc.bootstrap import bootstrap_roi, ci

MIN_GROUP_FRACTION = 0.02

# A leave-one-group-out check is only informative if excluding the group
# leaves a residual sample large enough for a bootstrap CI to mean anything.
# 30 is the conventional minimum sample size; chosen before any Monte Carlo
# results existed, prompted by the real backtest sample being ~99.8%
# category="unknown" (Gamma stores no category for this era), which would
# otherwise leave a ~5-bet residual and let bootstrap noise masquerade as a
# "concentrated" finding.
MIN_LOO_RESIDUAL = 30


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

    out_groups, skipped, skipped_uninformative = {}, [], []
    concentrated = False
    for name, idx in sorted(groups.items()):
        if len(idx) < MIN_GROUP_FRACTION * n:
            skipped.append(name)
            continue
        if n - len(idx) < MIN_LOO_RESIDUAL:
            skipped_uninformative.append(name)
            continue
        loo = _loo_ci(results, idx, n_sims, seed)
        out_groups[name] = {"n_excluded": len(idx), "ci": loo}
        if full["lower"] > 0 and loo["lower"] <= 0:
            concentrated = True
    return {"full_ci": full, "groups": out_groups,
            "skipped_small_groups": skipped,
            "skipped_uninformative": skipped_uninformative,
            "concentrated": concentrated}
