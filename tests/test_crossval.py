from data.schema import MarketRecord
from data.crossval import cross_validate


class StubClient:
    def __init__(self, outcome="YES", closed_time="2023-11-15 12:00:00+00", fail=False,
                 outcome_prices=None):
        self.outcome = outcome
        self.closed_time = closed_time
        self.fail = fail
        self.outcome_prices = outcome_prices

    def get_market(self, condition_id):
        if self.fail:
            raise RuntimeError("api down")
        if self.outcome_prices is not None:
            prices_json = self.outcome_prices
        else:
            prices_json = "[\"1\", \"0\"]" if self.outcome == "YES" else "[\"0\", \"1\"]"
        return {"conditionId": condition_id,
                "outcomes": "[\"Yes\", \"No\"]",
                "outcomePrices": prices_json,
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


def test_near_one_settlement_still_agrees():
    # Real Gamma settled markets show prices like 0.9995, not exactly 1.0.
    # The threshold-consistent comparison should still agree.
    out = cross_validate(
        [rec(outcome="YES")],
        StubClient(outcome_prices='["0.9995", "0.0005"]'),
        n_sample=1,
        seed=0
    )
    assert out["n_checked"] == 1
    assert out["agreement_pct"] == 100.0
    assert out["mismatches"] == []


def test_ambiguous_api_prices_recorded_as_mismatch():
    # Ambiguous API prices (e.g., neither side >= 0.999) don't match any outcome.
    # Should be recorded as a mismatch with resolved_outcome field.
    out = cross_validate(
        [rec(outcome="YES")],
        StubClient(outcome_prices='["0.6", "0.4"]'),
        n_sample=1,
        seed=0
    )
    assert out["n_checked"] == 1
    assert out["agreement_pct"] == 0.0
    assert len(out["mismatches"]) == 1
    mismatch = out["mismatches"][0]
    assert mismatch["field"] == "resolved_outcome"
    assert mismatch["ours"] == "YES"
    assert mismatch["theirs"] is None
