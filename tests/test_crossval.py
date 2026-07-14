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
