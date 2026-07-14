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
