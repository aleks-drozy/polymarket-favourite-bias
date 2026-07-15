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
