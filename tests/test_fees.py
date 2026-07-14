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
