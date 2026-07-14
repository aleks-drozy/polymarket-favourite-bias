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
