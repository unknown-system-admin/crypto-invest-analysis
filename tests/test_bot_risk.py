from bot.risk import (RiskDecision, check_daily_loss, check_stop_loss,
                      check_position_limit, position_notional,
                      trailing_stop_price, check_trailing_stop)


def test_daily_loss_allows_within_limit():
    d = check_daily_loss(10000.0, 9500.0, 0.10)
    assert d.allowed is True


def test_daily_loss_blocks_at_limit():
    d = check_daily_loss(10000.0, 8990.0, 0.10)  # -10.1%
    assert d.allowed is False
    assert "daily loss" in d.reason


def test_stop_loss_long_fires():
    assert check_stop_loss(100.0, 97.0, "long", 0.025) is True   # -3%
    assert check_stop_loss(100.0, 98.5, "long", 0.025) is False  # -1.5%


def test_stop_loss_short_fires():
    assert check_stop_loss(100.0, 103.0, "short", 0.025) is True  # +3% against short
    assert check_stop_loss(100.0, 101.0, "short", 0.025) is False


def test_position_notional_matches_config():
    assert position_notional(10000.0, 0.20, 5) == 10000.0


def test_position_limit_per_symbol():
    d = check_position_limit(["BTC/USDT:USDT"], "BTC/USDT:USDT", max_per_symbol=1)
    assert d.allowed is False
    d2 = check_position_limit(["SOL/USDT:USDT"], "BTC/USDT:USDT", max_per_symbol=1)
    assert d2.allowed is True


def test_trailing_stop_price_long_uses_peak():
    stop = trailing_stop_price(100.0, 110.0, 2.0, "long", 2.5, 0.025)
    assert stop == 105.0  # max(110-5=105, 97.5)


def test_trailing_stop_price_long_floor_hard_stop():
    stop = trailing_stop_price(100.0, 100.0, 2.0, "long", 2.5, 0.025)
    assert stop == 97.5


def test_trailing_stop_price_short_uses_peak():
    stop = trailing_stop_price(100.0, 90.0, 2.0, "short", 2.5, 0.025)
    assert stop == 95.0  # min(90+5=95, 102.5)


def test_check_trailing_stop_long_fires_below_stop():
    assert check_trailing_stop(100.0, 110.0, 104.5, 2.0, "long", 2.5, 0.025) is True
    assert check_trailing_stop(100.0, 110.0, 105.5, 2.0, "long", 2.5, 0.025) is False