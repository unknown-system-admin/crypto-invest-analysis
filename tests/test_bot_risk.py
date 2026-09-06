from bot.risk import (RiskDecision, check_daily_loss, check_stop_loss,
                      check_position_limit, position_notional)


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