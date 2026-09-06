from bot.config import BotConfig


def test_default_config_values():
    cfg = BotConfig()
    assert cfg.symbols == ["BTC/USDT:USDT", "SOL/USDT:USDT"]
    assert cfg.htf_timeframe == "1h"
    assert cfg.ltf_timeframe == "15m"
    assert cfg.ltf_candles == 1440
    assert cfg.leverage == 5
    assert cfg.margin_pct == 0.20
    assert cfg.stop_loss_pct == 0.025
    assert cfg.max_daily_loss_pct == 0.10
    assert cfg.dry_run is True


def test_config_position_notional():
    cfg = BotConfig(margin_pct=0.20, leverage=5)
    # 20% margin x 5x leverage on $10k equity = $10k notional
    assert cfg.position_notional(10000.0) == 10000.0


def test_config_overridable():
    cfg = BotConfig(leverage=3, stop_loss_pct=0.02)
    assert cfg.leverage == 3
    assert cfg.stop_loss_pct == 0.02


def test_signal_v2_defaults():
    cfg = BotConfig()
    assert cfg.atr_stop_mult == 2.5
    assert cfg.min_atr_pct == 0.0015


def test_meanreversion_config_defaults():
    cfg = BotConfig()
    assert cfg.signal_mode == "momentum"   # default stays momentum
    assert cfg.rsi_oversold == 25
    assert cfg.rsi_overbought == 75


def test_funding_gate_default_disabled():
    cfg = BotConfig()
    assert cfg.funding_threshold == 0.0   # 0.0 = disabled (calibration: funding never crossed contrarian threshold)